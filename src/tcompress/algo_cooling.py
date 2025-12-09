import quimb as qu
import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
import cotengrust as ctgr

import numpy as np


from tqdm import tqdm
import itertools

import autoray as ar
import torch
import jax
import jax.numpy as jnp

import re


from time import sleep
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import nlopt
from collections import deque

import logging
logger = logging.getLogger(__name__)


def backend_torch(device = "cpu", dtype = torch.float64, requires_grad=False):
    
    def to_backend(x, device=device, dtype=dtype, requires_grad=requires_grad):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    
    return to_backend

def backend_numpy(dtype=np.float64):
    
    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)
    
    return to_backend


def backend_jax(dtype=jnp.float64, device=jax.devices("cpu")[0]):
    # device = jax.devices("cpu")[0]
    # dtype=jnp.float64
    # def to_backend(x, device=device, dtype=dtype):
    #     return jax.device_put(jnp.array(x, dtype=dtype), device)

    
    def to_backend(x, dtype=dtype, device=device):
        arr = jax.device_put(jnp.array(x, dtype=dtype), device)
        return arr

    return to_backend


def opt_(progbar=True, max_repeats=2**9, optlib="cmaes", max_time="rate:1e8",
         alpha=64, target_size = 2**34, subtree_size=12):


    # high quality: max_time="equil:128",max_repeats=2**10, optlib=nevergrad
    # terminate search if contraction is cheap: "rate:1e9"

    opt = ctg.ReusableHyperOptimizer(
        minimize=f'combo-{alpha}',
        slicing_opts={'target_size': 2**40},         # first do basic slicing
        slicing_reconf_opts={'target_size': target_size},  # then advanced slicing with reconfiguring
        reconf_opts={'subtree_size': subtree_size},            # then finally just higher quality reconfiguring
        max_repeats=max_repeats,
        parallel=True,  # optimize in parallel
        optlib=optlib,  # an efficient parallel meta-optimizer
        hash_method="b",  # most generous cache hits
        directory="cash/",  # cache paths to disk
        progbar=progbar,  # show live progress
        max_time =max_time,
    )
    return opt

def copt_(progbar=True, chi=4, directory=None, max_repeats=2**8):

    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time = "rate:1e8",
        directory=directory,  # cache paths to disk

    )
    return copt

def apply_peps(peps, pepo, flat=False):
    "pepo_2 @ pepo_1"
    peps = peps.copy()
    pepo = pepo.copy()
    
    Lx = peps.Lx
    Ly = peps.Ly
    
    map_inds  = {f"b{i},{j}": f"k{i},{j}" for i in range(Lx) for j in range(Ly)}
    #map_inds_ = {k.replace("b", "k"): v for k, v in map_inds.items()}
    
    #pepo_1.reindex_(map_inds)
    #pepo_2.reindex_(map_inds_)
    tn = peps | pepo
    tn.reindex_( {idx: qtn.rand_uuid() for idx in tn.inner_inds()} )
    tn.reindex_( map_inds )
    
    if flat:
        tn.flatten(fuse_multibonds=True, inplace=True)
        tn.view_as_(qtn.tensor_2d.PEPS, Lx=Lx, Ly=Ly, site_tag_id='I{},{}', x_tag_id='X{}', y_tag_id='Y{}',
                  site_ind_id='k{},{}',
                )
        
    return tn




def apply_pepo(pepo_1, pepo_2, flat=False, tags=[]):
    "pepo_2 @ pepo_1"
    pepo_1 = pepo_1.copy()
    pepo_2 = pepo_2.copy()

    pepo_2.add_tag(tags)
    Lx = pepo_1.Lx
    Ly = pepo_1.Ly
    
    map_inds  = {f"b{i},{j}": qtn.rand_uuid() for i in range(Lx) for j in range(Ly)}
    map_inds_ = {k.replace("b", "k"): v for k, v in map_inds.items()}
    
    pepo_1.reindex_(map_inds)
    pepo_2.reindex_(map_inds_)
    tn = pepo_1 | pepo_2
    tn.reindex_( {idx: qtn.rand_uuid() for idx in tn.inner_inds()} )
    if flat:
        tn.flatten(fuse_multibonds=True, inplace=True)
        tn.view_as_(
            qtn.tensor_2d.PEPO,
            Lx=Lx, 
            Ly=Ly,
            site_tag_id='I{},{}',
            x_tag_id='X{}',
            y_tag_id='Y{}',
            upper_ind_id='k{},{}',
            lower_ind_id='b{},{}',
        )
    
    return tn

def pepo_trotter_ITF(edges, sites, depth=1, cutoff=1.e-10,
                     Lx=4, Ly=4, to_backend=None, params={}):

    sites = sorted({ (site,) for edge in edges for site in edge})

    
    rx   = qtn.circuit.rx_gate_param_gen( [-params[f"rx_depth{depth}"]]  )  
    rzz   = qtn.circuit.rzz_param_gen( [-params[f"rzz_depth{depth}"]]  )  

    
    pepo = pepo_identity(Lx, Ly)
    pepo.apply_to_arrays(to_backend)

    for count, site in enumerate(sites):

        gate_2d(pepo, site, rx, ind_id="b{},{}", site_tags="I{},{}",
                cutoff=cutoff, contract=True, inplace=True)
    
    for count, edge in enumerate(edges):

        gate_2d(pepo, edge, rzz, ind_id="b{},{}", site_tags="I{},{}",
                cutoff=cutoff, contract='reduce-split', inplace=True)
    
    for count, site in enumerate(sites):

        gate_2d(pepo, site, rx, ind_id="b{},{}", site_tags="I{},{}",
                cutoff=cutoff, contract=True, inplace=True)


    return pepo






def gate_1d(tn, where, G, ind_id="k{}", site_tags="I{}",
            cutoff=1.e-10, contract='split-gate', inplace=True):

    if len(where)==2:
        x, y = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(x), ind_id.format(y)], contract=contract, tags=[], info=None, 
                                inplace=inplace,
                                **{"cutoff":cutoff}
                                    )

        # adding site tags
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(x)] ][0]
        t.add_tag(site_tags.format(x))
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(y)] ][0]
        t.add_tag(site_tags.format(y))

    if len(where)==1:
        x, = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(x), ind_id.format(y)], contract=True, tags=[], info=None, 
                                inplace=inplace,
                                **{"cutoff":cutoff}
                                    )

    return tn


def gate_2d(tn, where, G, ind_id="k{},{}", site_tags="I{},{}",
            cutoff=1.-10, contract='split-gate', inplace=True):

    if len(where)==2:
        ((i1, j1), (i2, j2)) = where
        
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(i1, j1), ind_id.format(i2, j2)], 
                                contract=contract, tags=[], info=None, 
                                inplace=inplace,
                                **{"cutoff":cutoff}
                                    )

        # adding site tags
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(i1, j1)] ][0]
        t.add_tag(site_tags.format(i1, j1))
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(i2, j2)] ][0]
        t.add_tag(site_tags.format(i2, j2))

    if len(where)==1:
        ((i, j),) = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(i, j)], 
                                contract=True, tags=[], info=None, 
                                inplace=inplace,
                                **{"cutoff":cutoff}
                                    )

    return tn


            # theta = to_backend_( torch.tensor( -h * delta ).clone().detach() )
            # params[f"u3l_theta_{site}_d{depth}"] = theta
            
            # phi = to_backend_( torch.tensor( -h * delta ).clone().detach() )
            # params[f"u3l_phi_{site}_d{depth}"] = phi

            # gamma = to_backend_( torch.tensor( -h * delta ).clone().detach() )
            # params[f"u3l_gamma_{site}_d{depth}"] = gamma
            # u3 = qtn.circuit.u3_gate_param_gen([phi, theta, gamma])

def skeleten_pepo(params, edges, sites, depth_total=2, contract=False, to_backend=None, Lx=2, Ly=2):

    pepo = pepo_identity(Lx, Ly)
    pepo.apply_to_arrays(to_backend)
   

    for depth in range(depth_total ):
        
        phi = params[f"rx_depth{depth}"]        
        rx = qtn.circuit.rx_gate_param_gen([phi])
        theta = params[f"rzz_depth{depth}"]
        rzz = qtn.circuit.rzz_param_gen([theta])
        
        for count, site in enumerate(sites):
            gate_2d(pepo, site, rx, ind_id="b{},{}", site_tags="I{},{}", contract=True, inplace=True)

        
        for count, edge in enumerate(edges):               
            gate_2d(pepo, edge, rzz, ind_id="b{},{}", site_tags="I{},{}",
                    cutoff=1.e-12, contract=contract, inplace=True)


        for count, site in enumerate(sites):
            gate_2d(pepo, site, rx, ind_id="b{},{}", site_tags="I{},{}", contract=True, inplace=True)

    

    return pepo

def cost_function_su(params, pepo=None, gauges=None, sites=None, edges=None, opt="auto-hq",  
                     Lx=2, Ly=2, stable=True, equalize_norms=False,chi_mps=4,
                     progbar=False, chi=4, renorm=False, equilib=False, bp_cal=False,
                     depth_total=2, cutoff=1.e-12, map_tags_2d=None):

    L = Lx * Ly
    pbar = tqdm(total=depth_total, desc="SU", ncols=100, disable=not progbar, dynamic_ncols=True)
    results = {"t": [], "Bp_norm": [], "Error": [], "cost":None}

    start_time = time.time()
    gauges_ = { u:v*1. for u, v in gauges.items()}
    pepo_ = pepo.copy()


    
    for depth in range(depth_total):
    
        # U_\dagger: notice minus sign
        rx   = qtn.circuit.rx_gate_param_gen( [-params[f"rx_depth{depth}"]]  )  
        rzz   = qtn.circuit.rzz_param_gen( [-params[f"rzz_depth{depth}"]]  )  
    
        # apply RX (two-site) with truncation e(+i X h dt / 2)
        for count, site in enumerate(sites):
            pepo_.gate_simple_(rx, site, gauges=gauges_)
    
        # apply RZZ (two-site) with truncation e(+i ZZ J dt)
        for where in edges:
            pepo_.gate_simple_(rzz, where, gauges=gauges_, 
                              max_bond=chi, cutoff=cutoff, 
                              cutoff_mode='rsum2', renorm=renorm
                             )
            
        # apply RX (two-site) with truncation e(+i X h dt / 2)    
        for count, site in enumerate(sites):
            pepo_.gate_simple_(rx, site, gauges=gauges_ )
    
    
        # (optional) equilibrate gauges
        if equilib:
            # ensure gauge is equilibrated
            pepo_.gauge_all_simple_(max_iterations=100, tol=1e-6, gauges=gauges, progbar=False,)
    
    
        if bp_cal:
            # 6) build gauged copy for norm/BP
            pepo_bp = pepo_.copy()
            pepo_bp.gauge_simple_insert(gauges_)
    
            # boundary propagation norm estimate
            bp = L2BP(pepo_bp, optimize=opt, site_tags=site_tags)
            bp.run(max_iterations=2_000, tol=1.e-7, progbar=False, diis=True)
            est_norm = complex(bp.contract()).real
            # bpnorm = complex(est_norm * 10**(2*pepo.exponent)).real / 2**L
            
            # stable norm: 
            log_val = np.log(est_norm) + (2 * complex(pepo.exponent).real * np.log(10)) - (L * np.log(2))
            bpnorm = np.exp(log_val)
    
            results.setdefault("Bp_norm", []).append(bpnorm)
            results.setdefault("Error", []).append(1  -  (np.log(bpnorm) / L))
    
    
        
        # optional: update progress (guard against None/unused vars)
        pbar.set_postfix({
            "depth":depth,
            "Bp_norm":  "—" if not len(results["Bp_norm"]) else round(bpnorm, 4),
            "Error":  "—" if not len(results["Error"]) else round(bpnorm, 4),
    
        })
        pbar.update(1)
    
    pbar.close()
    # print( "su", (time.time() - start_time)  )
    start_time = time.time()
    pepo_.gauge_simple_insert(gauges_)
    pepo_.retag_(map_tags_2d)
    pepo_ = trace_2d(pepo_, Lx, Ly)
    pepo_.contract_boundary_( max_bond=chi_mps, final_contract = True, 
                                   final_contract_opts={"optimize":opt}, 
                                   max_separation = 1, cutoff=cutoff,
                                   sequence = ['xmin', 'xmax', "ymin", "ymax" ], 
                                   equalize_norms=equalize_norms, progbar=progbar,
                                   )    

    # print( "bmps",  (time.time() - start_time)  )
    
    if stable:
        trace_, exponent = pepo_.contract(all, optimize=opt, strip_exponent=True)
        log_ratio = ar.do("log", abs(trace_)) + exponent * ar.do("log",10) - L * ar.do("log",2)
        ratio =  ar.do("exp",log_ratio)  # = abs(val) / 2**L, stably
        cost = 1 - ratio**2
    else:
        overlap = pepo_.contract(all, optimize=opt, strip_exponent=False)
        cost = 1 - ( ar.do("abs", overlap) / 2**L )**2

    results["cost"] = cost
    return cost


def cost_function(pepo_fix, params, sites, edges, opt,  Lx=2, Ly=2, depth_total=2, to_backend=None):
    
    L = Lx * Ly
    pepo = skeleten_pepo(params, edges, sites, depth_total=depth_total, 
                         to_backend=to_backend, Lx=Lx, Ly=Ly)
               

    
    tn = (pepo.H | pepo_fix)
    tn.full_simplify_(seq='R', output_inds={}, split_method='svd', inplace=True)
    overlap = tn.contract(all, optimize=opt)
    
    return 1 - (( ar.do("abs", overlap) / 2**L )**2)


def pepo_identity(Lx, Ly, dtype="complex128"):
    pepo = qtn.PEPO.rand(Lx=Lx, Ly=Ly, bond_dim=1, seed=666, dtype=dtype)        
    I = qu.pauli('I', dtype=dtype)
    for t in pepo:
        if len(t.data.shape) == 4:
            W = np.zeros([1,1,2,2], dtype=dtype)
            W[0,0,:,:] = I
            t.modify(data = W)
        if len(t.data.shape) == 5:
            W = np.zeros([1,1,1,2,2], dtype=dtype)
            W[0,0,0,:,:] = I
            t.modify(data = W)
        if len(t.data.shape) == 6:
            W = np.zeros([1,1,1,1,2,2], dtype=dtype)
            W[0,0,0,0,:,:] = I
            t.modify(data = W)
    return pepo



def gate_fidelity(U, U_approx):
    """
    Compute average gate fidelity and entanglement fidelity between two matrices U and U_approx.
    Normalization is applied automatically.
    """
    U = np.asarray(U, dtype=np.complex128)
    U_approx = np.asarray(U_approx, dtype=np.complex128)
    d = U.shape[0]

    # Normalize both matrices to unitary scale (by Frobenius norm)
    U = U / np.sqrt(np.trace(U.conj().T @ U))
    U_approx = U_approx / np.sqrt(np.trace(U_approx.conj().T @ U_approx))

    # Entanglement fidelity
    overlap = np.trace(U.conj().T @ U_approx)
    F_ent = (np.abs(overlap) ** 2)

    return  float(np.real_if_close(F_ent))

def internal_inds(psi):
    open_inds = psi.outer_inds()
    innre_inds = []
    for t in psi:
        t_list = list(t.inds)
        for j in t_list :
            if j not in open_inds:
                innre_inds.append(j)
    return innre_inds

# def fidel_mps(psi, psi_fix, opt):
#     val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
#     val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
#     val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))
#     val_1 = val_1 ** 2
    
#     return  val_1 / (val_0 * val_) 


def fidel_mps_normalized(psi, psi_fix, opt,  cur_orthog=None):
    tn = psi.H & psi_fix
    val_1 = abs(tn.contract(all, optimize=opt))
    return  val_1 ** 2





def rand_uni(n, to_backend=None, dtype=torch.complex64, requires_grad_=True, device="cpu", seed=2 ):
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Step 1: Make a complex-valued random matrix
    A = torch.randn(n, n, dtype=dtype, device=device)
    
    # Step 2: QR decomposition to get unitary matrix
    Q, R = torch.linalg.qr(A)
    
    # Step 3: Normalize to ensure det(Q) = 1 (optional for SU(n))
    # Adjust sign to ensure unitarity (optional for full unitarity)
    d = torch.diagonal(R)
    Q = Q * (d / torch.abs(d)).conj()

    # Step 4: Enable gradient tracking
    if to_backend:
        Q = to_backend(Q)
        Q = Q.clone().detach().requires_grad_(requires_grad_)
        return Q
    else:
        Q = Q.clone().detach().requires_grad_(requires_grad_)
        return Q




def fidel_mps(psi, psi_fix):

    opt = opt_(progbar=False)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1 ** 2
    f = complex(val_1 / (val_0 * val_) ).real
    return  f

def trace_2d(pepo, Lx, Ly):

    for count, t in enumerate(pepo):
        l = t.inds
        k_ind = [i for i in l if i.startswith("k")]
        b_ind = [i for i in l if i.startswith("b")]
        t.trace(k_ind, b_ind, preserve_tensor=False, inplace=True)
    pepo.view_as_(
        qtn.tensor_2d.TensorNetwork2DFlat,
        Lx=Lx,
        Ly=Ly,
        site_tag_id='I{},{}',
        x_tag_id='X{}',
        y_tag_id='Y{}',
    )
    return pepo



class FIT:
    """
    Fidelity Fitting for tensor networks.

    Parameters
    ----------
    tn : TensorNetwork
        Target tensor network to fit.
    p0 : TensorNetwork, optional
        Initial MPS (starting state). Must support `.copy()` and `.canonize()`.
    cutoffs : float, optional
        Numerical cutoff for truncation (default: 1e-9).
    backend : str or None, optional
        Backend specification for tensor operations.
    n_iter : int, optional
        Number of optimization iterations (default: 4).
    verbose : bool, optional
        If True, logs fidelity at each iteration.
    re_tag : bool, default=True
        If True, (re)tag the target TN for environment construction.
    """

    def __init__(self, tn, p=None, cutoffs=1.e-10, backend=None, 
                 site_tag_id="I{}", opt = "auto-hq", range_int=[],
                 re_tag=False, info={}, warning=True):

        if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
            if warning:
                logger.warning("No initial MPS `p` provided. FIT requires an initial state for fitting.")        
        
        self.L = len(p.tensor_map.keys())
        
        self.p = p.copy() if p is not None else None
        if site_tag_id:
            self.p.view_as_(qtn.MatrixProductState, L = self.L, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
        
        
        
        self.site_tag_id = site_tag_id
        self.tn = tn.copy()
        self.opt = opt
        self.cutoffs = cutoffs
        self.backend = backend
        self.loss = []
        self.info = info
        self.range_int = range_int

        # Reindex tensor network with random UUIDs for internal indices
        self.tn.reindex_( {idx: qtn.rand_uuid() for idx in self.tn.inner_inds()} )



        
        if set(self.tn.outer_inds()) != set(self.p.outer_inds()):
            if warning:
                logger.warning("tn & p contains different inds ")        

        
        # re_new tags of tn to be used for effective envs:
        if re_tag:
            self._re_tag()


    def visual(self, figsize=(14, 14), layout="neato", show_tags=False, tags_=[], show_inds=False):
        # Visualize network with MPS
        tags = [  self.site_tag_id.format(i)  for i in range(self.L)] + tags_
        return (self.tn & self.p).draw(tags, legend=False, show_inds=show_inds,
                                 show_tags=show_tags, figsize=figsize, node_outline_darkness=0.1, 
                                       node_outline_size=None, highlight_inds_color="darkred",
                                      edge_scale=2.0, layout=layout,refine_layout="auto",
                                      highlight_inds=self.p.outer_inds(),
                                      )

    
    # -------------------------
    # Tagging methods
    # -------------------------
    def _deep_tag(self):
        """
        Propagates tags through the tensor network to ensure every tensor
        receives at least one site tag. Useful for layered TNs.
        """
        tn = self.tn
        count = 1

        while count >= 1:
            tags = tn.tags
            count = 0
            for tag in tags:
                tids = tn.tag_map[tag]
                neighbors = qtn.oset()
                for tid in tids:
                    t = tn.tensor_map[tid]
                    for ix in t.inds:
                        neighbors |= tn.ind_map[ix]
                for tid in neighbors:
                    t = tn.tensor_map[tid]
                    if not t.tags:
                        t.add_tag(tag)
                        count += 1

    def _re_tag(self):
        
        # drop tags
        tn = self.tn
        tn.drop_tags()

        # get outer inds and all tags
        p = self.p
        site_tags = [ self.site_tag_id.format(i) for i in range(p.L)   ]
        inds = list(p.outer_inds())
        

        # smart tagging for the first layer: meaning each tensor in tn is connected directly to p's tensors
        for site_tag in site_tags:
            indx = [i for i in p[site_tag].inds if i in inds][0]
            
            t = [tn.tensor_map[tid] for tid in tn.ind_map[indx]][0]
            
            if not t.tags:
                t.add_tag(site_tag)
                


        if len(tn.tensor_map.keys()) != len(tn.tags):
            if warning:
                logger.warning("Missing tags in the tensor network — it’s probably a layered TN.") 
            self._deep_tag()

            
    def run(self, n_iter=6, verbose=True):
        
        """Run the fitting process."""
        if self.p is None:
            raise ValueError("Initial state `p0` must be provided.")

        psi = self.p
        L = self.L        
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        for iteration in range(n_iter):            
            for site in range(L):
                
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1

                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                
                psi_h = psi.H.select([site_tag_id.format(site)], "!any")
                tn_ = psi_h | self.tn


                # Contract and normalize
                f = tn_.contract(all, optimize=opt)
                f = f.transpose(*psi[site].inds)

                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))

    def _build_env_right(self, psi, env_right):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        # iterate from rightmost to leftmost
        for i in reversed(range(L)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block 

                
            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                # tie to previously computed right environment
                t |= env_right[site_tag_id.format(i+1)]
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)




    def _right_range(self, psi, env_right, start, stop):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        # iterate from rightmost to leftmost
        # for i in reversed(range(L)):
        for count, i in enumerate(range(stop, start, -1)):
            
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            # Is there any tensor in tn to be included in env
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block 

                
            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                
                if count==0:
                    indx = psi.bond(stop+1, stop)
                    indx_ = self.tn.bond(stop+1, stop)

                    
                # tie to previously computed right environment
                if env_right[site_tag_id.format(i+1)] is not None:
                    t |= env_right[site_tag_id.format(i+1)]
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
                else:
                    t = t.reindex( {indx:indx_} ) 
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)

    def _left_range(self, psi, site, count, env_left):
        """Update left environment incrementally for current site."""

        # get tensor at stie from p
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block 
            
        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            if count - 1 == 0:
                indx = psi.bond(site-1, site)
                indx_ = self.tn.bond(site-1, site)
                t = t.copy()
                t = t.reindex( {indx:indx_} )
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
            else:
                t |= env_left[site_tag_id.format(site-1)]
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)



    def _update_env_left(self, psi, site: int, env_left):
        """Update left environment incrementally for current site."""
        
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block 
            
        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            t |= env_left[site_tag_id.format(site-1)]
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)

    
    def run_eff(self, n_iter=6, verbose=True):

        """Run the eefective fitting process"""
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        opt = self.opt

        #info_c = self.info_c
        #range_int = self.range_int 


        
        env_left = { site_tag_id.format(i):None   for i in range(psi.L)}
        env_right = { site_tag_id.format(i):None   for i in range(psi.L)}

        
        for iteration in range(n_iter):    
            
            for site in range(L):


                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1
                # Canonicalize psi at the current site

                
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                
                self._build_env_right(psi, env_right) if site == 0 else self._update_env_left(psi, site-1, env_left)
                
                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None
                
                if site == 0:
                    if tn_site:
                        tn =  tn_site | env_right[site_tag_id.format(site+1)]
                    else:
                        tn = env_right[site_tag_id.format(site+1)]
                    
                if site > 0 and site < L-1:
                    if tn_site:
                        tn =  tn_site  |  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
            
                if site == L-1:
                    if tn_site:
                        tn =  tn_site | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_left[site_tag_id.format(site-1)]

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(*psi[site_tag_id.format(site)].inds)
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                
               
                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))



    def run_gate(self, n_iter=6, verbose=True):

        """Run the eefective fitting process"""
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        opt = self.opt

        start, stop = self.range_int 
        
        env_left = { site_tag_id.format(i):None   for i in range(psi.L)}
        env_right = { site_tag_id.format(i):None   for i in range(psi.L)}


        for iteration in range(n_iter):    
            
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)

            for count_, site in enumerate(range(start, stop+1)):
 
                
                self._right_range(psi, env_right, start, stop) if count_ == 0 else self._left_range(psi, site-1, count_, env_left)

                
                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                if site == 0:
                    if tn_site:
                        tn =  tn_site | env_right[site_tag_id.format(site+1)]
                    else:
                        tn = env_right[site_tag_id.format(site+1)]


                
                if site > 0 and site < L-1:

                    # Boundary consistency: the left and right indices must match between tn and p
                    if count_ == 0:
                        indx = psi.bond(start-1, start)
                        indx_ = self.tn.bond(start-1, start)
                        tn_site = tn_site.reindex({indx_:indx})
                    if count_ == stop  - start:
                        indx = psi.bond(stop+1, stop)
                        indx_ = self.tn.bond(stop+1, stop)
                        tn_site = tn_site.reindex({indx_:indx})
                        
                    
                    if tn_site:
                        if env_right[site_tag_id.format(site+1)] is not None and env_left[site_tag_id.format(site-1)] is not None:
                            tn =  tn_site  |  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
                        elif env_left[site_tag_id.format(site-1)] is not None:
                            tn =  tn_site | env_left[site_tag_id.format(site-1)]
                        elif env_right[site_tag_id.format(site+1)] is not None:
                            tn =  tn_site | env_right[site_tag_id.format(site+1)]
                        else:
                            tn =  tn_site 
                         
                    else:
                        tn = env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
            
                if site == L-1:
                    if tn_site:
                        tn =  tn_site | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_left[site_tag_id.format(site-1)]

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(*psi[site_tag_id.format(site)].inds)
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                
               
                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)

                if site < stop:
                    psi.left_canonize_site(site, bra=None)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))




            


class Trck_boundary:

    def __init__(self, tn, opt="auto-hq", chi=4, cutoffs=1.e-10, to_backend=None, to_backend_=None, Lx=4, Ly=4):


        self.tn = tn
        self.opt = opt
        self.chi = chi
        self.to_backend = to_backend
        self.to_backend_ = to_backend_
        self.Lx = Lx
        self.Ly = Ly
        self.cutoffs = cutoffs

        self.mps_b = self._init_left(site_tag_id = "X{}", cut_tag_id = "Y{}")
        self.mps_b |= self._init_right(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mps_b |= self._init_right(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mps_b |= self._init_left(site_tag_id = "Y{}", cut_tag_id = "X{}")

        self.mpo_b = self._init_left_(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mpo_b |= self._init_left_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mpo_b |= self._init_right_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        self.mpo_b |= self._init_right_(site_tag_id = "X{}", cut_tag_id = "Y{}")



    
    def _init_left(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            
            if count == 0:
                mps = tn
            else:
                mps = tn | p_b_left[cut_tag_id.format(count-1) + "_l"]
        
            
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
            
            # print(site_tags, numbers)
        
            
            inds_ = []
            inds_size = {}
            for tag in site_tags:
                tn_select = mps.select(tag)
                inds_local = []
                for j_ in tn_select.outer_inds():
                    if j_ in outer_inds:
                        inds_local.append(j_)
                        #print( mps.ind_size(j_) )
                        inds_size |= {j_:mps.ind_size(j_)}
                inds_.append(inds_local)
        
        
            inds_k = {}
            count_ = 0
            for inds in inds_:
                for indx in inds:
                    inds_k |= {f"k{count_}":indx}  
                    count_ += 1
        
        
            # create the nodes, by default just the scalar 1.0
            tensors = [qtn.Tensor() for _ in range(L_mps)]
            
            for i_ in range(L_mps):
                if i_ < (L_mps-1):
                    # add the physical indices, each of size 2
                    tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                    tensors[i_].add_tag(f"I{i_}")
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
                if i_ == L_mps-1:
                    # add the physical indices, each of size 2
                    tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                    tensors[i_].add_tag(f"I{i_}")
        
                    
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.to_backend_)
            p.randomize( seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
            p.normalize()
            p_b_left[cut_tag_id.format(count) + "_l"] = p    
    
    
        return p_b_left



    def _init_right(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
    
        p_b_right = {}
        
        # iterate from Ly-1 down to 0 (inclusive)
        for count in range(0, self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any").copy()
        
            # first iteration (top row) -> initialize; otherwise attach previously built block
            if count == 0:
                mps = tn
            else:
                # when going downward, previously-built block is at count+1
                
                mps = tn | p_b_right[cut_tag_id.format(count - 1) + "_r"]
        
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            # Build regex to capture the integer inside site_tag_id, e.g. "X(\d+)"
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
        
            # Build index lists and sizes for the selected site tags
            inds_ = []
            inds_size = {}
            for tag in site_tags:
                tn_select = mps.select(tag)
                inds_local = []
                for j_ in tn_select.outer_inds():
                    if j_ in outer_inds:
                        inds_local.append(j_)
                        inds_size[j_] = mps.ind_size(j_)
                inds_.append(inds_local)
        
            # flatten inds -> new names k0, k1, ... (keeps order consistent with sorted site_tags)
            inds_k = {}
            count_ = 0
            for inds in inds_:
                for indx in inds:
                    inds_k[f"k{count_}"] = indx
                    count_ += 1
        
            # create tensors (one per outer index)
            tensors = [qtn.Tensor() for _ in range(L_mps)]
        
            for i_ in range(L_mps):
                # create physical index k{i_} with recorded size
                tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                tensors[i_].add_tag(f"I{i_}")
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (L_mps - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
        
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L=L_mps, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.to_backend_)
            p.randomize(seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
        
            # store the block at the current cut tag
            p.normalize()
            p_b_right[cut_tag_id.format(count) + "_r"] = p
    
    
        return p_b_right
    
    def _init_left_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        
    
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            tn.compress_all(inplace=True, **{"max_bond":chi_1, "canonize_distance": 2, "cutoff":1.e-12})
        
            
            if count == 0:
                mps = tn
            else:
                mps_ = mps_b[cut_tag_id.format(count-1) + "_l"].copy()
                mps_.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
                mps = tn | mps_
                mps.drop_tags(cut_tag_id.format(count-1))
        
        
        
            for i in range(self.Lx): 
                mps.contract_tags_(
                                    site_tag_id.format(i), optimize=self.opt)
            
            mps.fuse_multibonds_()
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
            mps.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
            
            mps.apply_to_arrays(self.to_backend_)
        
            mps.expand_bond_dimension(self.chi, 
                                      rand_strength=0.01
                                     )
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
            mps.normalize()
            mps.apply_to_arrays(self.to_backend)
        
            # mps.draw([f"X{i}" for i in range(Lx)], show_inds="bond-size", show_tags=False)
        
            mps_b[cut_tag_id.format(count) + "_l"] = mps
        return mps_b

    
    
    def _init_right_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any")
            tn = tn.copy()
            tn.compress_all(inplace=True, **{"max_bond":chi_1, "canonize_distance": 4, "cutoff":1.e-12})
        
            
            if count == 0:
                mps = tn
            else:
                mps_ = mps_b[cut_tag_id.format(count-1) + "_r"].copy()
                mps_.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
                mps = tn | mps_
                mps.drop_tags(cut_tag_id.format(self.Ly-1-count+1))
        
        
        
            for i in range(self.Lx): 
                mps.contract_tags_(
                                    site_tag_id.format(i), optimize=self.opt)
            
            mps.fuse_multibonds_()
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
            mps.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
            
            mps.apply_to_arrays(self.to_backend_)
        
            mps.expand_bond_dimension(self.chi, 
                                      rand_strength=0.01
                                     )
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)

            mps.normalize()
            mps.apply_to_arrays(self.to_backend)
        
            # mps.draw([f"X{i}" for i in range(Lx)], show_inds="bond-size", show_tags=False)
        
            mps_b[cut_tag_id.format(count) + "_r"] = mps
        return mps_b





def nlopt_optimize(pepo_fix, params, sites, edges, cost_opts, cost_fn, its_max=2500):
    # ---------------------- Config ----------------------
    dtype_  = torch.float64
    device  = "cpu"

    energy_log = deque(maxlen=5000)
    grad_log   = deque(maxlen=5000)
    step_ctr   = 0

    # ------------- Flatten / Unflatten utils ------------
    def flatten_params(trainable_params):
        return parameters_to_vector(list(trainable_params.values())).detach().cpu().numpy()

    def unflatten_params(trainable_params, x: np.ndarray):
        if not x.flags.writeable:
            x = np.copy(x)
        vec = torch.from_numpy(x).to(device=device, dtype=dtype_)
        vector_to_parameters(vec, list(trainable_params.values()))

    # tqdm progress bar (dynamic updates)
    pbar = tqdm(total=its_max, desc="nlopt", ncols=100, dynamic_ncols=True)

    # -------------------- Objective ---------------------
    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        nonlocal step_ctr
        t0 = time.perf_counter()
        step_ctr += 1

        unflatten_params(params, x)
        for p in params.values():
            if p.grad is not None:
                p.grad.zero_()

        with torch.autograd.set_detect_anomaly(False):
            loss = cost_fn(pepo_fix, params, sites, edges, **cost_opts)

        loss.backward()

        if grad.size > 0:
            flat_g = parameters_to_vector([
                (p.grad if p.grad is not None else torch.zeros_like(p))
                for p in params.values()
            ])
            np.copyto(grad, flat_g.detach().cpu().numpy(), casting='no')
            grad_norm = float(flat_g.norm().item())
        else:
            grad_norm = float('nan')

        energy = float(loss.item())
        energy_log.append(energy)
        grad_log.append(grad_norm)

        # --- Update tqdm ---
        pbar.set_postfix({
            "loss": f"{energy:.3e}",
            "||grad||": f"{grad_norm:.2e}"
        })
        pbar.update(1)

        # --- Stability guard ---
        if not np.isfinite(energy) or not np.isfinite(grad).all():
            return 1e50

        return energy

    # -------------------- Optimize ----------------------
    x0 = flatten_params(params)
    opt_nl = nlopt.opt(nlopt.LD_VAR2, len(x0))
    # opt_nl = nlopt.opt(nlopt.LD_LBFGS, len(x0))


    
    opt_nl.set_min_objective(objective)
    opt_nl.set_maxeval(its_max)
    opt_nl.set_ftol_rel(1e-9)
    opt_nl.set_xtol_rel(1e-9)
    opt_nl.set_ftol_abs(1e-9)

    x_opt = opt_nl.optimize(x0)
    final_loss = opt_nl.last_optimum_value()
    pbar.close()

    # --- Load optimized x_opt back into params ---
    unflatten_params(params, x_opt)

    # Convert logs to arrays
    energy_hist = np.fromiter(energy_log, dtype=float)
    grad_hist   = np.fromiter(grad_log, dtype=float)

    # ✅ Return updated parameter dictionary instead of x_opt
    return params, energy_hist, grad_hist




def nlopt_optimize_(params, cost_opts, cost_fn, its_max=2500, optimizer="LBFGS", 
                    device  = "cpu"):
    # ---------------------- Config ----------------------
    
    dtype_  = torch.float64
    device  = device

    energy_log = deque(maxlen=5000)
    grad_log   = deque(maxlen=5000)
    step_ctr   = 0

    # ------------- Flatten / Unflatten utils ------------
    def flatten_params(trainable_params):
        return parameters_to_vector(list(trainable_params.values())).detach().cpu().numpy()

    def unflatten_params(trainable_params, x: np.ndarray):
        if not x.flags.writeable:
            x = np.copy(x)
        vec = torch.from_numpy(x).to(device=device, dtype=dtype_)
        vector_to_parameters(vec, list(trainable_params.values()))

    # tqdm progress bar (dynamic updates)
    pbar = tqdm(total=its_max, desc="nlopt", ncols=100, dynamic_ncols=True)

    # -------------------- Objective ---------------------
    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        nonlocal step_ctr
        t0 = time.perf_counter()
        step_ctr += 1

        unflatten_params(params, x)
        for p in params.values():
            if p.grad is not None:
                p.grad.zero_()

        with torch.autograd.set_detect_anomaly(False):
            loss = cost_fn(params, **cost_opts)

        loss.backward()

        if grad.size > 0:
            flat_g = parameters_to_vector([
                (p.grad if p.grad is not None else torch.zeros_like(p))
                for p in params.values()
            ])
            np.copyto(grad, flat_g.detach().cpu().numpy(), casting='no')
            grad_norm = float(flat_g.norm().item())
        else:
            grad_norm = float('nan')

        energy = float(loss.item())
        energy_log.append(energy)
        grad_log.append(grad_norm)

        # --- Update tqdm ---
        pbar.set_postfix({
            "loss": f"{energy:.3e}",
            "||grad||": f"{grad_norm:.2e}"
        })
        pbar.update(1)

        # --- Stability guard ---
        if not np.isfinite(energy) or not np.isfinite(grad).all():
            return 1e50

        return energy

    # -------------------- Optimize ----------------------
    x0 = flatten_params(params)
    
    if optimizer== "LBFGS":
        opt_nl = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    else:
        opt_nl = nlopt.opt(nlopt.LD_VAR2, len(x0))

    
    opt_nl.set_min_objective(objective)
    opt_nl.set_maxeval(its_max)
    opt_nl.set_ftol_rel(1e-9)
    opt_nl.set_xtol_rel(1e-9)
    opt_nl.set_ftol_abs(1e-9)

    x_opt = opt_nl.optimize(x0)
    final_loss = opt_nl.last_optimum_value()
    pbar.close()

    # --- Load optimized x_opt back into params ---
    unflatten_params(params, x_opt)

    # Convert logs to arrays
    energy_hist = np.fromiter(energy_log, dtype=float)
    grad_hist   = np.fromiter(grad_log, dtype=float)

    # ✅ Return updated parameter dictionary instead of x_opt
    return params, energy_hist, grad_hist





    
def adam_optimize(pepo_fix, params, sites, edges, cost_opts, its_max=2500, lr=0.1):

    
    import torch.optim as optim
    
    params_list = list(params.values())
    optimizer = optim.Adam(params_list, lr=lr)
    

    loss_history = []
    pbar = tqdm(total=its_max, desc="adam", ncols=100, dynamic_ncols=True)
    
    for step in range(its_max):
        optimizer.zero_grad()
        loss = cost_function(pepo_fix, params, sites, edges, **cost_opts)
        loss.backward(retain_graph=False)  # keep graph if needed
        optimizer.step()
        
        loss_history.append(loss.item())   # store scalar for plotting
        
        #print(f"Step {step}: Loss = {loss.item()}")
        pbar.set_postfix({
                "loss": f"{loss.item():.3e}",
            })
        pbar.update(1)
    pbar.close()

    return params, loss_history
