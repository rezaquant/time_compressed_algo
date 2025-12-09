import quimb.tensor as qtn
import quimb as qu
import cotengra as ctg
import autoray as ar
from . import register_ as reg
from . import algo_cooling as algo
from . import quf
import time
import numpy as np
from tqdm import tqdm
import nlopt
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from quimb.tensor.belief_propagation.l2bp import L2BP
from .gate_arb import TensorNetworkGenVector


import matplotlib as mpl
import matplotlib.pyplot as plt

reg.reg_complex_svd()

to_backend = algo.backend_torch(device = "cpu", dtype = torch.float64, requires_grad=False)
to_backend_c = algo.backend_torch(device = "cpu", dtype = torch.complex128, requires_grad=False)
to_backend_ = algo.backend_torch(device = "cpu", dtype = torch.float64, requires_grad=True)

opt = algo.opt_(progbar=False, max_time="rate:1e9", max_repeats=128, optlib="cmaes")
info_su = qu.load_from_disk(f"store_state/info_su")
info_bp = qu.load_from_disk(f"store_state/info_bp")

#ITF params
J, h, chi, dt, depth = info_su["J"], info_su["h"], info_su["chi"], info_su["dt"], info_su["depth"],
Lx, Ly, L = info_su["Lx"], info_su["Ly"], info_su["L"]
t = depth * dt
print("t", t, "chi", chi, "L", L)

edges_1d, sites, site_tags = info_su["edges_1d"], info_su["sites"], info_su["site_tags"]

pepo, gauges = info_su["pepo"], info_su["gauges"]
# pepo.apply_to_arrays(to_backend_c)
# to_backend_c(pepo.exponent)
# gauges = { u:to_backend(v)  for u, v in gauges.items()}



depth_total = 4
params = {}

#p1 = 1.0 / (2.0 - 2.0**(1/3))
#p2 = - (2.0**(1/3)) / (2.0 - 2.0**(1/3))


#for depth in range(depth_total):
#        if depth ==1:
#            phi = -h * 0.5 * p2
#            theta = -J * 2 * 0.5 *p2
#        else:
#            phi = -h * 0.5 * p1
#            theta = -J * 2 * 0.5 *p1

#        params[f"rx_depth{depth}"] = to_backend_( torch.tensor( phi ).clone().detach() )
#        params[f"rzz_depth{depth}"] = to_backend_( torch.tensor( theta ).clone().detach() )

params = {}
depth_total = 4
for depth in range(depth_total):
         phi = -0.36
         theta = -0.24
         params[f"rx_depth{depth}"] = to_backend_( torch.tensor( phi ).clone().detach() )
         params[f"rzz_depth{depth}"] = to_backend_( torch.tensor( theta ).clone().detach() )

chi = 12
chi_mps = chi

print("chi", chi, "chi_mps", chi_mps, "depth_total", depth_total)

cost_opts = { "Lx":Lx, "Ly":Ly, "depth_total":depth_total, "opt":opt, "edges":edges_1d, "stable":True  }
cost_opts |= { "cutoff":0.0, "map_tags_2d":info_su["map_tags_2d"], "opt":opt, "chi":chi, "sites": sites }
cost_opts |= { "renorm":False, "equalize_norms":True, "chi_mps":chi_mps }
cost_opts |= { "pepo":pepo, "gauges":gauges  }

cost = algo.cost_function_su(params,  **cost_opts)
print( float(cost) )
import torch.optim as optim

params_list = list(params.values())
optimizer = optim.Adam(params_list, lr=0.01)

its_max = 400
loss_history = []
pbar = tqdm(total=its_max, desc="adam", ncols=100, dynamic_ncols=True)

for step in range(its_max):
    optimizer.zero_grad()

    with torch.no_grad():
        pepo_ = pepo.copy()
        gauges_ = {u: v * 1.0 for u, v in gauges.items()}
        cost_opts |= { "pepo":pepo_, "gauges":gauges_  }
    

    loss = algo.cost_function_su(params,  **cost_opts)
    loss.backward()  # keep graph if needed
    optimizer.step()
    
    loss_history.append(loss.item())   # store scalar for plotting
    
    #print(f"Step {step}: Loss = {loss.item()}")
    pbar.set_postfix({
            "loss": f"{loss.item():.3e}",
        })
    pbar.update(1)
pbar.close()


qu.save_to_disk(params, f"store/params_su_chi{chi}_L{L}_d{depth_total}")
qu.save_to_disk(loss_history, f"store/loss_su_chi{chi}_L{L}_d{depth_total}")
