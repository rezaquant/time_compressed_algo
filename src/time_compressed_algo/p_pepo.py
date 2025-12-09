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


import matplotlib.pyplot as plt

reg.reg_complex_svd()

to_backend = algo.backend_torch(device = "cpu", dtype = torch.float64, requires_grad=False)
to_backend_c = algo.backend_torch(device = "cpu", dtype = torch.complex128, requires_grad=False)
to_backend_ = algo.backend_torch(device = "cpu", dtype = torch.float64, requires_grad=True)

opt = algo.opt_(progbar=True, max_time="rate:1e9", max_repeats=2**7, optlib="cmaes")

#ITF params
J = 1
h = 3.05
delta = 1.0
Lx, Ly = 3, 4       # lattice dimensions: rows x columns
L = Lx * Ly          # total number of sites
edges = qtn.edges_2d_square(Lx=Lx, Ly=Ly, cyclic=False)
sites = sorted({ (site,) for edge in edges for site in edge})
N = len(sites)

pepo = quf.pepo_identity(Lx, Ly)
pepo.apply_to_arrays(to_backend_c)

#depth_total = 4
#params = {}

#p1 = 1.0 / (2.0 - 2.0**(1/3))
#p2 = - (2.0**(1/3)) / (2.0 - 2.0**(1/3))


#for depth in range(depth_total):
        #if depth ==1:
            #phi = -h * 0.05 * p2
            #theta = -J * 2 * 0.05 *p2
        #else:
            #phi = -h * 0.05 * p1
            #theta = -J * 2 * 0.05 *p1

        #params[f"rx_depth{depth}"] = to_backend_( torch.tensor( phi ).clone().detach() )
        #params[f"rzz_depth{depth}"] = to_backend_( torch.tensor( theta ).clone().detach() )

params = {}
depth_total = 4
for depth in range(depth_total):
         phi = -0.32
         theta = -0.24
         params[f"rx_depth{depth}"] = to_backend_( torch.tensor( phi ).clone().detach() )
         params[f"rzz_depth{depth}"] = to_backend_( torch.tensor( theta ).clone().detach() )



pepo = algo.skeleten_pepo(params, edges, sites, depth_total=depth_total,contract=True,
                                  to_backend=to_backend_c, Lx=Lx, Ly=Ly)

info_su = qu.load_from_disk(f"store_state/info_su")
info_bp = qu.load_from_disk(f"store_state/info_bp")

pepo_fix = info_su["tn"]
# pepo_fix = info_bp["pepo"]

# pepo_fix = qu.load_from_disk("store/pepo")
# pepo_fix = qu.load_from_disk("store/U_tn")

#pepo_fix.show()

cost_opts = { "Lx":Lx, "Ly":Ly, "depth_total":depth_total, "opt":opt, "to_backend":to_backend_c  }
cost = algo.cost_function(pepo_fix, params, sites, edges, **cost_opts)
print(  float(cost)   )

params, loss_history = algo.adam_optimize(pepo_fix, params, sites, edges, cost_opts, its_max=400, lr=0.01)


# params, energy_hist, grad_hist = algo.nlopt_optimize(pepo_fix, params, sites, edges, cost_opts, 
#                                                      cost_fn=algo.cost_function, its_max=20
#                                                     )


qu.save_to_disk(params, f"store/params_L{L}_d{depth_total}")
qu.save_to_disk(loss_history, f"store/loss_L{L}_d{depth_total}")
