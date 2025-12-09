import quimb.tensor as qtn
import quimb as qu
import cotengra as ctg
import autoray as ar
import register_ as reg
import algo_cooling as algo
import quf
import time
import numpy as np
from tqdm import tqdm
import nlopt
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


import matplotlib.pyplot as plt
reg.reg_complex_svd()

to_backend = algo.backend_torch(device="cpu", dtype = torch.float64, requires_grad=False)
to_backend_c = algo.backend_torch(device="cpu", dtype = torch.complex128, requires_grad=False)
to_backend_ = algo.backend_torch(device="cpu", dtype = torch.float64, requires_grad=True)

opt = algo.opt_(progbar=True, )
#ITF params
J = 1
h = 3.05
delta = 0.05
Lx, Ly = 3, 4       # lattice dimensions: rows x columns
L = Lx * Ly          # total number of sites
edges = qtn.edges_2d_square(Lx=Lx, Ly=Ly, cyclic=False)
sites = sorted({ (site,) for edge in edges for site in edge})
N = len(sites)
pepo = quf.pepo_identity(Lx, Ly)
pepo.apply_to_arrays(to_backend_c)

depth_total = 4
its_max = 5000

print("depth_total", depth_total, "its_max", its_max)

pepo, params = algo.varU_params_(pepo, edges, sites, depth_total=depth_total, h=h, delta=delta, J=J, to_backend_=to_backend_)

pepo_fix = qu.load_from_disk("store/pepo")
pepo_fix = qu.load_from_disk("store/U_tn")


cost_opts = { "Lx":Lx, "Ly":Ly, "depth_total":depth_total, "opt":opt, "to_backend":to_backend_c  }
algo.cost_function(pepo_fix, params, sites, edges, **cost_opts)

params, energy_hist, grad_hist = algo.nlopt_optimize(pepo_fix, params, sites, edges, cost_opts, 
                                                     cost_fn=algo.cost_function, its_max=its_max,
                                                    )



qu.save_to_disk(params, f"store/params_d{depth_total}_")
qu.save_to_disk(energy_hist, f"store/loss_d{depth_total}_")
qu.save_to_disk(grad_hist, f"store/grad_d{depth_total}_")
