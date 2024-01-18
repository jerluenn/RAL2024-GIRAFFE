import numpy as np
from scipy.linalg import block_diag
from casadi import *
import pandas as pd 
from matplotlib import pyplot as plt
from controller_solver import *

Tf = 1.0  # prediction horizon
N = 50  # number of discretization steps
T = 30.00  # maximum simulation time[s]
Q = block_diag(1e5, 50) 
R = np.array([10])
h = Tf/N

# load model
control_model = export_control_model()
sim_model = export_sim_model()
solver = export_ocp_solver(control_model, N, h, Q, R)
plant = export_sim_solver(sim_model, N, h)

# dimensions
nx = control_model.x.size()[0]
nu = control_model.u.size()[0]
ny = nx + nu
# Nsim = int(T * N / Tf)
Nsim = 50

simX = np.ndarray((Nsim, nx))
simU = np.ndarray((Nsim, nu))

yref = np.array([5, 0, 0])

simT = np.ndarray((Nsim))

x0 = np.array([0, 0, 0.00])

u = 5
t = 0

for i in range(N): 

    solver.set(i, 'yref', yref)

for i in range(Nsim): 

    solver.set(0, 'lbx', x0)
    solver.set(0, 'ubx', x0)

    solver.solve()

    u = solver.get(0, 'u')

    # # print(u)

    # print(solver.get_residuals())

    t += Tf/N

    plant.set('x', x0)
    plant.set('u', u)

    plant.solve()

    x0 = plant.get('x')

    simX[i, :] = x0
    simU[i, :] = u
    simT[i] = t

    print("x0, t", x0, t)

    V = x0[1]/10
    Ff = x0[2]
    sigma = 10
    T_dot = x0[1]
    Fc = x0[0]*(1 - np.exp(-0.45*5))

    print("Fc", Fc)
    print("Ratio", Ff/(Fc + 1e-2))

    print("Ff_dot", (((V - (Ff/(Fc + 1e-2))*mod_approx_sim(V))*sigma)))
    print("Lyap_dot: ", (((V - (Ff/(Fc + 1e-2))*mod_approx_sim(V))*sigma)) - T_dot)


plt.plot(simT, simX)
plt.show()