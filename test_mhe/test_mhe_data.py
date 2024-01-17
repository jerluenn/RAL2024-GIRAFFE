import numpy as np
from scipy.linalg import block_diag
from casadi import *
from export_mhe_model import *
import pandas as pd 
from matplotlib import pyplot as plt

filename_tension = 'tension_kalmanfiltered' + '.csv'
filename_tension_rate = 'tension_rate_kalmanfiltered' + '.csv'

pd_tension = pd.read_csv(filename_tension, delimiter=',')
pd_tension_rate = pd.read_csv(filename_tension_rate, delimiter=',')



proximal_side_tension_rate = pd_tension_rate.to_numpy()[:, 4]
distal_side_tension_rate = pd_tension_rate.to_numpy()[:, 6]

proximal_side_tension = pd_tension.to_numpy()[:, 4]
distal_side_tension = pd_tension.to_numpy()[:, 6]

t_array = pd_tension.to_numpy()[:, 1]

plt.plot(proximal_side_tension, distal_side_tension)
plt.show()

model = export_mhe_ode_model_with_param(1.0)

N = 20
h = 0.01

w_std = 0.05
v_std = 0.05

Q = np.array([1/w_std])
R = np.array([1/v_std])
Q0 = 0.01*Q

solver = export_mhe_solver_with_param(model, N, h, Q, Q0, R, use_cython=False)

x0_bar = np.array([0.0])
u0 = np.array([0.0])