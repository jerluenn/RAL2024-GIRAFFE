import numpy as np
from scipy.linalg import block_diag
from casadi import *
from export_mhe_model import *

tension_csv = 'loadcell_kalmanfiltered' + '.csv'
tension_rate_csv = 'tension_rate_filtered' + '.csv'

tension_np = np.loadtxt(tension_csv, delimiter=',')
tension_rate_np = np.loadtxt(tension_rate_csv, delimiter=',')