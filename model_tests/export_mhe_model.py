import numpy as np
from scipy.linalg import block_diag
from acados_template import AcadosOcpSolver, AcadosOcp
from casadi import *
from acados_template import AcadosModel

def mod_approx(x):

    epsilon = 1e-2

    return sqrt(x**2 + epsilon)

def export_mhe_ode_model_with_param(sigma):
    '''
    Export ode model augmented with an additional state corresponding to the
    parameter l, which is identified online
    '''

    model_name = 'mhe_friction_model'

    # constants

    nx = 1

    # set up states
    Ff      = SX.sym('Ff')

    x = vertcat(Ff)

    # state noise

    w_Ff      = SX.sym('w_Ff')

    w = vertcat(w_Ff)

    # xdot
    Ff_dot      = SX.sym('Ff_dot')

    xdot = vertcat(Ff_dot)

    # algebraic variables
    z = []

    # parameters <= controls (Add V and T*e^(-mu*))
    F = SX.sym('F')

    V = SX.sym('V')
    Fc = SX.sym('Fc')

    p = vertcat(V, Fc)

    # dynamics

    f_expl = vertcat((V - (Ff/Fc)*mod_approx(V)*sigma) + w)
    # add additive state noise
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = w
    model.z = z
    model.p = p
    model.name = model_name

    return model

def export_mhe_solver_with_param(model, N, h, Q, Q0, R, use_cython=False):

    # create render arguments
    ocp_mhe = AcadosOcp()

    ocp_mhe.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    nparam = model.p.size()[0]

    ny = R.shape[0] + Q.shape[0]                    # h(x), w
    ny_e = 0
    ny_0 = R.shape[0] + Q.shape[0] + Q0.shape[0]    # h(x), w and arrival cost

    # set number of shooting nodes
    ocp_mhe.dims.N = N

    x = ocp_mhe.model.x
    u = ocp_mhe.model.u

    # set cost type
    ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
    ocp_mhe.cost.cost_type_e = 'LINEAR_LS'
    ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'

    ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
    ocp_mhe.model.cost_y_expr_0 = vertcat(x, u, x)
    ocp_mhe.cost.yref_0 = np.zeros((ny_0,))


    # cost intermediate stages
    ocp_mhe.cost.W = block_diag(R, Q)

    ocp_mhe.model.cost_y_expr = vertcat(x, u)

    ocp_mhe.parameter_values = np.zeros((nparam, ))

    # set y_ref for all stages
    ocp_mhe.cost.yref  = np.zeros((ny,))
    ocp_mhe.cost.yref_e = np.zeros((ny_e, ))
    ocp_mhe.cost.yref_0  = np.zeros((ny_0,))

    # set QP solver
    # ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp_mhe.solver_options.integrator_type = 'ERK'

    # set prediction horizon
    ocp_mhe.solver_options.tf = N*h

    ocp_mhe.solver_options.nlp_solver_type = 'SQP'
    # ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp_mhe.solver_options.nlp_solver_max_iter = 200
    ocp_mhe.code_export_directory = 'mhe_generated_code'

    if use_cython:
        AcadosOcpSolver.generate(ocp_mhe, json_file='acados_ocp_mhe.json')
        AcadosOcpSolver.build(ocp_mhe.code_export_directory, with_cython=True)
        acados_solver_mhe = AcadosOcpSolver.create_cython_solver('acados_ocp_mhe.json')
    else:
        acados_solver_mhe = AcadosOcpSolver(ocp_mhe, json_file = 'acados_ocp_mhe.json')

    # set arrival cost weighting matrix
    acados_solver_mhe.cost_set(0, "W", block_diag(R, Q, Q0))

    return acados_solver_mhe

if __name__ == "__main__": 

    model = export_mhe_ode_model_with_param(1.0)

    N = 20
    h = 0.01

    w_std = 0.05
    v_std = 0.05

    Q = np.array([1/w_std])
    R = np.array([1/v_std])
    Q0 = 0.01*Q

    export_mhe_solver_with_param(model, N, h, Q, Q0, R, use_cython=False)
