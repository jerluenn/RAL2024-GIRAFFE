from acados_template.acados_sim import AcadosSim
import numpy as np
from scipy.linalg import block_diag
from acados_template import *
from casadi import *

def mod_approx_sim(x):

    epsilon = 1e-10

    return sqrt(x**2 + epsilon)

def mod_approx_control(x): 

    epsilon = 1e-4

    return sqrt(x**2 + epsilon)

def export_sim_model(): 

    model_name = 'sim_model'

    # constants

    sigma = 70

    a = np.array([-1, -2])

    nx = 3

    k = 10

    gamma = 0.2

    # set up states
    Ff      = SX.sym('Ff')
    T       = SX.sym('T')
    T_dot   = SX.sym('T_dot')

    x = vertcat(T, T_dot, Ff)

    # xdot
    Ff_dot      = SX.sym('Ff_dot')
    T_dot_dot   = SX.sym('T_dot')

    xdot = vertcat(T_dot, T_dot_dot, Ff_dot)

    mu      = SX.sym('mu')

    p = vertcat(mu)

    z = []

    # p = vertcat(V, Fc)

    F = SX.sym('F')
    u = vertcat(F)

    V = T_dot/k
    Fc = T*(1 - exp(-mu*gamma))

    # dynamics

    f_expl = vertcat(T_dot, 
        a[0]*T + a[1]*T_dot + F, 
        ((V - (Ff/(Fc + 1e-2))*mod_approx_sim(V))*sigma)
        )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.con_h_expr = (((V - (Ff/(Fc + 1e-2))*mod_approx_sim(V))*sigma)) - T_dot

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    return model

def export_sim_solver(model): 

    sim = AcadosSim()

    # set model
    sim.model = model

    Tf = N*h

    sim.parameter_values = np.array([5])

    sim.solver_options.T = h
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 10

    acados_integrator = AcadosSimSolver(sim)

    return acados_integrator

def export_control_model():

    model_name = 'control_model'

    # constants

    sigma = 70

    a = np.array([-1, -2])

    nx = 3

    k = 10

    gamma = 0.2

    # set up states
    Ff      = SX.sym('Ff')
    T       = SX.sym('T')
    T_dot   = SX.sym('T_dot')

    x = vertcat(T, T_dot, Ff)

    # xdot
    Ff_dot      = SX.sym('Ff_dot')
    T_dot_dot   = SX.sym('T_dot')

    xdot = vertcat(T_dot, T_dot_dot, Ff_dot)

    mu      = SX.sym('mu')

    p = vertcat(mu)

    z = []

    # p = vertcat(V, Fc)

    F = SX.sym('F')
    u = vertcat(F)

    V = T_dot/k
    Fc = T*(1 - exp(-mu*gamma))

    # dynamics

    f_expl = vertcat(T_dot, 
        a[0]*T + a[1]*T_dot + F, 
        ((V - (Ff/(Fc + 1e-2))*mod_approx_control(V))*sigma)
        )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.con_h_expr = (((V - (Ff/(Fc + 1e-2))*mod_approx_control(V))*sigma)) - T_dot

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    return model

def export_ocp_solver(model, N, h, Q, R, Fmax=80, use_cython=False):

    ocp = AcadosOcp()

    # set model
    ocp.model = model

    Tf = N*h

    ocp.parameter_values = np.array([5])

    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 10
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu - 1 # we do not control friction.
    ny_e = nx - 1 

    ocp.dims.N = N

    # set cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.cost.W = block_diag(Q, R)

    ocp.cost.W_e = Q

    ocp.solver_options.levenberg_marquardt = 10.0

    x = ocp.model.x
    u = ocp.model.u

    T_out = x[0] - x[2]

    ocp.model.cost_y_expr = vertcat(T_out, x[1], u)

    ocp.model.cost_y_expr_e = vertcat(T_out, x[1])

    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    # setting bounds
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.ubx = np.array([100, 50, 100])
    ocp.constraints.lbx = np.array([0, -50, 0])
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    ns = 1

    ocp.cost.zl = 100 * np.ones((ns,))
    ocp.cost.zu = 0 * np.ones((ns,))
    ocp.cost.Zl = 1 * np.ones((ns,))
    ocp.cost.Zu = 0 * np.ones((ns,))

    ocp.constraints.lh = np.array([0])
    ocp.constraints.uh = np.array([100])

    ocp.constraints.lsh = np.zeros(ns)
    ocp.constraints.ush = np.zeros(ns)
    ocp.constraints.idxsh = np.array(range(ns))

    # ocp.constraints.lh = np.array

    # set QP solver
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.integrator_type = 'IRK'

    # set prediction horizon
    ocp.solver_options.tf = Tf
    # ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 200

    if use_cython:
        AcadosOcpSolver.generate(ocp, json_file='acados_ocp.json')
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_solver = AcadosOcpSolver.create_cython_solver('acados_ocp.json')
    else:
        acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

    return acados_solver

if __name__ == "__main__": 

    model = export_control_model(1.0)

    N = 20
    h = 0.01

    w_std = 0.05
    v_std = 0.05

    Q = np.array([1/w_std])
    R = np.array([1/v_std])
    Q0 = 0.01*Q

    # export_mhe_solver_with_param(model, N, h, Q, Q0, R, use_cython=False)
