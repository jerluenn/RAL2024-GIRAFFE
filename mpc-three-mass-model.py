from casadi import *
import numpy as np

import time
import os

from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim

from scipy.integrate import BDF

class Controller: 


    def __init__(self, b, m, mu, gamma, k, sigma): 

        self.b = b 
        self.m = m 
        self.mu = mu 
        self.gamma = gamma 
        self.k = k 
        self.sigma = sigma
    
    def createModel(self): 

        model_name = 'model'

        x1 = SX.sym('x1')
        x2 = SX.sym('x2')
        x3 = SX.sym('x3')

        x1dot = SX.sym('x1dot')
        x2dot = SX.sym('x2dot')
        x3dot = SX.sym('x3dot')

        Ff = SX.sym('Ff')

        F_act = SX.sym('F_act')

        gamma = SX.sym('gamma')

        f_expl = vertcat(
            x1dot,
            x2dot,
            x3dot,
            (F_act - self.b[0]*x1dot - self.k*(x1 - x3))/self.m[0],
            (self.b[1]*x2dot - self.k*(x2 - x3))/self.m[1],
            (- Ff - self.b[2]*x3dot - self.k*(-x1 - x2 + 2*x3))/self.m[2],
            self.sigma*(x3dot - (Ff*self.mod_approx(x3dot))/(self.mu*gamma*self.k*(x2-x1)))
        ) 

        x = vertcat(x1, x2, x3, x1dot, x2dot, x3dot, Ff)

        u = vertcat(F_act)

        model = AcadosModel()

        gamma = SX.sym('gamma')

        params = vertcat(gamma)

        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = model_name        
        model.p = params

        return model

    def mod_approx(self, x):

        epsilon = 1e-2

        return sqrt(x**2 + epsilon)

    def createSolver(self, x0, max_tension, N_horizon, RTI, Tf): 

        ocp = AcadosOcp()

        model = self.createModel()

        # nx = model.x.size()[0]
        # nu = model.u.size()[0]
        # ny = nx + nu
        # ny_e = nx

        ocp.model = model

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.model.cost_y_expr = vertcat(model.p[0]*model.x[0] + model.p[1]*model.x[1], model.u)
        ocp.model.cost_y_expr_e = vertcat(model.p[0]*model.x[0] + model.p[1]*model.x[1])
        # ocp.model.cost_y_expr_0 = vertcat(model.p[0]*model.x[0] + model.p[1]*model.x[1], model.u)
        # ocp.cost.yref_0 = np.zeros((ny, ))
        ocp.cost.yref  = np.zeros((2, ))
        ocp.cost.yref_e = np.zeros((1, ))

        ocp.constraints.lbu = np.array([-max_tension])
        ocp.constraints.ubu = np.array([max_tension])

        ocp.cost.W = np.diag([10, 0.01])
        ocp.cost.W_e = np.diag([100])

        ocp.constraints.x0 = x0
        ocp.constraints.idxbu = np.array([0])

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_newton_iter = 10
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 10
        ocp.solver_options.levenberg_marquardt = 1.0

        if RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.dims.N = N_horizon
        # ocp.solver_options.qp_solver_cond_N = N_horizon
        ocp.parameter_values = np.array([self.alpha_ten, self.alpha_h])

        # set prediction horizon
        ocp.solver_options.tf = Tf

        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator

