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


class Hysteresis_MPC_Controller: 

    def __init__(self, rho, mu, sigma, kappa): 

        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.sigma = sigma
        self.kappa = kappa

    def createModel(self): 

        model_name = 'asymmetric-hysteresis_model'

        zeta = SX.sym('zeta')
        x = SX.sym('x')
        x_dot = SX.sym('x_dot')

        x = vertcat(zeta, x, x_dot)
        
        F = SX.sym('F')
        u = vertcat(F)

        tension_dot = (1/self.gamma_tension)*(tension_des - tension)

        f_expl = vertcat(tension_dot, self.rho*(tension_dot - self.sigma*self.mod_approx(tension_dot)*h*self.mod_approx(h)**(self.n-1) - (self.sigma - 1)*tension_dot*self.mod_approx(h)**self.n))

        model = AcadosModel()

        alpha_h = SX.sym('alpha_h')
        alpha_tension = SX.sym('alpha_tension')

        params = vertcat(alpha_tension, alpha_h)

        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = model_name        
        model.p = params

        return model

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


    def mod_approx(self, x):

        epsilon = 1e-6

        return sqrt(x**2 + epsilon)

    def sgn_approx(self, x): 

        epsilon = 1e-6

        return tanh(x/epsilon)

    def step_approx(self, x): 

        epsilon = 1e-6

        return 0.5 + 0.5*tanh(x/epsilon)



def sim_example(): 

    # alpha_ten, alpha_h, rho, sigma, gamma_tension, n

    obj = Hysteresis_MPC_Controller(0.5, 0.5, 0.5, 1.0, 3.0, 2)
    solver, integrator = obj.createSolver(np.zeros(2), 30, 40, 1, 2)

    x0 = np.zeros(2)
    num_sim_time = 2000

    states = np.zeros((num_sim_time+1, 3))
    simU = np.zeros((num_sim_time, 1))
    t_array = np.zeros(num_sim_time)
    t = 0 
    freq = 0.05


    for i in range(num_sim_time): 

        t += 0.05
        t_array[i] = t

        solver.set(0, 'lbx', x0)
        solver.set(0, 'ubx', x0)
        solver.solve() # Testing the solver.


        if i > 500 : 

        # u = 5*np.sin(i*freq) + 5
            u = 0

        else: 

            u = 5 

        simU[i, :] = u

        print(simU[i, :])

        integrator.set('x', x0)
        integrator.set('u', simU[i, :])
        integrator.solve()
        x0 = integrator.get('x')
        states[i+1,0:2] = x0
        states[i+1,2] = obj.alpha_ten*x0[0] + obj.alpha_h*x0[1]



    plt.plot(states[:, 0], states[:, 2])

    plt.show()

    plt.plot(t_array, states[0:num_sim_time, 1])
    plt.plot(t_array, states[0:num_sim_time, 0])
    plt.plot(t_array, states[0:num_sim_time, 2])

    plt.show()



