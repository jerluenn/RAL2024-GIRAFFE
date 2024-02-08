from re import X
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


    def __init__(self, b, m, mu, gamma, k, sigma, k_env): 

        self.b = b 
        self.m = m 
        self.mu = mu 
        self.gamma = gamma 
        self.k = k 
        self.sigma = sigma
        self.k_env = k_env
    
    def createModel(self): 

        model_name = 'model_three_mass'

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
            (F_act - self.b[0]*(x1dot) - self.k*(x1 - x3))/self.m[0], 
            # 0,
            (- self.k_env*x2 - self.b[1]*(x2dot) - self.k*(x2 - x3))/self.m[1], 
            (Ff - self.b[2]*(x3dot) - self.k*(-x1 - x2 + 2*x3))/self.m[2], 
            self.sigma*(x3dot - ((Ff*self.mod_approx(x3dot))/(self.k*(x2-x1)*self.mu*gamma + 1e-6)))
        ) 

        x = vertcat(x1, x2, x3, x1dot, x2dot, x3dot, Ff)
        u = vertcat(F_act)
        params = vertcat(gamma)

        model = AcadosModel()

        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = model_name        
        model.p = params

        return model

    #Tendon-SheathActuatedRobotsandTransmissionSystem

    def sgn_approx(self, x1): 

        epsilon = 1e-10

        return tanh((x1)/epsilon)

    def mod_approx(self, x):

        epsilon = 1e-10

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

        ocp.model.cost_y_expr = vertcat(model.x[1] - model.x[2], model.u)
        ocp.model.cost_y_expr_e = vertcat(model.x[1] - model.x[2])
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
        ocp.parameter_values = np.array([self.gamma])

        # set prediction horizon
        ocp.solver_options.tf = Tf

        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator

def sim_example():  

    # b, m, mu, gamma, k, sigma

    b = np.array([3500, 100, 1000])
    m = np.array([2.0, 1.3e-5, 0.05])
    mu = 0.45
    gamma = 1
    k = 7.54e4
    sigma = 1.2e3
    k_env = 500
    ref_tension1 = 10
    ref_tension2 = 0.5

    Tf = 0.0001
    N = 40 

    time_step = Tf/N

    obj = Controller(b, m, mu, gamma, k, sigma, k_env)
    solver, integrator = obj.createSolver(np.zeros(7), 30, N, 1, Tf)

    x0 = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0])
    num_sim_time = 50000
    
    states = np.zeros((num_sim_time+1, 7))
    simU = np.zeros((num_sim_time, 1))
    t_array = np.zeros(num_sim_time)
    t = 0 

    for i in range(num_sim_time): 

        t += time_step
        t_array[i] = t

        # solver.set(0, 'lbx', x0)
        # solver.set(0, 'ubx', x0)
        # solver.solve() # Testing the solver.

        freq = 0.01

        if i > 750 : 
        # if i > num_sim_time : 

            # u = 5*np.sin(i*freq) + 5
            u = ref_tension2

        else: 

            u = ref_tension1

        simU[i, :] = u

        integrator.set('x', x0)
        integrator.set('u', simU[i, :])
        integrator.set('p', gamma)
        integrator.solve()
        x0 = integrator.get('x')
        states[i+1,:] = x0

        print(x0)

    print("Total tension input: ", ref_tension2)
    print("Expected tension output: ", ref_tension2*np.exp(-mu*gamma))
    print("Total tension loss: ", ref_tension2 - ref_tension2*np.exp(-mu*gamma))
    print("Steady state friction: ", x0[6])

    plt.plot(t_array, states[0:num_sim_time, 6])
    # plt.plot(t_array, simU)
    plt.show()
    plt.plot(t_array, states[0:num_sim_time, 0:6])
    plt.legend(['x1', 'x2', 'x3', 'x1dot', 'x2dot', 'x3dot'])
    plt.show()
    plt.plot(t_array, -k*((states[0:num_sim_time, 1]) - (states[0:num_sim_time, 2])))
    plt.plot(t_array, -k*((states[0:num_sim_time, 2]) - (states[0:num_sim_time, 0])))
    plt.show()
    plt.plot(-k*((states[0:num_sim_time, 2]) - (states[0:num_sim_time, 0])),-k*((states[0:num_sim_time, 1]) - (states[0:num_sim_time, 2])))
    # plt.plot(t_array, k_env*((states[0:num_sim_time, 2])))
    plt.show()

sim_example()