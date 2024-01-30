from casadi import *
import numpy as np
import scipy

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

    def __init__(self, rho, sigma, gamma_tension, n, gamma, mu, alpha): 

        self.rho = rho
        self.sigma = sigma
        self.gamma_tension = gamma_tension
        self.n = n
        self.gamma = gamma 
        self.mu = mu
        self.alpha = alpha

    def createModel(self): 

        model_name = 'hysteresis_model'

        tension = SX.sym('tension')
        h = SX.sym('h')

        x = vertcat(tension, h)
        
        tensiondot = SX.sym('tensiondot')
        hdot = SX.sym('hdot')

        xdot = vertcat(tensiondot, hdot)

        tension_des = SX.sym('tension_des')
        u = vertcat(tension_des)

        tension_dot = (1/self.gamma_tension)*(tension_des - tension)

        Ff = SX.sym('Ff')

        z = vertcat(Ff)

        phi = 0

        Ff_ss = (1 - exp(-self.mu*self.gamma))*tension

        h_dot = self.rho*(tension_dot - self.sigma*self.mod_approx(tension_dot)*h*self.mod_approx(h)**(self.n-1) - (self.sigma - 1)*tension_dot*self.mod_approx(h)**self.n)

        f1 = 0
        f2 = 0

        Ff_expr = self.mod_approx(tension_dot)*((1-self.alpha[0])*Ff_ss + self.alpha[0]*Ff_ss*exp(-f1)) + self.mod_approx(-tension_dot)*((1-self.alpha[1])*Ff_ss + self.alpha[1]*Ff_ss*exp(-f2))

        f_impl = vertcat(tensiondot - tension_dot, hdot - h_dot, Ff - Ff_expr)
        f_expl = vertcat(tension_dot, self.rho*(tension_dot - self.sigma*self.mod_approx(tension_dot)*h*self.mod_approx(h)**(self.n-1) - (self.sigma - 1)*tension_dot*self.mod_approx(h)**self.n))

        model = AcadosModel()

        # params = vertcat(alpha_tension, alpha_h)

        model.z = z
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.xdot = xdot
        model.x = x
        model.u = u
        model.name = model_name        
        # model.p = params

        return model

    def createSolver(self, x0, max_tension, N_horizon, RTI, Tf): 

        ocp = AcadosOcp()

        model = self.createModel()

        # nx = model.x.size()[0]
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        nz = model.z.size()[0]
        ny = nu + nx
        ny_e = nx

        ocp.model = model

        Q = np.diag([1, 1])
        R = np.diag([1])

        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        Vx = np.zeros((ny, nx))
        Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        ocp.cost.Vu = Vu

        Vz = np.zeros((ny, nz))
        Vz[0, 0] = 1.0

        ocp.cost.Vz = np.zeros((ny, nz))

        Q_e = np.diag([1, 1])
        ocp.cost.W_e = Q_e

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)

        ocp.cost.Vx_e = Vx_e

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        ocp.constraints.lbu = np.array([-max_tension])
        ocp.constraints.ubu = np.array([max_tension])

        # ocp.cost.W = np.diag([10, 0.01])
        # ocp.cost.W_e = np.diag([100])

        ocp.constraints.x0 = x0
        ocp.constraints.idxbu = np.array([0])

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.levenberg_marquardt = 1.0

        if RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.dims.N = N_horizon
        # ocp.solver_options.qp_solver_cond_N = N_horizon
        # ocp.parameter_values = np.array([self.alpha_ten, self.alpha_h])

        # set prediction horizon
        ocp.solver_options.tf = Tf

        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator


    def mod_approx(self, x):

        epsilon = 1e-8

        return sqrt(x**2 + epsilon)



def sim_example(): 

    # rho, sigma, gamma_tension, n, gamma, mu, alpha

    alpha = np.array([0.5, 0.5])

    obj = Hysteresis_MPC_Controller(7.0, 20.0, 3.0, 4, 0.45, 2, alpha)
    solver, integrator = obj.createSolver(np.zeros(2), 30, 40, 1, 2)

    x0 = np.zeros(2)
    num_sim_time = 1000

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

            u = 0
            # u = 0

        else: 

            u = 10 

        simU[i, :] = u

        print(simU[i, :])

        integrator.set('x', x0)
        integrator.set('u', simU[i, :])
        integrator.solve()
        x0 = integrator.get('x')
        z = integrator.get('z')
        states[i+1,0:2] = x0
        states[i+1,2] = z



    plt.plot(states[:, 0], states[:, 2])

    plt.show()

    plt.plot(t_array, states[0:num_sim_time, 1])
    plt.plot(t_array, states[0:num_sim_time, 0])
    plt.plot(t_array, states[0:num_sim_time, 2])

    plt.show()


def control_example(): 

    obj = Hysteresis_MPC_Controller(0.4, -0.1, 3.0, 1.0, 1.0, 3)
    solver, integrator = obj.createSolver(np.zeros(2), 30, 40, 1, 2)

    x0 = np.zeros(2)
    num_sim_time = 1000

    states = np.zeros((num_sim_time+1, 3))
    simU = np.zeros((num_sim_time, 1))
    simRef = np.zeros((num_sim_time, 1))
    t_array = np.zeros(num_sim_time)
    t = 0 
    freq = 0.05


    for i in range(num_sim_time): 

        for k in range(20): 

            solver.cost_set(k, 'yref', np.array([5*np.sin(freq*i)+5, 0]))

        t += 0.05
        t_array[i] = t
        simRef[i, :] = 5*np.sin(freq*i)+5
        solver.set(0, 'lbx', x0)
        solver.set(0, 'ubx', x0)

        solver.solve()

        simU[i, :] = solver.get(0, 'u')

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

    plt.show()

    plt.plot(t_array, states[0:num_sim_time, 2])
    plt.plot(t_array, simRef)

    plt.show()

sim_example()
# control_example()
