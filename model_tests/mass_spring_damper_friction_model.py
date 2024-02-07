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

from scipy import linalg

class Controller: 


    def __init__(self, b, m, mu, gamma, k, sigma, num_elements): 

        self.b = b 
        self.m = m 
        self.mu = mu 
        self.gamma = gamma 
        self.k = k 
        self.sigma = sigma
        self.num_elements = num_elements
    
    def createModel(self): 

        model_name = 'mass_spring_damper_friction'

        p = SX.sym('p', self.num_elements)
        v = SX.sym('v', self.num_elements)
        Ff = SX.sym('Ff', self.num_elements)
        T_in = SX.sym('T_in', 1)
        F_c = SX.sym('F_c', 1)

        gamma = SX.sym('gamma')
        mu = SX.sym('mu')

        B = 2*self.b*SX.eye(self.num_elements)
        B[0] = B[0]/2

        for i in range(self.num_elements-1): 

            B[i, i+1] = -self.b
            B[i+1, i] = -self.b

        M = self.m*SX.eye(self.num_elements)
        D = SX.zeros((self.num_elements, self.num_elements+1))

        for i in range(self.num_elements): 

            D[i, i] = 1
            D[i, i+1] = -1

        K = SX.zeros((self.num_elements+1, self.num_elements))

        for i in range(self.num_elements): 

            K[i+1, i] = self.k
            K[i, i]   = -self.k
        
        K[0, 0] = 0

        T_in_array = SX.zeros(self.num_elements+1)
        T_in_array[0] = T_in

        p_dot = v 
        v_dot = inv(M)@(-B@v + D@(K@p + T_in_array) - Ff) 
        Ff_dot = self.sigma*(v - Ff*self.mod_approx(v)/(F_c + 1e-8))
        # Ff_dot = SX.zeros(self.num_elements)


        impl_terms = SX.sym('impl_terms', self.num_elements*3)

        u = vertcat(T_in)
        x = vertcat(p, v, Ff)
        z = F_c
        f_expl = vertcat(p_dot, v_dot, Ff_dot)
        f_impl = vertcat(impl_terms - f_expl, F_c - T_in*exp(-mu*gamma*self.sgn_approx(v[0])))

        params = vertcat(gamma, mu)

        model = AcadosModel()

        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = impl_terms
        model.u = u
        model.name = model_name        
        model.p = params
        model.z = z 

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

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        nz = model.z.size()[0]
        ny = nu + nx
        ny_e = nx

        ocp.model = model

        # Q = np.diag([1, 1, 1])
        # R = np.diag([1])
        Q = np.eye(nx)
        R = np.eye(nu)

        ocp.cost.W = linalg.block_diag(Q, R)

        Vx = np.zeros((ny, nx))
        Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx = Vx

        Vu = np.zeros((ny, nu))
        ocp.cost.Vu = Vu

        Vz = np.zeros((ny, nz))
        Vz[0, 0] = 1.0

        ocp.cost.Vz = np.zeros((ny, nz))

        Q_e = np.eye(nx)
        ocp.cost.W_e = Q_e

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)

        ocp.cost.Vx_e = Vx_e
        ocp.parameter_values = np.array([self.gamma, self.mu])

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        x0 = np.zeros(nx)

        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.x0 = x0

        ocp.constraints.lbu = np.array([-max_tension])
        ocp.constraints.ubu = np.array([max_tension])

        # ocp.constraints.idxbx = np.array([i for i in range(nx)])

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
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
        ocp.parameter_values = np.array([self.gamma, self.mu])

        # set prediction horizon
        ocp.solver_options.tf = Tf

        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

        return acados_ocp_solver, acados_integrator

def sim_example():  

    # b, m, mu, gamma, k, sigma

    length = 2
    b = 100
    num_elements = 5
    radius = 0.4*1e-3
    density = 26.0
    area = np.pi*radius**2
    mass = density*area*length
    mass_per_element = mass/num_elements
    E = 1.5e11
    k = E*area/length
    sigma = 1.5e3
    mu = 0.45
    gamma = 0.1

    Tf = 0.001
    N = 40

    step_size = Tf/N

    # b = 100
    # mass_per_element = 0.1 
    # k = 100 
    # num_elements = 3


    obj = Controller(b, mass_per_element, mu, gamma, k, sigma, num_elements)
    solver, integrator = obj.createSolver(np.zeros(7), 30, N, 1, Tf)

    x0 = np.zeros(3*num_elements)
    num_sim_time = 2000
    
    states = np.zeros((num_sim_time+1, 3*num_elements))
    simU = np.zeros((num_sim_time, 1))
    tension_states = np.zeros((num_sim_time+1, num_elements+1))
    t_array = np.zeros(num_sim_time)
    t = 0 

    u0 = np.linspace(0, 10, int(num_sim_time/2))
    u1 = np.linspace(10, 0, int(num_sim_time/2))

    for i in range(num_sim_time): 

        t += step_size
        t_array[i] = t

        # solver.set(0, 'lbx', x0)
        # solver.set(0, 'ubx', x0)
        # solver.solve() # Testing the solver.

        freq = 0.002

        if i > num_sim_time/2 : 
        # if i > num_sim_time : 

            # u = 5*np.sin(i*freq) + 5
            u = 0

        else: 

            u = 10

        simU[i, :] = u

        integrator.set('x', x0)
        integrator.set('u', simU[i, :])
        integrator.set('p', np.array([gamma, mu]))
        print(integrator.get('z'))
        integrator.solve()
        x0 = integrator.get('x')
        states[i+1,:] = x0

        # print(x0)

    tension_states[0:num_sim_time, 0] = simU[:, 0]
    tension_states[:, num_elements] = k*(states[:, num_elements-1])

    for i in range(num_elements - 1): 
        
        tension_states[:, i+1] = k*(states[:, i] - states[:, i+1])


    plt.plot(t_array, states[0:num_sim_time, 0:2*num_elements])
    # plt.plot(t_array, simU)
    # plt.show()
    # plt.plot(t_array, states[0:num_sim_time, 0:6])A
    plt.legend(['x1', 'x2', 'x3', 'x1dot', 'x2dot', 'x3dot'])
    plt.show()

    plt.plot(t_array, states[0:num_sim_time, 2*num_elements:3*num_elements])
    plt.show()

    plt.plot(t_array, tension_states[0:num_sim_time, :])
    plt.show()

    plt.plot(tension_states[0:num_sim_time, 1], tension_states[0:num_sim_time, -1])
    plt.show()

sim_example()