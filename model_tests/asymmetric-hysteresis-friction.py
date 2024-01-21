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


class Asymmetric_Hysteresis_MPC_Controller: 

    def __init__(self, rho, mu, sigma, kappa, a): 

        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.sigma = sigma
        self.kappa = kappa
        self.a = a

    def createModel(self): 

        model_name = 'asymmetric_hysteresis_model'

        zeta = SX.sym('zeta')
        x = SX.sym('x')
        x_dot = SX.sym('x_dot')

        states = vertcat(x, x_dot, zeta)
        
        zetadot = SX.sym('zetadot')
        xdot = SX.sym('xdot')
        xdotdot = SX.sym('xdotdot')

        statesdot = vertcat(xdot, xdotdot, zetadot)

        F = SX.sym('F')
        u = vertcat(F)

        Ff = SX.sym('Ff')

        z = vertcat(Ff)

        xdotdot_expl_expr = (-self.a[0]*x - self.a[1]*xdot + F)

        f1 = (self.kappa[0]*self.mod_approx(x_dot))/(self.mod_approx(self.kappa[1]*xdotdot_expl_expr) + 1)
        f2 = (self.kappa[3]*self.mod_approx(x_dot))/(self.mod_approx(xdotdot_expl_expr) + self.kappa[4])

        A1 = self.mu[0]*exp(-f1) 
        A2 = self.mu[1]*(1-exp(self.mod_approx(self.kappa[2]*xdotdot_expl_expr)))
        A3 = self.mu[2]*exp(-f2)

        f_expl = []

        f_impl = vertcat(
                xdot - x_dot,
                xdotdot - (-self.a[0]*x - self.a[1]*xdot + F), 
                zetadot - (xdot - self.sigma*zeta*((self.mod_approx(x_dot))/(Ff + 1e-6))),
                Ff - self.step_approx(x_dot)*(self.rho[0] + A1 + self.step_approx(xdotdot_expl_expr)*A2) - self.step_approx(-x_dot)*(self.rho[1] + self.sgn_approx(xdotdot_expl_expr)*A3)
        )

        model = AcadosModel()

        params = []

        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = statesdot
        model.u = u
        model.z = z
        model.name = model_name        
        model.p = params

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

        Q = np.diag([1, 1, 1])
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

        Q_e = np.diag([1, 1, 1])
        ocp.cost.W_e = Q_e

        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)

        ocp.cost.Vx_e = Vx_e

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)

        # ocp.cost.cost_type = 'NONLINEAR_LS'
        # ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # ocp.cost.cost_type_0 = 'LINEAR_LS'

        # ocp.cost.Vz_0 = np.diag([1])

        # ocp.cost.W_0 = np.diag([1, 1, 10])
        # ocp.model.cost_y_expr = vertcat(model.x[0], model.u)
        # ocp.model.cost_y_expr_e = vertcat(model.x[0])
        # # ocp.model.cost_y_expr_0 = vertcat(model.p[0]*model.x[0] + model.p[1]*model.x[1], model.u)
        # # ocp.cost.yref_0 = np.zeros((ny, ))
        # ocp.cost.yref  = np.zeros((2, ))
        # ocp.cost.yref_e = np.zeros((1, ))

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

        epsilon = 1e-10

        return sqrt(x**2 + epsilon)

    def sgn_approx(self, x): 

        epsilon = 1e-10

        return tanh(x/epsilon)

    def step_approx(self, x): 

        epsilon = 1e-10

        return 0.5 + 0.5*tanh(x/epsilon)



def sim_example(): 

    # rho 2, mu 3, sigma 1, kappa 5, a 2

    a = np.array([1, 2])
    rho = np.array([10, 20])
    # rho = np.zeros(3)
    mu = np.array([10, 0.05, -5])
    # mu = np.zeros(3)
    kappa = np.array([0.172, 1.228, 0.016, 5.549, 0.005])
    sigma = 2.57

    obj = Asymmetric_Hysteresis_MPC_Controller(rho, mu, sigma, kappa, a)
    solver, integrator = obj.createSolver(np.zeros(3), 100, 50, 1, 1)

    x0 = np.zeros(3)
    num_sim_time = 2000

    states = np.zeros((num_sim_time+1, 4))
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

        if i > num_sim_time/2 : 

        # u = 5*np.sin(i*freq) + 5
            u = 0.01
            u = 0

        else: 

            u = 10

        simU[i, :] = u

        print(simU[i, :])

        integrator.set('x', x0)
        integrator.set('u', simU[i, :])
        integrator.solve()
        x0 = integrator.get('x')
        z = integrator.get('z')
        print("states, z: ", x0, z)
        print(integrator.get('time_tot'))
        states[i+1,0:3] = x0
        states[i+1, 3] = z
        # states[i+1,2] = obj.alpha_ten*x0[0] + obj.alpha_h*x0[1]



    plt.plot(states[:, 0], states[:,0] - states[:, 2])

    plt.show()

    plt.plot(t_array, states[0:num_sim_time, 1])
    plt.plot(t_array, states[0:num_sim_time, 0])
    plt.plot(t_array, states[0:num_sim_time, 2])

    plt.show()

    plt.plot(t_array, states[0:num_sim_time, 3])

    plt.show()


sim_example()