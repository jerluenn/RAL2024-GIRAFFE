from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from generate_robot_arm_model import Robot_Arm_Model

import time

from matplotlib import pyplot as plt

class Multiple_Shooting_Solver:

    def __init__(self, robot_arm_model): 
    
        self._robot_arm_model = robot_arm_model
        self._boundary_length = self._robot_arm_model.get_boundary_length()
        self._integration_steps = self._robot_arm_model.get_num_integration_steps()
        self._MAX_ITERATIONS = 1000
        self.create_static_solver()
        self.create_integrator_with_curvature()

    def create_static_solver(self):

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_static_robot_arm_model()
        self.nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = self.nx + nu

        x = self.ocp.model.x
        u = self.ocp.model.u
        # n_p = self.ocp.model.p.size()[0]

        self.ocp.dims.N = self._integration_steps
        self.ocp.solver_options.qp_solver_iter_max = 400
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # self.ocp.model.cost_y_expr_e = vertcat(self._robot_arm_model.get_tendon_point_force_opposite_direction() - x[7:13])
        self.ocp.model.cost_y_expr_e = vertcat(self._robot_arm_model.get_tendon_point_force() - x[7:13])
        self.ocp.cost.W_e = np.identity(6)
        self.ocp.cost.yref_e = np.zeros((6))

        self.ocp.solver_options.sim_method_num_steps = self._integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2
        # self.ocp.parameter_values = np.zeros((n_p))

        self.ocp.solver_options.levenberg_marquardt = 0.00001

        self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length

        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50
        self.curvature_ub = 100

        self.ocp.constraints.idxbx_0 = np.arange(self.nx)

        self.ocp.constraints.lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_lb*np.ones(6), np.zeros(self.nx - 13)))

        self.ocp.constraints.ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_ub*np.ones(6), np.zeros(self.nx - 13)))        

        self.ocp.constraints.idxbx = np.arange(self.nx)

        self.ocp.constraints.lbx = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))

        self.ocp.constraints.ubx = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.ones(self.nx-13)*self.tension_max))

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 100

        self.ocp.code_export_directory = 'ocp_solver' + self.ocp.model.name

        self._x_upper_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_ub*np.ones(6), np.zeros(self.nx - 13)))  
        self._x_lower_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_lb*np.ones(6), np.zeros(self.nx - 13)))

        self._x_lower = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))
        self._x_upper = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.zeros(self.nx-13)))

        # AcadosOcpSolver.generate(self.ocp, json_file=f'{self.ocp.model.name}.json')
        # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{self.ocp.model.name}.json')
        self._solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        self._integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        self._state_vector = np.zeros((self._integration_steps+1, self.nx+1))

        return self._solver, self._integrator
    
    def create_integrator_with_curvature(self): 

        self._integrator_with_curvature = self._robot_arm_model._create_static_integrator_with_curvature()

    def set_tensions(self, tension): 

        self._x_lower[13:13+np.size(tension)] = tension
        self._x_upper[13:13+np.size(tension)] = tension

        self._x_lower_0[13:13+np.size(tension)] = tension
        self._x_upper_0[13:13+np.size(tension)] = tension

        self._solver.constraints_set(0, 'lbx', self._x_lower_0)
        self._solver.constraints_set(0, 'ubx', self._x_upper_0)

        for i in range(self._integration_steps-1): 

            self._solver.constraints_set(i+1, 'lbx', self._x_lower)
            self._solver.constraints_set(i+1, 'ubx', self._x_upper)

    def initialise_solver(self, initial_solution, tension): 

        initial_solution = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), initial_solution, tension))

        self._solver.set(0, 'x', initial_solution)

        initial_x_upper = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), self.wrench_ub*np.ones(6), tension))
        initial_x_lower = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), self.wrench_lb*np.ones(6), tension))

        self._solver.constraints_set(0, 'lbx', initial_x_lower)
        self._solver.constraints_set(0, 'ubx', initial_x_upper)

        subseq_solution = initial_solution

        for i in range(self._integration_steps): 

            self._integrator.set('x', subseq_solution)
            self._integrator.solve()
            subseq_solution = self._integrator.get('x')

            self._solver.set(i+1, 'x', subseq_solution)  

        print("Initial solution at distal end: ",subseq_solution)

    def integrate_one_step_with_curvature_integrator(self, state): 

        self._integrator_with_curvature.set('x', state)
        self._integrator_with_curvature.solve()

        return self._integrator_with_curvature.get('x')

    def integrate_with_curvature_integrator(self):

        for i in range(self._integration_steps): 

            self._state_vector[i+1, :] = self.integrate_one_step_with_curvature_integrator(self._state_vector[i, :]) 

    def solve_static(self): 

        for i in range(self._MAX_ITERATIONS): 

            self._solver.solve() 

            if self._solver.get_cost() < 1e-7: 

                break 

        for i in range(self._integration_steps + 1): 

            self._state_vector[i, 0:16] = self._solver.get(i, 'x')

        self.integrate_with_curvature_integrator()
        print("Distal states: ", self._state_vector[-1, :])

        print("Initial static problem solved with cost function: ", self._solver.get_cost())


    def get_states(self): 

        return self._state_vector

    def get_curvature(self): 

        return self._state_vector[:, -1]
    
    def plot_data(self): 

        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.4, 0)
        ax.plot(self._state_vector[:, 0], self._state_vector[:, 1], self._state_vector[:, 2])
        plt.show()
