import sys
import numpy as np
import time
from casadi import *

sys.path.insert(0, "../tdcr")
sys.path.insert(0, "../mass_spring_damper")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

from mass_spring_damper_friction_model import Controller 

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TDCR_With_Hysteresis: 

    def __init__(self, multiple_shooting_solver, friction_model_integrator_list): 

        self._forward_kinematics_solver = multiple_shooting_solver
        self._friction_model_integrator_list = friction_model_integrator_list
        self._num_tendons = len(self._friction_model_integrator_list)
        self._initialise_states()

    def _initialise_states(self): 

        if self._num_tendons != self._forward_kinematics_solver._robot_arm_model._num_tendons:

            raise ValueError("Check number of tendons vs number of friction integrators!")    

        # Set num_integration_steps = num_states/3 in friction integrator? 
        # Assume they are equal for now, i

        self._tension = np.zeros((self._forward_kinematics_solver._robot_arm_model.get_num_integration_steps(), self._num_tendons))
        self._x0_friction_model = np.zeros((self._friction_model_integrator_list[0].num_elements*3, self._num_tendons))
        self._curvature = np.zeros(self._forward_kinematics_solver._integration_steps+1)
        self._local_curvature = np.zeros(self._forward_kinematics_solver._integration_steps)
        self._initial_guess = np.zeros(6)

        self._forward_kinematics_solver.initialise_solver(self._initial_guess, self._tension[0, :])
        self._forward_kinematics_solver.solve_static()
        self._set_curvature_from_solver_equal_elements()

    def solve_unequal_elements(self, u): 

        pass 

    def solve_equal_elements(self, u): 

        """u must be n_tendons*1 shape."""

        for i in range(self._num_tendons):

            self._x0_friction_model[:, i] = self._friction_model_integrator_list[i].solve_next_step(u[i:i+1], self._local_curvature, self._x0_friction_model[:, i])
            self._tension[:, i] = self._friction_model_integrator_list[i].compute_tension(self._x0_friction_model[:, i], u[i])

        breakpoint()

        # update tension
        self._forward_kinematics_solver.set_vector_tensions(self._tension)
        # solve for shape
        self._forward_kinematics_solver.solve_static()
        self._set_curvature_from_solver_equal_elements()

        return self._forward_kinematics_solver.get_states()

    def set_curvature(self, curvature): 

        self._curvature = curvature
        self._local_curvature = np.diff(self._curvature)

    def _set_curvature_from_solver_equal_elements(self): 

        self._curvature = self._forward_kinematics_solver.get_states()[:, -1]
        self._local_curvature = np.diff(self._curvature)

    def _set_curvature_from_solver_unequal_elements(self): 

        # self._curvature = self._forward_kinematics_solver.get_states()[:, -1]

        pass 

    def plot_current_states(self): 

        pass 