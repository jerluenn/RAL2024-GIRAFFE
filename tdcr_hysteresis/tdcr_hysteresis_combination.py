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

    def __init__(self, multiple_shooting_solver, friction_model_integrator): 

        self._forward_kinematics_solver = multiple_shooting_solver
        self._friction_model = friction_model_integrator
        self._initialise_states()

    def _initialise_states(self): 

        self._x0 = np.zeros(self._friction_model.num_elements)
        self._curvature = np.zeros(self._forward_kinematics_solver._integration_steps)

    def solve(self, u): 

        # get curvature 
        self._friction_model_integrator.solve_next_step(u, self._curvatures, self._x0)
        self._x0 = self._friction_model_integrator.get('x') #fix this too. 
        self._tension = self._friction_model.compute_tension(self._x0, u)
        # update tension
        self._forward_kinematics_solver.set_tensions(self._tension)
        # solve for shape
        self._forward_kinematics_solver.solve_static()

        return self._forward_kinematics_solver.get_states()

    def set_curvatures(self, curvatures): 

        self._curvature = curvatures