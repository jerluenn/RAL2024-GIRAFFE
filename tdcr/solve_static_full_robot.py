import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tendon_radiuses_list = [[0.0175, 0, 0], [-0.00875, 0.0151554, 0], [-0.00875, -0.0151554, 0]]
tendon_radiuses = SX(tendon_radiuses_list)
robot_arm_1 = Robot_Arm_Params(0.15, "1", 0.)
robot_arm_1.from_solid_rod(0.0005, 70e9, 200e9, 8000)
robot_arm_1.set_gravity_vector('x')
robot_arm_1.set_tendon_radiuses(tendon_radiuses_list)
robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)
ms_solver = Multiple_Shooting_Solver(robot_arm_model_1)

tension = np.array([0.0, 0.0, 0])
initial_guess = np.zeros(6)

ms_solver.initialise_solver(initial_guess, tension)
ms_solver.solve_static()
ms_solver.plot_data()

# print(ms_solver.get_states())

tension = np.array([12.0, 0.0, 0])
ms_solver.set_tensions(tension)
ms_solver.solve_static()
ms_solver.plot_data()

print(ms_solver.get_states()[:, -1])