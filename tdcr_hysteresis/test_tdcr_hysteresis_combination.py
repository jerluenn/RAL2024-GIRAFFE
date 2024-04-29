import tdcr_hysteresis_combination 
import sys
import numpy as np
from casadi import *

sys.path.insert(0, "../tdcr")
sys.path.insert(0, "../mass_spring_damper")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

from mass_spring_damper_friction_model import Controller

if __name__ == "__main__": 

    tendon_radiuses_list = [[0.0175, 0, 0], [-0.00875, 0.0151554, 0], [-0.00875, -0.0151554, 0]]
    tendon_radiuses = SX(tendon_radiuses_list)
    robot_arm_1 = Robot_Arm_Params(0.48, "1", 0.)
    rho = 190000
    robot_arm_1.from_hollow_rod(0.001, 0.0008, 200e9, 70e9, rho)
    robot_arm_1.set_gravity_vector('z')
    robot_arm_1.set_tendon_radiuses(tendon_radiuses_list)
    robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)
    robot_arm_model_1.set_num_integrator_steps(10)
    ms_solver = Multiple_Shooting_Solver(robot_arm_model_1)

    tension = np.array([0.0, 0.0, 0])
    initial_guess = np.zeros(6)

    ms_solver.initialise_solver(initial_guess, tension)
    ms_solver.solve_static()
    ms_solver.plot_data()

    tension = np.array([0.0, 3.0, 0])
    initial_guess = np.zeros(6) 

    ms_solver.set_tensions(tension)
    ms_solver.solve_static()

    tension = np.array([0.0, 10.0, 0])

    ms_solver.set_tensions(tension)
    ms_solver.solve_static()
    ms_solver.plot_data()