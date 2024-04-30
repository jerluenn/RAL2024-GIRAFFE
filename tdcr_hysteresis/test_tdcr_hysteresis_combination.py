import tdcr_hysteresis_combination 
import sys
import numpy as np
from casadi import *
import pdb

sys.path.insert(0, "../tdcr")
sys.path.insert(0, "../mass_spring_damper")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params

from mass_spring_damper_friction_model import Controller

if __name__ == "__main__": 

    """Create a CR Multiple Shooting solver first. """   

    # 12 mm in y direction, 22.5mm in -x direction. 

    tendon_radiuses_list = [[-0.0225, -0.012, 0], [-0.0225, 0.012, 0]]
    tendon_radiuses = SX(tendon_radiuses_list)
    robot_arm_1 = Robot_Arm_Params(0.48, "1", 0.)
    rho = 190000
    robot_arm_1.from_hollow_rod(0.001, 0.0008, 200e9, 70e9, rho)
    robot_arm_1.set_gravity_vector('x')
    robot_arm_1.set_tendon_radiuses(tendon_radiuses_list)
    robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)
    robot_arm_model_1.set_num_integrator_steps(10)
    ms_solver = Multiple_Shooting_Solver(robot_arm_model_1)

    """Create tendons solver"""

    length = 0.47
    b = 100
    num_elements = 10
    radius = 0.4*1e-3
    density = 26.0
    area = np.pi*radius**2
    mass = density*area*length
    mass_per_element = mass/num_elements
    E = 1.2e11
    k = E*area/length
    sigma = 7.5e6
    mu = 0.45
    gamma = 1.75*np.ones((1, num_elements))

    Tf = 0.01
    N = 40

    step_size = Tf/N

    obj = Controller(b, mass_per_element, mu, gamma, k, sigma, num_elements)
    obj2 = Controller(b, mass_per_element, mu, gamma, k, sigma, num_elements)

    integrator_list = [obj, obj2]

    """ """

    TDCR_hysteresis_obj = tdcr_hysteresis_combination.TDCR_With_Hysteresis(ms_solver, integrator_list)
    input_test = np.array([1, 0])
    TDCR_hysteresis_obj.solve_equal_elements(input_test)

    print(TDCR_hysteresis_obj._forward_kinematics_solver._solver.get_residuals())

    breakpoint()

    