import numpy as np
import time
from casadi import * 

class Robot_Arm_Params: 

    def __init__(self, robot_arm_length, _id, mass_body): 

        self._mass_body = mass_body
        self._robot_arm_length = robot_arm_length
        self._created = 0 
        self._id = _id
        self._g_direction = 'z'

    def from_custom(self, Kse, Kbt):

        self._Kse = Kse 
        self._Kbt = Kbt 

    def from_hollow_rod(self, _or, _ir, shear_mod, E, rho): 

        area = np.pi * _or**2 - np.pi * _ir**2
        I = ((np.pi * self._or**4) / 4) - \
            ((np.pi * self._ir**4) / 4)
        J = 2 * self._I
        self._Kse = diag([shear_mod * area,
                        shear_mod * area, E * area])
        self._Kbt = diag([E * I, E *
                        I, shear_mod * J])
        self._mass_distribution = rho * area

    def from_solid_rod(self, r, shear_mod, E, rho):

        area = np.pi * r**2
        I = (np.pi * r**4) / 4
        J = 2 * I
        self._Kse = diag([shear_mod * area,
                        shear_mod * area, E * area])
        self._Kbt = diag([E * I, E *
                        I, shear_mod * J])
        self._mass_distribution = rho * area
        self._rho = rho
        self._J = J 
        self._I = I

    def get_mass_body(self): 

        return self._mass_body

    def set_tendon_radiuses(self, r): 

        self._tendon_radiuses = SX(r)
        self._tendon_radiuses_numpy = np.array(r)

    def set_damping_coefficient(self, C): 

        self._C = C 

    def set_damping_factor(self, Bbt, Bse): 

        self._Bbt = Bbt
        self._Bse = Bse

    def set_rotational_inertia_matrix(self, J): 

        self._J = J

    def set_mass_distributions(self, mass_distribution): 

        self._mass_distribution = mass_distribution

    def set_rho(self, rho): 

        self._rho = rho

    def get_tendon_radiuses(self): 

        return self._tendon_radiuses

    def get_id(self):

        return self._id

    def get_J(self): 

        return self._J

    def get_Bbt(self): 

        return self._Bbt

    def get_Bse(self): 

        return self._Bse

    def get_arm_length(self): 

        return self._robot_arm_length 

    def get_Kse(self):

        return self._Kse

    def get_Kbt(self): 

        return self._Kbt

    def get_C(self): 

        return self._C 

    def get_mass_distribution(self): 

        return self._mass_distribution

    def get_rho(self): 

        return self._rho

    def set_gravity_vector(self, vector): 

        self._g_direction = vector