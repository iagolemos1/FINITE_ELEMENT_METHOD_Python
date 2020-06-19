# -*- coding: utf-8 -*-
"""
@author: Iago Pereira Lemos
"""
import numpy as np
from FEM_Bar_element import fem_model


# Constants values
E = 2.1E11   #Young's modulus [N/m^2]
A = 0.0005   #Area [m^2]
L = 0.2      #Length [m]
                       
properties = np.array([[E,   A,   L],
                       [2*E, A,   L],
                       [E,   A/2, 2*L],
                       [3*E, A,   1.5*L]])
F = 3000 #[N]          

bc = np.array([[0, 0],
               [4, 0]])

imp_F = np.array([[1, F],
                  [2, F],
                  [3, 3*F]])

model = fem_model(properties, bc, imp_F)
strain = model.solve_model('strain')
stress = model.solve_model('stress')
displacement = model.solve_model('displacement')
[displacement, stress, strain] = model.solve_model()
