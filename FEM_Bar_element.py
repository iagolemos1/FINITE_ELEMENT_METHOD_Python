# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright (c) 2020 Iago Pereira Lemos
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#-----------------------------------------------------------------------------
# Finite Element Method - Bar Element Solver for Axial Charges
# Created by: Iago Pereira Lemos (lemosiago123@gmail.com)
# Federal University of Uberlândia, School of Mechanical Engineering
#-----------------------------------------------------------------------------

"""
Import numpy library
"""
import numpy as np

#------------------------------------------------------------------------------
"""
Defining and initializing the finite element method model class object
"""
class fem_model(object):
    #*model_params: properties of the model, boundary conditions, imposed forces
    #-----check the example for runing the solver in the githug repository-----
    def __init__(self, *model_params): 
        self.nb_ele  = model_params[0].shape[0] #number of elements
        self.nb_node = self.nb_ele + 1          #number of nodes
        self.properties = model_params[0]       #matrix of properties
        self.bc = model_params[1]               #matrix of boundary conditions
        self.imp_F = model_params[2]            #matrix of imposed forces
        
        #Constructing the conectivite matrix
        conec_mat = np.empty((self.nb_ele, 2))
        for i in range(0, conec_mat.shape[0]):
            conec_mat[i,:] = [i, i+1]  
            
        self.conec_mat = conec_mat
#------------------------------------------------------------------------------
    """
    Function for computing and construct the stiffness global matrix
    by using the assembly of the stiffness global matrix process algorithm
    """        
    def K_global(self):
        nb_node = self.nb_node
        properties = self.properties
        nb_ele = self.nb_ele
        conec_mat = self.conec_mat
        K_global = np.zeros((nb_node, nb_node))
        for i in range(0, int(nb_ele)):
           coef = properties[i,0]*properties[i,1]/properties[i,2]
           K_ele = coef*np.array([[1, -1],[-1, 1]])
           eye_mat = np.eye(nb_node)
           transf_mat = np.array([eye_mat[int(conec_mat[i,0]),:],eye_mat[int(conec_mat[i,1]),:]])
           K_global = np.add(K_global, np.matmul(np.matmul(transf_mat.transpose(),K_ele), transf_mat))
    
        return K_global
#------------------------------------------------------------------------------
        
    """
    Function for solving and run the calculations for the defined model
    """ 
    
    def solve_model(self, *result_type):
        properties = self.properties
        nb_node = self.nb_node
        nb_ele = nb_node - 1
        imp_F = self.imp_F
        bc = self.bc
        dof_free = np.array(imp_F[:,0])   #taking the free nodes
        dof_imp  = np.array(bc[:,0])      #taking the imposed nodes
        F_dof_free = np.array(imp_F[:,1]) #taking the force in the free nodes
        displ_dof_imp = np.array(bc[:,1]) #taking the displacement in the imposed nodes
        
        K_global = fem_model.K_global(self) #construct the stiffness global matrix
        
        
        #Solving the model
        K_ff = np.empty((len(dof_free), len(dof_free)))
        for i in range(0, len(dof_free)):
            for j in range(0, len(dof_free)):
                K_ff[i,j] = K_global[dof_free[i], dof_free[j]]
        
        K_fi = np.empty((len(dof_free), len(dof_imp)))
        for i in range(0, len(dof_free)):
            for j in range(0, len(dof_imp)):
                K_fi[i,j] = K_global[dof_free[i], dof_imp[j]]
        
        K_ii = np.empty((len(dof_imp), len(dof_imp)))
        for i in range(0, len(dof_imp)):
            for j in range(0, len(dof_imp)):
                K_ii[i,j] = K_global[dof_imp[i], dof_imp[j]]
            
        inv_K_ff = np.linalg.inv(K_ff)
        mult = np.matmul(K_fi, displ_dof_imp)
        sub = np.subtract(F_dof_free, mult)
        displ_free = np.matmul(inv_K_ff, sub)
        
        self.F_reaction = np.add(np.matmul(K_fi.transpose(), displ_free),np.matmul(K_ii,displ_dof_imp))
        
        self.disp = np.zeros((nb_node, 1))
        self.disp[dof_free, 0] = displ_free
        
        self.delta_disp= np.diff(self.disp, axis = 0)
        
        self.stress = np.empty((nb_ele, 1))
       
        for i in range(0, nb_ele):
            self.stress[i,0] = self.delta_disp[i,0]*properties[i,0]/properties[i,2]
        
        self.strain = np.empty((properties.shape[0], 1))
        self.strain[:,0] = self.delta_disp[:,0]/self.properties[:,2]
        
        
        #Outputing the results
        try:
    
            if result_type[0] == 'displacement':
                for i in range(0, self.disp.shape[0]):
                    print('\nNode: {}'.format(i))
                    print('Displacement: {:.4E} [m]\n'.format(self.disp[i,0]))
                    
                return(self.disp)
            
            elif result_type[0] == 'strain':
                for i in range(0, nb_ele):
                    print('\nElement: {}'.format(i))
                    print('Strain: {:.4}\n'.format(self.strain[i,0]))
                    
                return(self.strain)
                    
            elif result_type[0] == 'stress':
                for i in range(0, nb_ele):
                    print('\nElement: {}'.format(i))
                    print('Stress: {:.4} [Pa]\n'.format(self.stress[i,0]))
            
            return(self.stress)
        
        except IndexError:
            return(self.disp, self.stress, self.strain)

#------------------------------------------------------------------------------        
                

                
 
        
        
        

    