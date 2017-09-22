# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:46:05 2017

@author: Chinmay Rawool
"""
import numpy as np
from numpy import linalg as LA 

a=np.array([[0,-1],[2,3]],dtype=float)
a

eig_val_a, eig_vec_a = LA.eig(a)
eig_val_a,eig_vec_a