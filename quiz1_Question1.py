# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:50:15 2017

@author: Chinmay Rawool
"""

import numpy as np
from numpy import linalg as LA 

import pandas as pd
import matplotlib.pyplot as plt

Y = pd.read_csv('D:\Coding\Python\ML in class\dataset_1.csv', sep=',',header=0);
Y

data = np.array(Y,dtype=float)
data

#variance of x, y,z
var_x = np.var(data[:,0])
var_y = np.var(data[:,1])
var_z = np.var(data[:,2])

cov_xy = np.cov(data[:,0],data[:,1])
cov_xy

cov_yz = np.cov(data[:,1],data[:,2])
cov_yz


mean_x = np.mean(data[:,0])
mean_y = np.mean(data[:,1])
mean_z = np.mean(data[:,2])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

y = np.ones((1000,3),dtype=float)
y

y = y * np.transpose(mean_vector)
y

mean_centered_data = data - y
mean_centered_data

covMat = np.cov(mean_centered_data)
covMat

cov = np.cov([mean_centered_data[:,0],mean_centered_data[:,1],mean_centered_data[:,2]])
cov

eig_val_cov, eig_vec_cov = LA.eig(cov)
eig_val_cov
eig_vec_cov


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)


#P matrix in sorted value
w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1),eig_pairs[2][1].reshape(3,1)))
print('Matrix W:\n', w)

#PCA Y = X * P
Y_output = np.matmul(data,w)
Y_output

fig = plt.figure()
aX = fig.add_subplot(1,1,1)
aX.scatter(Y_output[:,0],Y_output[:,1])
fig.show()