#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:53:04 2017

@author: rodrigoazs
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import linalg
from numpy.polynomial import Polynomial

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Importing train dataset
dataset = pd.read_table('Dados-medicos.txt')

############################

X = dataset.values[:,2]
y = dataset.values[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=172, random_state=0)

X_mean = np.mean(X_train)
X_sigma = np.std(X_train)
y_mean = np.mean(y_train)
y_sigma = np.std(y_train)
Xy_cov = np.cov(X, y)

X_space = np.linspace(0, 400, 500)
y_space = np.linspace(0, 80, 500)
X_mesh, y_mesh = np.meshgrid(X_space, y_space)

#==============================================================================
Z = bivariate_normal(X_mesh, y_mesh, X_sigma, y_sigma, X_mean, y_mean, Xy_cov[0][1])
 
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X_mesh, y_mesh, Z, cmap='plasma', linewidth = 0)
ax.set_xlabel('Carga')
ax.set_ylabel('VO2')
#ax.view_init(27, -21)
#==============================================================================




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[0.0:400.0:500j, 0.0:80.0:500j]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

mu = np.array([X_mean, y_mean])

sigma = np.array([.5, .5])
covariance = np.diag(sigma**2)

z = multivariate_normal.pdf(xy, mean=mu, cov=np.array(Xy_cov))

# Reshape back to a (30, 30) grid.
z = z.reshape(x.shape)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Carga')
ax.set_ylabel('VO2')

ax.plot_surface(x,y,z)
#ax.plot_wireframe(x,y,z)

plt.show()


##############

from scipy.stats import multivariate_normal

a = multivariate_normal.pdf([220, 51.174289246], mean=[X_mean, y_mean], cov=Xy_cov)
