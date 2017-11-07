# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
from numpy.polynomial import Polynomial


# Importing train dataset
dataset = pd.read_table('Dados-medicos.txt')

############################

X = dataset.values[:,2]
y = dataset.values[:,3]

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def X_matrix(data, d):
    X = []
    for i in range(len(data)):
        a = [1.0]
        for j in range(1,d):
            a.append(data[i]**j)
        X.append(a)
    return X
    
def X_matrix2(data, d):
    X = []
    for i in range(len(data)):
        a = [1.0]
        for j in range(1,d):
            for p in range(len(data[i])):
                a.append(data[i][p]**j)
        X.append(a)
    return X

def coefs(b):
    n = []
    for i in range(len(b)):
        n.append(b[i][0])
    return n
        
def plot(poly, X, y):
    dataset_X_bound = [min(X), max(X)]
    dataset_y_bound = [min(y), max(y)]
    poly_points = poly.linspace(1000, [dataset_X_bound[0] - 0.1, dataset_X_bound[1] + 0.1])
    plt.plot(X, y, 'o', label='Data', color='white', fillstyle='full', markeredgecolor='black')
    plt.plot(poly_points[0], poly_points[1], color='green')
    plt.legend()
    plt.axis((dataset_X_bound[0] - 0.1, dataset_X_bound[1] + 0.1, dataset_y_bound[0] - 0.1, dataset_y_bound[1] + 0.1))
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=172, random_state=0)

# questao 1.1
for d in range(2,10):
    X = np.matrix(X_matrix(X_train, d))
    X_t = np.matrix(X_matrix(X_test, d))
    y = (np.matrix(y_train)).transpose()
    b = linalg.inv(X.transpose() * X) * X.transpose() * y
    
    p = np.array(b)
    plot(Polynomial(coefs(p)), X_train, y_train)
    nll_in = 0.5 * (y - X * b).transpose() * (y - X * b) 
    nll_out = 0.5 * ((np.matrix(y_test)).transpose() - X_t * b).transpose() * ((np.matrix(y_test)).transpose() - X_t * b) 
    print('Parametros d='+ str(d) +', ', p)
    print("NLL (in sample): %.2f" % nll_in)
    print("NLL (out sample): %.2f" % nll_out)
    print("Mean squared error (in sample): %.2f"
          % mean_squared_error(y_train, X*b))
    print("Mean squared error (out sample): %.2f"
          % mean_squared_error(y_test, X_t * b))
####################

# questao 1.2
X = dataset.values[:,1:3]
y = dataset.values[:,3]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=172, random_state=0)

print('\n')
print('peso, carga -> 1 + peso + carga + peso^2 + carga^2...')
for d in range(2,10):
    X = np.matrix(X_matrix2(X_train, d))
    X_t = np.matrix(X_matrix2(X_test, d))
    y = (np.matrix(y_train)).transpose()
    b = linalg.inv(X.transpose() * X) * X.transpose() * y
    p = np.array(b)
    nll_in = 0.5 * (y - X * b).transpose() * (y - X * b) 
    nll_out = 0.5 * ((np.matrix(y_test)).transpose() - X_t * b).transpose() * ((np.matrix(y_test)).transpose() - X_t * b) 
    print('\n')
    print('Parametros d='+ str(d) +', ', p)
    print("NLL (in sample): %.2f" % nll_in)
    print("NLL (out sample): %.2f" % nll_out)
    print("Mean squared error (in sample): %.2f"
          % mean_squared_error(y_train, X*b))
    print("Mean squared error (out sample): %.2f"
      % mean_squared_error(y_test, X_t * b))
    
####################

# questao 1.3

X = dataset.values[:,:3]
y = dataset.values[:,3]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=172, random_state=0)

print('\n')
print('idade, peso, carga -> 1 + idade + peso + carga + idade^2 + peso^2 + ...')
for d in range(2,10):
    X = np.matrix(X_matrix2(X_train, d))
    X_t = np.matrix(X_matrix2(X_test, d))
    y = (np.matrix(y_train)).transpose()
    b = linalg.inv(X.transpose() * X) * X.transpose() * y
    p = np.array(b)
    nll_in = 0.5 * (y - X * b).transpose() * (y - X * b) 
    nll_out = 0.5 * ((np.matrix(y_test)).transpose() - X_t * b).transpose() * ((np.matrix(y_test)).transpose() - X_t * b) 
    print('\n')
    print('Parametros d='+ str(d) +', ', p)
    print("NLL (in sample): %.2f" % nll_in)
    print("NLL (out sample): %.2f" % nll_out)
    print("Mean squared error (in sample): %.2f"
          % mean_squared_error(y_train, X*b))
    print("Mean squared error (out sample): %.2f"
      % mean_squared_error(y_test, X_t * b))
    
##########################

def AmericanCollegeSports(data):
    a = []
    for i in range(len(data)):
        a.append((data[i][1] * 11.4 + 260 + data[i][0] * 3.5)/ data[i][0])
    return a
    

# questao 1.4
X = dataset.values[:,1:3]
y = dataset.values[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=172, random_state=0)

print('\n')
print('peso, carga -> 1 + peso^-1 + carga*peso^-1')

X_n = []
for i in range(len(X_train)):
    a = [1.0, X_train[i][0]**(-1), X_train[i][1]*X_train[i][0]**(-1)]
    X_n.append(a)
X = X_n

X_n = []
for i in range(len(X_test)):
    a = [1.0, X_test[i][0]**(-1), X_test[i][1]*X_test[i][0]**(-1)]
    X_n.append(a)
X_t = X_n

X = np.matrix(X)
X_t = np.matrix(X_t)
y = (np.matrix(y_train)).transpose()
b = linalg.inv(X.transpose() * X) * X.transpose() * y
p = np.array(b)
nll_in = 0.5 * (y - X * b).transpose() * (y - X * b) 
nll_out = 0.5 * ((np.matrix(y_test)).transpose() - X_t * b).transpose() * ((np.matrix(y_test)).transpose() - X_t * b) 
print('\n')
print('Parametros', p)
print("NLL (in sample): %.2f" % nll_in)
print("NLL (out sample): %.2f" % nll_out)
print("Mean squared error (in sample): %.2f"
      % mean_squared_error(y_train, X*b))
print("Mean squared error (out sample): %.2f"
  % mean_squared_error(y_test, X_t * b))


print('\n')
print('Parametros American College of Sports')
print("Mean squared error (in sample): %.2f"
      % mean_squared_error(y_train, AmericanCollegeSports(X_train)))
print("Mean squared error (out sample): %.2f"
  % mean_squared_error(y_test, AmericanCollegeSports(X_test)))