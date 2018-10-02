
# coding: utf-8

# # Lab 02 - SVM

# In[1]:


import numpy as np
import random, math
from scipy .optimize import minimize
import matplotlib.pyplot as plt 


# ## Data Generation

# In[122]:


classA = np.concatenate(           (np.random.randn(10,2)*0.2 + [1.5, 0.5],            np.random.randn(10,2)*0.2 + [-1.5,0.5]))
classB = np.random.randn(20,2)*0.2 + [0,-0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]

# shuffle data
permute = list(range(N))
random.shuffle(permute)

inputs = inputs[permute, :]
targets = targets[permute]


# ## Implementation

# In[24]:


# lookUpTable = np.zeros((N,N))


# In[123]:


def kernelFun(x,y):
    return np.dot(x,y) # linear kernel functionmyGlobal = 5
#     return (np.dot(x,y)+1)**2 # polynomial kernel p = 2
#     return (np.dot(x,y)+1)**3 # polynomial kernel p = 3
#     return np.exp(-(np.linalg.norm(x-y))**2 / (2*0.5)) # RBF kernel with variance = 0.5


# In[26]:


def buildTable(inputs, targets):
    global N
    global lookUpTable
    n = len(inputs)
    for i in range(N):
        for j in range(i+1):
            term = targets[i]*targets[j]*kernelFun(inputs[i],inputs[j])
            lookUpTable[i][j] = lookUpTable[j][i] = term


# In[27]:


def objective(alphas):
    global N
    global loopUpTable
    result = np.sum(lookUpTable*alphas*alphas.reshape((N,1)))/2 - np.sum(alphas)
    return result


# In[28]:


def zerofun(alphas):# input might need to be global
    global targets
    return np.dot(alphas,targets)


# In[29]:


def indicate(test, alpha, index, bias):
    result = np.sum([alpha[i]*targets[i]*kernelFun(inputs[i],test) for i in index]) - bias
    return result


# In[30]:


def indicator(x, y, alpha, index, bias):
    test = [x,y]
    result = np.sum([alpha[i]*targets[i]*kernelFun(inputs[i],test) for i in index]) - bias
    return result


# In[139]:


# find alphas
C = None
lookUpTable = np.zeros((N,N))
buildTable(inputs, targets)
bounds = [(0,C) for i in range(N)]
constraint = {'type':'eq', 'fun':zerofun};
ret = minimize(objective, np.zeros(N), bounds=bounds, constraints=constraint)
alpha = ret['x']
print(ret['success'])


# In[140]:


# extract non-zero alphas
threshold = 1e-5
index = np.where(abs(alpha)>abs(threshold))[0]

k = index[0] # index of first support vector
bias = np.sum([alpha[i]*targets[i]*kernelFun(inputs[i],inputs[k]) for i in index]) - targets[k]
print(index)
# validate bias
# k = 35
# print(np.sum([alpha[i]*targets[i]*kernelFun(inputs[i],inputs[k]) for i in index]) - bias)
# print(indicate(inputs[k], alpha, index, bias))
# print(targets[k])


# ## 6. Plotting

# In[141]:


plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

xgrid = np.linspace(-4, 4)
ygrid = np.linspace(-2, 2)
grid = np.array([[indicator(x,y, alpha, index, bias) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid, (-1, 0, 1), colors=('red','black','blue'), linewidths=(1,3,1))
plt.axis('equal')
plt.show()

# test on classA
# result = [indicator(p[0],p[1],alpha,index,bias) for p in classA]
# print(np.sign(result))
# print(result)


# In[135]:


# np.dot(alpha[index],targets[index])
# print(alpha[index], targets[index])
# np.mean([alpha[i]*targets[i]*inputs[i,:] for i in index], axis=0) # normal vector of decision boundary


# In[136]:


a = 5e-5
a*10


# In[137]:


alpha[index]


# In[138]:



delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2
print(Z)

