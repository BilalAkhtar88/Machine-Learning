import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Task 01, Defining Kernel Function
def kernelFun(x1, x2, kernelType, p=1, sigma=2):
    a = numpy.array(x1)
    b = numpy.array(x2)
    if kernelType == 1:
        kernelOutput = numpy.dot(a,b)
    elif kernelType == 2:
        kernelOutput = pow((numpy.dot(a,b) + 1), p)
    elif kernelType == 3:
        dist = numpy.linalg.norm(a-b)
        expPower = (pow(dist,2)) / (2*pow(sigma,2))
        kernelOutput = math.exp(-expPower)    
    return kernelOutput

# Task 02, Defining Objective Function
def objectiveFun (alphaVec):
#    alpha = numpy.array(alphaVec)
    N = len(alpha)
    sumAlpha = 0
    dualForm = 0
    while i < N:
        sumAlpha += alpha[i] 
        while j < N:
            dualForm += (alpha[i]*alpha[j]*P[i,j])
            #Where P is precomputed N x N matrix declared globally
    return 0.5*dualForm - sumAlpha
                         
# Task 03, Defining Zerofun Function
# Task 04, Calling Minimize Function
# Task 05, Extracting non-zero alpha values
# Task 06, Calculating b values
# Task 07, Implementing Indicator Function
# Task 08, Generating Test Data (Provided)
# Task 09, Plotting Results (Provided)
# Task 10, Exploring and Reporting
# ------>  SubTask 01, Effects of Moving Clusters around and changing their sizes
# ------>  SubTask 02, Effects of non-linear kernels
# ------>  SubTask 03, Effects of different parameters of non-linear kernels
# ------>  SubTask 04, Effects of slack Parameter C
# ------>  SubTask 05, Imagination :-)


    
#-----------------------------------------------------------------------------------------------------------------------------------
# The code for assignment runs above this line
# Everything below this line tests the functions used in this assignment
#-----------------------------------------------------------------------------------------------------------------------------------

# Test Bed for checking Kernel functions
arr1 = [10, 8, 6]
arr2 = [9, 6, 6]
dotProd = kernelFun(arr1,arr2,1,3)
print(dotProd)

# ret = minimize(objective, start, bounds=B, constraints=XC)
# alpha = ret['x']


# ---- TO IMPLEMENT ----

# Kernel Function
# Takes two data points as arguments and returns a "scalar product-like" similarity measure
# Start with linear kernel and explore others as well

# Objective in our case is (equation 4),
# a function that takes vector alpha as argument and returns a scalar

# start is a vector with initial guess of the alpha vector
# We can use start = numpy.zeros(N), where N is number of training samples

# B is list of pairs stating the lower and upper bounds for elements in alpha
# we can set bounds = [(0,C) for b in range(N)]
# to only have lower bound use bounds = [(0, None) for b in range(N)]
