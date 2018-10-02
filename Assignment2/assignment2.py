import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Task 01, Defining Kernel Function
def kernelFun(x1, x2, kernelType, p=1, sigma=2):
    #   a = numpy.array(x1)
    #   b = numpy.array(x2)
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
def objectiveFun (alpha):
#    alpha = numpy.array(alphaVec)
#    N = len(alpha)
    global N
    sumAlpha = 0
    dualForm = 0
    i = 0
    j = 0
    while i < N:
        sumAlpha += alpha[i] 
        while j < N:
            dualForm += (alpha[i]*alpha[j]*P[i,j])
            j +=1
            #Where P is precomputed N x N matrix declared globally
        i += 1
    return 0.5*dualForm - sumAlpha
                         
# Task 03, Defining Zerofun Function
# Task 04, Calling Minimize Function
# Task 05, Extracting non-zero alpha values
# Task 06, Calculating b values
# Task 07, Implementing Indicator Function
def indicator(a,b):
    indicat = 0
    global N
    i = 0
    while i < N:
        indicat += (alpha[i]*targets[i]*kernelFun([a,b],inputs[i],1))
        i += 1
    return indicat

# Task 08, Generating Test Data (Provided)

numpy.random.seed(100)  #For getting same random numbers everytime

classA = numpy.concatenate (
(numpy.random.randn(10,2)*0.2 + [1.5,0.5] ,
numpy.random.randn (10,2)*0.2 + [-1.5,0.5]))

classB = numpy.random.randn (20,2)*0.2 + [ 0.0 , -0.5]

inputs = numpy.concatenate((classA ,classB))
targets = numpy.concatenate (
(numpy.ones(classA.shape[ 0 ]),
-numpy.ones(classB.shape[0])))
N = inputs.shape[0] # Number of rows ( samples )
alpha = numpy.zeros(N)

permute = list(range(N))
random.shuffle(permute)
inputs= inputs[permute,:]
targets = targets[permute]

# Task 09, Plotting Results (Provided)

plt.plot([p[0] for p in classA] ,
[p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB] ,
[p[1] for p in classB], 'r.')
plt.axis('equal') # Force same s c a l e on both axes
#plt.savefig('svmplot.pdf') # Save a copy in a f i l e
plt.show() # Show t h e p l o t on t h e s c r e e n

xgrid = numpy.linspace(-5,5)
ygrid = numpy.linspace(-4,4)

grid = numpy.array([[indicator(x,y)
for x in xgrid]
for y in ygrid])
plt.contour (xgrid,ygrid,grid,
(-1.0,0.0,1.0),
colors =('red','black','blue'),
linewidths =(1,3,1))

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
