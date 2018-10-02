import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Task 01, Defining Kernel Function
def kernelFun(x1, x2):
    kernelOutput = numpy.dot(x1, x2)
    return kernelOutput

#    if kernelType == 1:
#    elif kernelType == 2:
#        kernelOutput = pow((numpy.dot(a,b) + 1), p)
#    elif kernelType == 3:
#        dist = numpy.linalg.norm(a-b)
#        expPower = (pow(dist,2)) / (2*pow(sigma,2))
#        kernelOutput = math.exp(-expPower)

# Task 02, Defining Objective Function
def buildTable(inputs, targets):
    global N
    global lookUpTable
    n = len(inputs)
    for i in range(N):
#        print(i)
        for j in range(i+1):
#            print(j)
            term = targets[i]*targets[j]*kernelFun(inputs[i],inputs[j])
            lookUpTable[i][j] = lookUpTable[j][i] = term

def objectiveFun(alphas):
    global N
    global lookUpTable
    result = numpy.sum(lookUpTable*alphas*alphas.reshape((N,1)))/2 - numpy.sum(alphas)
    return result


# Task 03, Defining Zerofun Function
def zerofun(alpha):
    return(numpy.dot(alpha,targets))

# Task 04, Calling Minimize Function


# Task 05, Extracting non-zero alpha values

# Task 06, Calculating b values
def biasCalc(alpha,svIndex):
    biasC = numpy.sum([alpha[i] * targets[i] * kernelFun(inputs[i], inputs[svIndex]) for i in index]) - targets[svIndex]
    return biasC

# Task 07, Implementing Indicator Function
def indicator(a,b):
    indicat = 0
    global N
    i = 0
    while i < N:
        indicat += ((alpha[i]*targets[i]*kernelFun([a,b],inputs[i])) - bias)
        i += 1
#        print(i)
    return indicat

# Task 08, Generating Test Data (Provided)

numpy.random.seed(100)  #For getting same random numbers everytime

classA = numpy.concatenate (
(numpy.random.randn(10,2)*0.2 + [1.5,0.5] ,
numpy.random.randn (10,2)*0.2 + [-1.5,0.5]))

classB = numpy.random.randn (20,2)*0.2 + [ 0.0 , -0.5]

inputs = numpy.concatenate((classA ,classB))
targets = numpy.concatenate (
(numpy.ones(classA.shape[0]),
-numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows ( samples )
#start = numpy.zeros(N)
#typeOfKernel = 1
C = None
lookUpTable = numpy.zeros((N,N))

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]
buildTable(inputs, targets)

XC = {'type': 'eq', 'fun': zerofun}
B = [(0,C) for b in range(N)]
ret = minimize(objectiveFun, numpy.zeros(N), bounds=B , constraints = XC)
alpha = ret['x']
#print(alpha)
threshold = 1e-5
index = numpy.where(abs(alpha)>abs(threshold))[0]
bias = biasCalc(alpha,index[0])
print(bias)

# Task 09, Plotting Results (Provided)

plt.plot([p[0] for p in classA] , [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB] , [p[1] for p in classB], 'r.')
#plt.axis('equal') # Force same s c a l e on both axes
#plt.savefig('svmplot.pdf') # Save a copy in a f i l e
#plt.show() # Show t h e p l o t on t h e s c r e e n

xgrid = numpy.linspace(-5,5)
ygrid = numpy.linspace(-4,4)
grid = numpy.array([[indicator(x,y) for x in xgrid] for y in ygrid])

plt.contour (xgrid,ygrid,grid, (-1.0,0.0,1.0), colors =('red','black','blue'), linewidths =(1,3,1))
plt.axis('equal')
plt.show()

# Task 10, Exploring and Reporting
# ------>  SubTask 01, Effects of Moving Clusters around and changing their sizes
# ------>  SubTask 02, Effects of non-linear kernels
# ------>  SubTask 03, Effects of different parameters of non-linear kernels
# ------>  SubTask 04, Effects of slack Parameter C
# ------>  SubTask 05, Imagination :-)