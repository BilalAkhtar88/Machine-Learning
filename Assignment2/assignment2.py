import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Task 08, Generating Test Data (Provided)
numpy.random.seed(100)  #For getting same random numbers everytime

classA = numpy.concatenate ((numpy.random.randn(10,2)*0.2 + [1.5,0.5] , numpy.random.randn (10,2)*0.2 + [-1.5,0.5]))
classB = numpy.random.randn (20,2)*0.2 + [ 0.0 , -0.5]

inputs = numpy.concatenate((classA ,classB))
targets = numpy.concatenate ((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows ( samples )
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]

# Task 01, Defining Kernel Function
def kernelFun(x1, x2):
    kernelOutput = numpy.dot(x1, x2)

#    kernelOutput = pow((numpy.dot(x1,x2) + 1), 4)

#    dist = numpy.linalg.norm(x1-x2)
#    expPower = (pow(dist,2)) / (2*pow(0.5,2))
#    kernelOutput = math.exp(-expPower)

    return kernelOutput

# Task 02, Defining Objective Function
def buildTable(inputs, targets):
    global lookUpTable
    for i in range(N):
        for j in range(i+1):
            term = targets[i]*targets[j]*kernelFun(inputs[i],inputs[j])
            lookUpTable[i][j] = lookUpTable[j][i] = term

def objectiveFun(alphas):
    result = numpy.sum(lookUpTable*alphas*alphas.reshape((N,1)))/2 - numpy.sum(alphas)
    return result

# Task 03, Defining Zerofun Function
def zerofun(alphas):
    return(numpy.dot(alphas,targets))

# Task 04, Calling Minimize Function
C = None
lookUpTable = numpy.zeros((N,N))
buildTable(inputs, targets)
XC = {'type': 'eq', 'fun': zerofun}
B = [(0,C) for b in range(N)]
ret = minimize(objectiveFun, numpy.zeros(N), bounds=B , constraints = XC)
alpha = ret['x']

# Task 05, Extracting non-zero alpha values
threshold = 1e-5
index = numpy.where(abs(alpha)>abs(threshold))[0]

# Task 06, Calculating b values
def biasCalc(alpha,svIndex):
    biasC = numpy.sum([alpha[i] * targets[i] * kernelFun(inputs[i], inputs[svIndex]) for i in index]) - targets[svIndex]
    return biasC

bias = biasCalc(alpha,index[0])

# Task 07, Implementing Indicator Function
def indicator(a,b, alpha, index, bias):
    test = [a,b]
    result = numpy.sum([alpha[i]*targets[i]*kernelFun(inputs[i],test) for i in index]) - bias
    return result


# Task 09, Plotting Results (Provided)

plt.plot([p[0] for p in classA] , [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB] , [p[1] for p in classB], 'r.')

xgrid = numpy.linspace(-5,5)
ygrid = numpy.linspace(-4,4)
grid = numpy.array([[indicator(x,y,alpha,index,bias) for x in xgrid] for y in ygrid])

plt.contour (xgrid,ygrid,grid, (-1.0,0.0,1.0), colors =('red','black','blue'), linewidths =(1,3,1))
plt.axis('equal')
plt.show()
#plt.savefig('svmplot.pdf') # Save a copy in a f i l e

# Task 10, Exploring and Reporting
# ------>  SubTask 01, Effects of Moving Clusters around and changing their sizes
# ------>  SubTask 02, Effects of non-linear kernels
# ------>  SubTask 03, Effects of different parameters of non-linear kernels
# ------>  SubTask 04, Effects of slack Parameter C
# ------>  SubTask 05, Imagination :-)