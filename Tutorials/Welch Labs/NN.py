#%% Imports
import numpy as np



#%% Data

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Scaling data
X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100



#%% Neural Network Class

class Neural_Network(object):
    def __init__(self):        
        # Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        # Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2

    #Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        # Converts 1D params into W1 and W2 matrices.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        # Get dJdW1, dJdW2 unrolled into vector
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))



#%%% Wrapper to train using scipy's BFGS

from scipy import optimize

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
        
    def callbackF(self, params):
        self.N.setParams(params)
        
        current_cost = self.N.costFunction(self.X, self.y) 
        self.J.append(current_cost)   
        print(current_cost)
        
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)                # sets W1 and W2
        cost = self.N.costFunction(X, y)        # returns scalar Cost
        grad = self.N.computeGradients(X, y)    # 1D array of gradients
        return cost, grad
        
    
    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)
        #   minimize() parameters:
        #       fun     : Objective function - takes params
        #       x0      : Initial Guess
        #       jac     : If jac is a Boolean and is True, fun is assumed to return 
        #                  the gradient along with the objective function.
        #       args    : Extra arguments passed to the objective function and its 
        #                  derivatives (Jacobian, Hessian)
        #       options : maxiter : Maximum number of iterations to perform
        #      callback : Called after each iteration, as callback(xk), 
        #                   where xk is the current parameter vector 
        
        #   minimize() returns
        #       x : solution array
        
        self.N.setParams(_res.x)
        self.optimizationResults = _res



#%% Training using Gradient Descent

scalar = 1
NN = Neural_Network()

for i in range(1,10):
    cost = NN.costFunction(X,y)
    dJdW1, dJdW2 = NN.costFunctionPrime(X,y)
    # print(cost, dJdW1, dJdW2)
    print(cost)
    
    NN.W1 = NN.W1 - scalar*dJdW1
    NN.W2 = NN.W2 - scalar*dJdW2



#%% Training using BFGS

NN = Neural_Network()
T = trainer(NN)
T.train(X,y)