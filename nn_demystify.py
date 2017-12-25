import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([3,5],[5,1],[10,2]),dtype=float)
y = np.array(([75],[82],[93]),dtype=float)

# scaling the numbers
X = X/np.amax(X,axis=0)
y = y/100 # max test score is 100

class NeuralNetwork(object):
    def __init__(self):
        # define hyperparameters of the network
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3
        
        #initialize weights
        self.W1 = np.random.randn(self.input_layer_size,self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size,self.output_layer_size)
        
    def forward(self, X):
        # propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
        
    def sigmoid(self, z):
        # apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
        
    def sigmoid_prime(self,z):
        # gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def cost_function(self, X, y):
        # compute cost for given X,y useing weights already stored in class\
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def cost_function_prime(self, X, y):
        # compute derivative with respect to W1 and W2 for a given X and y
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoid_prime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        
        return dJdW1, dJdW2
        
    # Helper functions for interacting with other classes
    
    def get_parameters(self):
        # get W1 and W2 unrolled into vector
        parameters = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return parameters
        
    def set_parameters(self, parameters):
    # set W1 and W2 using single parameter vector
        W1_start = 0
        W1_end = self.hidden_layer_size * self.input_layer_size
        self.W1 = np.reshape(parameters[W1_start:W1_end], (self.input_layer_size, self.hidden_layer_size))
        W2_end = W1_end + self.hidden_layer_size * self.output_layer_size
        self.W2 = np.reshape(parameters[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))
    
    def compute_gradients(self, X, y):
        dJdW1, dJdW2 = self.cost_function_prime(X, y)
        return np.concatenate((dJdW1.ravel(),dJdW2.ravel()))