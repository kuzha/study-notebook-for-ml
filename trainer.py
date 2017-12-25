from scipy import optimize

class trainer(object):
    def __init__(self, N):
        # make local reference to network
        self.N = N
        
    def callback_function(self, parameters):
        self.N.set_parameters(parameters)
        self.J.append(self.N.cost_function(self.X, self.y))
        
    def cost_function_wrapper(self, parameters, X, y):
        self.N.set_parameters(parameters)
        cost = self.N.cost_function(X, y)
        grad = self.N.compute_gradients(X, y)
        return cost, grad
        
    def train(self, X, y):
        # make an internal variable for the callback function
        self.X = X
        self.y = y
        
        # make an empty list to store costs
        self.J = []
        
        parameters0 = self.N.get_parameters()
        
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.cost_function_wrapper, parameters0, jac=True, \
        method='BFGS',args = (X,y), options=options, callback=self.callback_function)
        
        self.N.set_parameters(_res.x)
        self.optimization_results = _res
        