import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import theano
from theano import tensor as T
import time


params_chosen = [1.0 , 0, -1.0, 5.0]
params_chosen_tanh = [10.0, 0, -1.0, 5.0]

data_long = np.asarray(
    [-5.06279094, -3.07913632, -3.65501026, -2.67704646, -2.49804696,
     -1.05224107, -2.24888228, -1.24914231, -0.96866704,  0.10196036,
     -0.44501532,  0.54137897,  0.83736665, -0.0091876 ,  0.80395852,
      0.89656807,  0.3034906 ,  0.81762378,  2.1443069 ,  0.86089562,
      0.44570157,  1.2116284 ,  0.23296191,  0.87709155,  0.72194836,
      1.60927391,  1.87818843,  1.74450324,  1.42328076,  0.67616366,
      2.12236596,  1.83113031,  1.34116073,  1.52807377,  2.08071048,
      2.89823459,  3.13168034,  2.25879038,  3.28892569,  3.93525823])

data_short = np.asarray(
    [-5.27962646, -3.47487732, -2.62077997, -1.11694373, -0.51112509,
     1.20384878,  1.20820834, -0.47329551,  1.09841872,  1.75225833,
     0.66405822,  2.48762016,  0.94590405,  0.72055498,  1.78287262,
     1.16909767,  1.60426967,  3.04844251,  2.21983749,  3.83828095])

data_tanh = np.asarray(
    [0.19200746,  0.26390769,  0.27416557,  0.29274981,  0.34720758,
     0.33646458,  0.3954851 ,  0.38163466,  0.4721299 ,  0.4648484 ,
     0.45125321,  0.49003209,  0.4403167 ,  0.46711053,  0.48810206,
     0.459813  ,  0.46432161,  0.52867192,  0.55053968,  0.50073061,
     0.50371526,  0.50858915,  0.49070523,  0.5121518 ,  0.52849574,
     0.51643671,  0.44317211,  0.51967805,  0.49736733,  0.49924654,
     0.48524065,  0.50282076,  0.56767256,  0.50839501,  0.57744206,
     0.58775349,  0.62222579,  0.57015087,  0.63695049,  0.64991875])


def arbitrary_polynomial(x, params):
    return 0.05 * sum([param * (x**i)
                       for i, param in enumerate(params)])

class FeedForwardNetwork:
    """Feedforward neural network"""

    def __init__(self, n_hidden_units=100, seed = 1234, rand = True, W1_in=0.01, W2_in=0.01):
        x = T.dvector('x')
        y = T.dvector('y')
        self.method = 'GD'

        # initialize W and b parameters
        if rand:
            np.random.seed(seed)
            norm = np.sqrt(1./n_hidden_units)
            self.W1 = theano.shared(norm * np.random.randn(n_hidden_units),
                                name='W1')
            # self.b1 = theano.shared(norm * np.random.randn(n_hidden_units, 1),
                                # name='b1', broadcastable=(False, True))
            self.W2 = theano.shared(norm * np.random.randn(n_hidden_units),
                                name='W2')
            # self.b2 = theano.shared(norm * np.random.randn(1),
                                # name='b2')
        
        else:
            self.W1 = theano.shared(W1_in, name = 'W1')
            self.W2 = theano.shared(W2_in, name = 'W2')
        
        self.params = [self.W1, self.W2]

        # Define single hidden layer model with tanh activation
        hidden = T.tanh(T.outer(self.W1, x) )
        y_pred = T.tanh(T.dot(self.W2, hidden) )
        cost = T.sum(T.pow(y-y_pred, 2)) / (2 * x.shape[0])
       

        ## Alternative model with ReLU activation
        # hidden = T.nnet.relu(T.outer(self.W1,x) + self.b1, alpha=0.2)
        # y_pred = T.nnet.relu(T.dot(self.W2, hidden)
        #          + self.b2.dimshuffle(0, 'x'), alpha=0.2)

        # Define predictions
        self.predict = theano.function(inputs=[x], outputs=y_pred)

        # Define training
        alpha = T.dscalar('alpha') # learning rate
        gradient = T.grad(cost, [self.W1,
                                 self.W2])

        updates = ([self.W1, self.W1 - alpha*gradient[0]],
                   [self.W2, self.W2 - alpha*gradient[1]])

        #self.update_and_report_cost = theano.function(inputs=[x, y, alpha],
        #                                              outputs=cost,
        #                                              updates=updates)

        self.update_and_report_cost = theano.function(inputs=[x, y, alpha],
                                                      outputs=[cost, self.W1, self.W2],
                                                      updates=updates)

        ## Newton's method
        # elif method == 'NM':
        #    gradient = T.grad(cost, all_params)
        #    #print(gradient.eval({x: x_values[0], y: data[0]}).shape)
        #    hessian  = T.hessian(cost, all_params, consider_constant=[x,y])
        #    #print(hessian.eval({x: x_values[0], y: data[0]}).shape)
        #    #print(np.linalg.inv(hessian.eval({x: x_values[0], y: data[0]})).shape)
        #    newton_update_matrix = T.dot(T.nlinalg.matrix_inverse(hessian), gradient)
        #    #newton_update_matrix = T.nlinalg.matrix_inverse(hessian)
        #    #print(newton_update_matrix.eval({x: x_values[0], y: data[0]}).shape)
        #    updates = [(all_params, all_params - alpha*newton_update_matrix)]


    def learn(self, x_values, y_values, n_iters=10e7, learn_rate=0.001,
              save_to_path='./'):

        # keep track of weights as learning goes on
        self.W1_array = [self.W1.get_value()]
        # self.b1_array = []
        self.W2_array = [self.W2.get_value()]
        # self.b2_array = []

        self.cost_array = [self.update_and_report_cost(x_values, y_values, learn_rate)[0]]
        self.iters_array = []
        self.learn_rate = learn_rate
        self.n_iters = n_iters

        # cost = 0.0

        # learn_rate_start = 3.0
        # learn_saturate = 0.5 * n_iters

        print_freq = min(10e6, int(n_iters/10.))
        start = time.time()
        print("Started.")
        for i in np.arange(n_iters, dtype=np.float32):

            # Optionally vary the learning rate for some iterations
            # if i < learn_saturate:
            #    learn_rate += learn_rate_start * (1 - i/learn_saturate)

            cost = self.update_and_report_cost(x_values, y_values, learn_rate)[0]
            W1 = self.update_and_report_cost(x_values, y_values, learn_rate)[1]
            # b1 = self.update_and_report_cost(x_values, y_values, learn_rate)[2]
            W2 = self.update_and_report_cost(x_values, y_values, learn_rate)[2]
            # b2 = self.update_and_report_cost(x_values, y_values, learn_rate)[4]

            if (i % print_freq == 0 and i > 0) or i == n_iters-1:
                print("At iteration %d, cost = %.6f, Total run time = %.2f mins"
                      % (i, cost, (time.time()-start) / 60.))

                self.cost_array.append(cost)
                self.iters_array.append(i)
                self.W1_array.append(W1)
                # self.b1_array.append(b1)
                self.W2_array.append(W2)
                # self.b2_array.append(b2)
                # cost = 0

        if save_to_path:
            print("Saving...")
            self.save_parameters_and_cost(save_to_path)
            print("Saved.")


    def _name_parameter_file(self, parameter, optim_method, learn_rate,
                             iters_millions):
        filename = '_'.join([optim_method,
                             'lr{}'.format(learn_rate),
                             'iter{}M'.format(iters_millions),
                             '{}.npy'.format(parameter),
                             ])
        return filename

    def save_parameters_and_cost(self, path='./'):
        '''Save cost, iterations and W, b parameters as .npy files.'''
        if not os.path.exists(path):
            os.makedirs(path)

        iters_millions = self.n_iters/1e6
        model_variables = {'cost': self.cost_array,
                           'iters': self.iters_array,
                           'W1': self.W1.get_value(),
                        #    'b1': self.b1.get_value(),
                           'W2': self.W2.get_value(),
                        #    'b2': self.b2.get_value(),
                           'W1_array': self.W1_array,
                        #    'b1_array': self.b1_array,
                           'W2_array': self.W2_array
                        #    'b2_array': self.b2_array
                        }

        for var_name, value in model_variables.items():
            filename = self._name_parameter_file(var_name,
                                                 optim_method=self.method,
                                                 learn_rate=self.learn_rate,
                                                 iters_millions=iters_millions)
            fullpath = os.path.join(path, filename)
            np.save(fullpath, value)


    def restore_parameters_and_cost(self, learn_rate, iters_millions, method='GD', path='./'):
        '''Set cost, iterations and W, b parameters for files.'''
        fullpath = {var: os.path.join(path,
                                      self._name_parameter_file(parameter=var,
                                                                optim_method=method,
                                                                learn_rate=learn_rate,
                                                                iters_millions=iters_millions))
                    for var in ('cost', 'iters', 'W1', 'W2',
                                'W1_array',
                                'W2_array')}

        self.cost_array = np.load(fullpath['cost'])
        self.iters_array = np.load(fullpath['iters'])
        self.W1.set_value(np.load(fullpath['W1']))
        # self.b1.set_value(np.load(fullpath['b1']))
        self.W2.set_value(np.load(fullpath['W2']))
        # self.b2.set_value(np.load(fullpath['b2']))


    def plot_cost(self, figsize=(14,4)):
        plt.figure(figsize=figsize)
        plt.plot(self.iters_array, self.cost_array)
        plt.title('Cost function vs iteration step')
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()


    def plot_prediction(self, x_values, y_values, params_chosen, figsize=(14,4)):
        predictions = self.predict(x_values).reshape(len(x_values),)

        plt.figure(figsize=figsize)
        plt.plot(x_values, predictions, color='darkblue', label='Model prediction')
        plt.plot(x_values, arbitrary_polynomial(x_values, params_chosen), '-',
                 color='darkgreen', label='Underlying distribution')
        plt.plot(x_values, y_values,'.', color='darkorange', label='Training data')
        plt.title('Model prediction compared to true underlying distribution')
        plt.legend()
        plt.show()
