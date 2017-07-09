import matplotlib.pyplot as plt
import numpy as np
import os
import theano
import time

from theano import tensor as T

from utils import arbitrary_polynomial


class FeedForwardNetwork:
    """Feedforward neural network"""

    def __init__(self, n_hidden_units=100):
        x = T.dvector('x')
        y = T.dvector('y')
        self.method = 'GD'

        # initialize W and b parameters
        np.random.seed(1234)
        norm = np.sqrt(1./n_hidden_units)
        self.W1 = theano.shared(norm * np.random.randn(n_hidden_units),
                                name='W1')
        self.b1 = theano.shared(norm * np.random.randn(n_hidden_units, 1),
                                name='b1', broadcastable=(False, True))
        self.W2 = theano.shared(norm * np.random.randn(n_hidden_units),
                                name='W2')
        self.b2 = theano.shared(norm * np.random.randn(1),
                                name='b2')
        self.params = [self.W1, self.b1, self.W2, self.b2]

        # Define single hidden layer model with tanh activation
        hidden = T.tanh(T.outer(self.W1, x) + self.b1)
        y_pred = T.tanh(T.dot(self.W2, hidden) + self.b2.dimshuffle(0, 'x'))
        cost = T.sum(T.pow(y-y_pred, 2)) / (2 * x.shape[0])

        # Alternative model with ReLU activation
        # hidden = T.nnet.relu(T.outer(self.W1,x) + self.b1, alpha=0.2)
        # y_pred = T.nnet.relu(T.dot(self.W2, hidden)
        #          + self.b2.dimshuffle(0, 'x'), alpha=0.2)

        # Define predictions
        self.predict = theano.function(inputs=[x], outputs=y_pred)

        # Define training
        alpha = T.dscalar('alpha')  # learning rate
        gradient = T.grad(cost, [self.W1, self.b1, self.W2, self.b2])

        updates = ([self.W1, self.W1 - alpha*gradient[0]],
                   [self.b1, self.b1 - alpha*gradient[1]],
                   [self.W2, self.W2 - alpha*gradient[2]],
                   [self.b2, self.b2 - alpha*gradient[3]])

        self.update_and_report_cost = theano.function(inputs=[x, y, alpha],
                                                      outputs=cost,
                                                      updates=updates)

        # Newton's method
        # elif method == 'NM':
        #    gradient = T.grad(cost, all_params)
        #    #print(gradient.eval({x: x_values[0], y: data[0]}).shape)
        #    hessian  = T.hessian(cost, all_params, consider_constant=[x,y])
        #    #print(hessian.eval({x: x_values[0], y: data[0]}).shape)
        #    #print(np.linalg.inv(hessian.eval({x: x_values[0], y: data[0]})
        # ).shape)
        #    newton_update_matrix = T.dot(T.nlinalg.matrix_inverse(hessian),
        # gradient)
        #    #newton_update_matrix = T.nlinalg.matrix_inverse(hessian)
        #    #print(newton_update_matrix.eval({x: x_values[0], y: data[0]}
        # ).shape)
        #    updates = [(all_params, all_params - alpha*newton_update_matrix)]

    def learn(self, x_values, y_values, n_iters=10e7, learn_rate=0.001,
              save_to_path='./'):
        
        # keep track of weights as learning goes on 
        self.W1_array = []
        self.b1_array = []
        self.W2_array = []
        self.b2_array = []
        
        self.cost_array = []
        self.iters_array = []
        self.learn_rate = learn_rate
        self.n_iters = n_iters

        cost = 0.0

        # learn_rate_start = 3.0
        # learn_saturate = 0.5 * n_iters

        print_freq = min(10e6, int(n_iters/10.))
        start = time.time()
        print("Started.")
        for i in np.arange(n_iters, dtype=np.float32):

            # Optionally vary the learning rate for some iterations
            # if i < learn_saturate:
            #    learn_rate += learn_rate_start * (1 - i/learn_saturate)

            cost += self.update_and_report_cost(x_values, y_values, learn_rate)

            if (i % print_freq == 0 and i > 0) or i == n_iters-1:
                print("At iteration %d, cost = %.6f, Total run time = %.2f mins"
                      % (i, cost/print_freq, (time.time()-start) / 60.))

                self.cost_array.append(cost/print_freq)
                self.iters_array.append(i)
                self.W1_array.append(self.W1.get_value())
                self.b1_array.append(self.b1.get_value())
                self.W2_array.append(self.W2.get_value())
                self.b2_array.append(self.b2.get_value())
                cost = 0

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
                           'b1': self.b1.get_value(),
                           'W2': self.W2.get_value(),
                           'b2': self.b2.get_value(),
                           'W1_array': self.W1_array,
                           'b1_array': self.b1_array,
                           'W2_array': self.W2_array,
                           'b2_array': self.b2_array}

        for var_name, value in model_variables.items():
            filename = self._name_parameter_file(var_name,
                                                 optim_method=self.method,
                                                 learn_rate=self.learn_rate,
                                                 iters_millions=iters_millions)
            fullpath = os.path.join(path, filename)
            np.save(fullpath, value)

    def restore_parameters_and_cost(self, learn_rate, iters_millions,
                                    method='GD', path='./'):
        '''Set cost, iterations and W, b parameters for files.'''
        fullpath = {var: os.path.join(path, self._name_parameter_file(
            parameter=var,
            optim_method=method,
            learn_rate=learn_rate,
            iters_millions=iters_millions)
        ) for var in ('cost', 'iters', 'W1', 'b1', 'W2', 'b2')}

        self.cost_array = np.load(fullpath['cost'])
        self.iters_array = np.load(fullpath['iters'])
        self.W1.set_value(np.load(fullpath['W1']))
        self.b1.set_value(np.load(fullpath['b1']))
        self.W2.set_value(np.load(fullpath['W2']))
        self.b2.set_value(np.load(fullpath['b2']))

    def plot_cost(self, figsize=(14, 4)):
        plt.figure(figsize=figsize)
        plt.plot(self.iters_array, self.cost_array)
        plt.title('Cost function vs iteration step')
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.show()

    def plot_prediction(self, x_values, y_values, params_chosen,
                        figsize=(14, 4)):
        predictions = self.predict(x_values).reshape(len(x_values),)

        plt.figure(figsize=figsize)
        plt.plot(x_values, predictions, color='darkblue',
                 label='Model prediction')
        plt.plot(x_values, arbitrary_polynomial(x_values, params_chosen), '-',
                 color='darkgreen', label='Underlying distribution')
        plt.plot(x_values, y_values, '.', color='darkorange',
                 label='Training data')
        plt.title('Model prediction compared to true underlying distribution')
        plt.legend()
        plt.show()
