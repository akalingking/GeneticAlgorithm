# The MIT License (MIT)
# 
# Copyright (c) 2016 Ariel Kalingking <akalingking@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from fann2 import libfann
from sklearn.base import BaseEstimator
import numpy as np

""" Forward symbols from the fann library """
SIGMOID                     = libfann.SIGMOID                       # 3 0,1
SIGMOID_STEPWISE            = libfann.SIGMOID_STEPWISE              # 4 0,1
SIGMOID_SYMMETRIC           = libfann.SIGMOID_SYMMETRIC             # 5 tanh -1,1
SIGMOID_SYMMETRIC_STEPWISE  = libfann.SIGMOID_SYMMETRIC_STEPWISE    # 6 -1,1 faster than 5 but less precise
TRAIN_INCREMENTAL           = libfann.TRAIN_INCREMENTAL             # 0
TRAIN_BATCH                 = libfann.TRAIN_BATCH
TRAIN_RPROP                 = libfann.TRAIN_RPROP
TRAIN_QUICKPROP             = libfann.TRAIN_QUICKPROP
TRAIN_SARPROP               = libfann.TRAIN_SARPROP
    
    
class NeuralNetwork(BaseEstimator):
    DEFAULT_NETWORK = [2,4,1]    
    def __init__(self, 
                 network=DEFAULT_NETWORK, 
                 connection_rate=1,
                 learning_rate=0.6,
                 learning_momentum=0.01,
                 desired_error=0.0001,
                 epoch=100,
                 initial_weight=1,
                 hidden_activation=SIGMOID,
                 output_activation=SIGMOID,
                 training_algorithm = TRAIN_INCREMENTAL, 
                 show=100):
        self.network = network
        self.connection_rate = connection_rate
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.desired_error = desired_error
        self.initial_weight = initial_weight
        self.epoch = epoch
        self.show = show
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.training_algorithm = training_algorithm
        self.ann = None 
        
        
    def _create_network(self):
        """ create_sparse_array Creates a standard back propagation neural network """
        self.ann = libfann.neural_net()
        
        """ Setup the network """
        self.ann.create_sparse_array(self.connection_rate, self.network)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_learning_momentum(self.learning_momentum)
        self.ann.randomize_weights(-self.initial_weight,self.initial_weight)
        self.ann.set_training_algorithm(self.training_algorithm)
        
        """ set activation function """
        self.ann.set_activation_function_hidden(self.hidden_activation)
        self.ann.set_activation_function_output(self.output_activation)
       
        """ This option is only used in cascading network """
        #ann.set_training_algorithm(libfann.TRAIN_INCREMENTAL)
        
    
    def fit(self, X, y):
        self._create_network()
        x_train = libfann.training_data()
        if len(y.shape) == 1 and y.shape[0] > 0:
            """ fann requires a row vector"""
            y = y[:, np.newaxis]
        x_train.set_train_data(X, y)
        self.ann.train_on_data(x_train, self.epoch, self.show, self.desired_error)
        #print "Training MSE error on test data: %f" % self.ann.get_MSE()
    
    
    def predict_proba(self, X):
        self.ann.reset_MSE()
        y = np.array([ self.ann.run(x) for i,x in enumerate(X) ])
        return np.c_[y, y]
    
    
    def predict(self, X):
        result = self.predict_proba(X)[:,1]
        y_pred = np.array([(1 if x > 0.5 else 0) for x in result])
        return y_pred
    
    
    def score(self, X, y, sample_weight=None):
        from sklearn import metrics
        return np.sqrt(metrics.mean_squared_error(y, self.predict(X)))
        


""" Test for the NeuralNetwork class"""
def test_neuralnetwork():
    connection_rate = 1
    learning_rate = 0.7
    learning_momentum = 0.01
    n_input = 2
    n_hidden = 4
    n_output = 1
    desired_error = 0.0001
    epoch = 1000
    show = 10
    hidden_activation = SIGMOID_SYMMETRIC
    output_activation = SIGMOID_SYMMETRIC
    training_algorithm = TRAIN_RPROP 
    network = [2,4,1]
    
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    n = NeuralNetwork(network=network, 
                      connection_rate=connection_rate,
                      learning_rate=learning_rate,
                      learning_momentum=learning_momentum,
                      desired_error=desired_error,
                      epoch=epoch,
                      hidden_activation=hidden_activation,
                      output_activation=output_activation,
                      training_algorithm=training_algorithm,
                      show=show)
    
    n.fit(X,y)
    
    print n.predict(X)


if __name__=='__main__':
    test_neuralnetwork()