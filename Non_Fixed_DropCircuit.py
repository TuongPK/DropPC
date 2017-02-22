import theano
import theano.tensor as T
import numpy
import timeit
import os
import sys

from random import shuffle

# MNIST does not have normalization, other datasets have
from mnist_sgd import LogisticRegression, mnist_load_data
from logistic_sgd import load_data


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, drop, train, probability, W=None, b=None, activation=T.tanh):    
        self.input = input
        
        # If undefined then initialize
        if W is None:
            W_values = numpy.asarray(
                numpy.random.uniform(
                    low = -numpy.sqrt(6./ (n_in + n_out)),
                    high = numpy.sqrt(6./ (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            
            W = theano.shared(value = W_values, name='W', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b
        
        # Apply activation function on input sum
        lin_output = T.dot(input, self.W) + self.b
        output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        # Drop if being trained
        self.output = T.switch(T.neq(train,0), 
                               output * T.cast(drop, theano.config.floatX), # drop according to switch
                               (output * (1 - probability))
                              )
       
        self.params = [self.W, self.b]
        
class PC(object):
    def __init__(self, input, n_in, n_hidden_layer, n_hidden_node, train, probability, drop, sparsity):
                
        # Initialize hidden layers in circuit
        self.hiddenLayers = []
        for index in range(n_hidden_layer):
            self.hiddenLayers.append(
                HiddenLayer(
                    input = input if index == 0 else self.hiddenLayers[index - 1].output,
                    n_in = n_in if index == 0 else n_hidden_node[index - 1],
                    n_out = n_hidden_node[index],
                    train = train,
                    probability = probability, drop = drop,
                )
            )
        
        # Calculate sparsity penalty
        self.L1 = 0
        self.L2_sqr = 0
        
        self.params = []
        
        for layer in self.hiddenLayers:
            self.params += layer.params
        
        self.output = self.hiddenLayers[n_hidden_layer - 1].output

        # Collect weight and bias matrix
        if sparsity == 1: # full network sparsity
            for layer in self.hiddenLayers:
                self.L1 += abs(layer.output).sum()
                self.L2_sqr += (layer.output ** 2).sum()
        elif sparsity == 2: # penultimate sparsity
            self.L1 += abs(self.output).sum()
            self.L2_sqr += (self.output ** 2).sum()
        else: # no sparsity
            self.L1 = 0
            self.L2_sqr = 0
            # Do nothing
        
        self.input = input

class MLP(object):
    def __init__(self, input, n_in, n_hidden_layer, n_hidden_node, n_circuit, n_out, train, probability, mask, sparsity):
        
        # Sparsity coefficients 
        L1_reg = numpy.random.uniform(low=0, high=0.002, size=(n_circuit,)) * 0
        L2_reg = numpy.random.uniform(low=0, high=0.0002, size=(n_circuit,)) 
        
        # Initilize circuits in network
        self.circuits = []
        
        for index in xrange(n_circuit):
            self.circuits.append(
                PC(
                    input = input,
                    n_hidden_layer = n_hidden_layer,
                    n_hidden_node = [n / n_circuit for n in n_hidden_node],
                    n_in = n_in, # sharing same input
                    train = train,
                    drop = mask[index],
                    probability = probability,
                    sparsity = sparsity
                )
            )
            
        # Shared output
        penultimateLayer = []
        
        for circuit in self.circuits:
            penultimateLayer.append(circuit.output)
        
        self.logRegressionLayer = LogisticRegression(
            input = T.concatenate(penultimateLayer, axis=1),
            n_in = n_hidden_node[n_hidden_layer - 1],
            n_out = n_out
        )
        
        # Sum the total sparsity penalty of all hidden layers
        self.L1 = 0
        self.L2_sqr = 0
        
        for index in xrange(n_circuit):
            self.L1 += self.circuits[index].L1 * L1_reg[index]
            self.L2_sqr += self.circuits[index].L2_sqr * L2_reg[index]

        
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        
        self.errors = self.logRegressionLayer.errors
        
        self.params = []
        
        for circuit in self.circuits:
            self.params += circuit.params
        
        self.params += self.logRegressionLayer.params
        
        self.input = input
            
def run_NFD(dataset,
            n_hidden_node, n_circuit,
            learning_rate, n_epochs, momentum, batch_size,
            probability,
            sparsity
           ):
    
    # Load and divide datasets
    # MNIST's training and testing set is predifined. Others are at a whole
    if dataset == 'mnist.pkl.gz':
        datasets = mnist_load_data(dataset)
    else:
        datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    train = T.iscalar() # Dropout switch
    
    # Create PC drop mask
    srng = T.shared_randomstreams.RandomStreams()
    mask = srng.binomial(n=1, p=1 - probability, size=(n_circuit,))
    
    # Initilize network
    classifier = MLP(
        input = x,
        n_in = 784,
        n_hidden_layer = len(n_hidden_node),
        n_hidden_node = n_hidden_node,
        n_circuit = n_circuit,
        n_out = 10,
        train = train,
        mask = mask,
        probability = probability,
        sparsity = sparsity
    )
    
    # Apply sparsity penalty to error cost
    cost = (
        classifier.negative_log_likelihood(y)
        + classifier.L1
        + classifier.L2_sqr
    )
    
    # Test function returning error rate of batch 'index'
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x : test_set_x[index * batch_size: (index + 1) * batch_size],
            y : test_set_y[index * batch_size: (index + 1) * batch_size],
            train : numpy.cast['int32'](0)
        }
    )
    
    # Validate function returning error rate of batch 'index'
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x : valid_set_x[index * batch_size: (index + 1) * batch_size],
            y : valid_set_y[index * batch_size: (index + 1) * batch_size],
            train : numpy.cast['int32'](0)
        }
    )
    
    # Compute gradient for all weight & bias matrices
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    # Weight & bias update
    updates = []
    for param, gparam in zip(classifier.params, gparams):
        param_update = theano.shared(param.get_value() * 0.) # nullify
    
        updates.append((param, param - learning_rate * param_update)) # main update
        updates.append((param_update, momentum * (param_update) + (1 - momentum) * gparam)) # include momentum
    
    # Train function returning error rate of batch 'index'
    train_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        updates = updates,
        givens = {
            x : train_set_x[index * batch_size: (index + 1) * batch_size],
            y : train_set_y[index * batch_size: (index + 1) * batch_size],
            train : numpy.cast['int32'](1),
            mask: srng.binomial(n=1, p=1 - probability, size=(n_circuit,)) # regenerate the mask
        }
    )
    
    # Minimum improvement to be considered as a step forward
    improvement_threshold = 0.995
   
    best_valid_loss = numpy.inf
    start_time = timeit.default_timer()
    train_data = []
    valid_data = []
    test_data = []
    epoch = 0
    
    # Loop through 'n_epochs' iterations
    while (epoch < n_epochs):
        epoch = epoch + 1
        
        # Shuffle batche index for STD
        index = [i for i in xrange(n_train_batches)]
        shuffle(index)
        
        # Calculate training error for all batch
        train_losses = [train_model(i) for i in index]
        this_train_loss = numpy.mean(train_losses) # Average training error for this epoch
        train_data.append(this_train_loss)

        
        # Calculate validating error for all batch
        valid_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_valid_loss = numpy.mean(valid_losses) # Average validating error for this epoch
        valid_data.append(this_valid_loss)
        
        # Display this epoch's stat
        print(
            'epoch %i, train error %f %%, validation error %f %%' %
            (
                epoch,
                this_train_loss * 100.,
                this_valid_loss * 100.
            )
        )
        
        # If validating shows acceptable improvement, calculate testing error
        if this_valid_loss < best_valid_loss * improvement_threshold:     
            best_valid_loss = this_valid_loss 
            
            # Calculate testing error for all batch
            test_losses = [test_model(i) for i in xrange(n_test_batches)]
            this_test_loss = numpy.mean(test_losses) # Average testing error for this epoch
            test_data.append(this_test_loss)
            
            print(
                'epoch %i, test error of best model %f %%' %
                (
                    epoch, 
                    this_test_loss * 100.)
            )

        
    end_time = timeit.default_timer()
    
    output = [train_data, 
              valid_data, 
              test_data,
              end_time - start_time
             ]
    
    return output

