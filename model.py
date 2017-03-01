import timeit
from random import shuffle
# MNIST does not have normalization, other datasets have
from mnist_sgd import LogisticRegression, mnist_load_data
from logistic_sgd import load_data

from layers import *


class Circuit(object):
    """ Parallel Circuit components
    """

    def __init__(self, input, n_in, n_hidden_layer, n_hidden_node,
                 train, drop_type, probability, drop, sparsity):
        """ Initialize
        :param input            : input
        :param n_in             : number of input features
        :param n_hidden_layer   : number of hidden layers
        :param n_hidden_node    : network architectures
        :param train            : training flag
                                    1 - Train
                                    0 - Test
        :param drop_type        : type of dropout
                                    1 - Node Dropout
                                    2 - Drop Circuit
                                    3 - None
        :param probability      : dropping probability
        :param drop             : drop flag
                                    0 - Dropped
                                    1 - Survived
        :param sparsity         : type of sparsity penalty
                                    1 - All layer
                                    2 - Penultimate layer
                                    3 - None
        """

        self.hidden_layers = []
        for index in range(n_hidden_layer):
            self.hidden_layers.append(
                NonLinearLayer(
                    input=input if index == 0 else self.hidden_layers[index - 1].output,
                    n_in=n_in if index == 0 else n_hidden_node[index - 1],
                    n_out=n_hidden_node[index],
                    train=train,
                    drop_type=drop_type,
                    probability=probability,
                    drop=drop,
                )
            )

        # Sparsity penalty
        self.L1 = 0
        self.L2_sqr = 0

        if sparsity == 1:  # full network sparsity
            for layer in self.hidden_layers:
                self.L1 += abs(layer.output).sum()
                self.L2_sqr += (layer.output ** 2).sum()
        elif sparsity == 2:  # penultimate sparsity
            self.L1 += abs(self.output).sum()
            self.L2_sqr += (self.output ** 2).sum()
        else:  # no sparsity
            self.L1 = 0
            self.L2_sqr = 0

        # Input, output, weight and bias
        self.params = []

        for layer in self.hidden_layers:
            self.params += layer.params

        self.output = self.hidden_layers[n_hidden_layer - 1].output
        self.input = input


class Model(object):
    def __init__(self, input, n_in, n_hidden_layer, n_hidden_node, n_circuit, n_out,
                 train, drop_type, probability, mask,
                 sparsity, varied_coef):
        """
        :param input            : input
        :param n_in             : number of input features
        :param n_hidden_layer   : number of layers
        :param n_hidden_node    : network structures
        :param n_circuit        : number of circuits
        :param n_out            : number of output class
        :param train            : train flag
        :param drop_type        : type of dropout
                                    1 - Node Dropout
                                    2 - Drop Circuit
                                    3 - None
        :param probability      : dropping probability
        :param mask             : drop mask
        :param sparsity         : type of sparsity penalty
                                    1 - All layer
                                    2 - Penultimate layer
                                    3 - None
        :param varied_coef      : sparsity coefficient
                                    1 - Random
                                    2 - Fixed
        """
        # Random sparsity coefficients
        if varied_coef == 1:
            L1_reg = numpy.random.uniform(low=0, high=0.002, size=(n_circuit,)) * 0  # Not using L1
            L2_reg = numpy.random.uniform(low=0, high=0.0002, size=(n_circuit,))
        elif varied_coef == 2:
            L1_reg = numpy.zeros(shape=(n_circuit,))
            L2_reg = numpy.zeros(shape=(n_circuit,)) + 0.0001

        # Initialize circuits in network
        self.circuits = []

        for index in xrange(n_circuit):
            self.circuits.append(
                Circuit(
                    input=input,
                    n_hidden_layer=n_hidden_layer,
                    n_hidden_node=[n / n_circuit for n in n_hidden_node],
                    n_in=n_in,  # sharing same input
                    train=train,
                    drop_type=drop_type,
                    drop=mask[index],
                    probability=probability,
                    sparsity=sparsity
                )
            )

        # Shared output
        penultimate_layer = []

        for circuit in self.circuits:
            penultimate_layer.append(circuit.output)

        self.logRegressionLayer = LogisticRegression(
            input=T.concatenate(penultimate_layer, axis=1),
            n_in=n_hidden_node[n_hidden_layer - 1],
            n_out=n_out
        )

        # Sum the total sparsity penalty of circuits
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


def evaluate(dataset,
             n_hidden_node, n_circuit,
             learning_rate, n_epochs, momentum, batch_size,
             drop_type, probability,
             sparsity, varied_coef
             ):
    """
    :param dataset          : name of dataset
    :param n_hidden_node    : network architecture
    :param n_circuit        : number of circuits
    :param learning_rate    : learning rate
    :param n_epochs         : number of training epochs
    :param momentum         : momentum
    :param batch_size       : minibatch size
    :param drop_type        : type of dropout
                                1 - Node Dropout
                                2 - Drop Circuit
                                3 - None
    :param probability      : dropping probability
    :param sparsity         : type of sparsity penalty
                                1 - All layer
                                2 - Penultimate layer
                                3 - None
    :param varied_coef      : sparsity coefficient
                                1 - Random
                                2 - Fixed
    """

    # Load and divide datasets
    # Training and testing set of MNIST are predefined. Others are at a whole

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

    train = T.iscalar()  # Dropout switch

    # Create PC drop mask
    srng = T.shared_randomstreams.RandomStreams()
    mask = srng.binomial(n=1, p=1 - probability, size=(n_circuit,))

    # Initialize network
    classifier = Model(
        input=x,
        n_in=784,
        n_hidden_layer=len(n_hidden_node),
        n_hidden_node=n_hidden_node,
        n_circuit=n_circuit,
        n_out=10,
        train=train,
        mask=mask,
        drop_type=drop_type,
        probability=probability,
        sparsity=sparsity,
        varied_coef=varied_coef
    )

    # Apply sparsity penalty to error cost
    cost = (
        classifier.negative_log_likelihood(y)
        + classifier.L1
        + classifier.L2_sqr
    )

    # Test function returning error rate of batch 'index'
    test_model = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            train: numpy.cast['int32'](0)
        }
    )

    # Validate function returning error rate of batch 'index'
    validate_model = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            train: numpy.cast['int32'](0)
        }
    )

    # Compute gradient for all weight & bias matrices
    g_params = [T.grad(cost, param) for param in classifier.params]

    # Weight & bias update
    updates = []
    for param, g_param in zip(classifier.params, g_params):
        param_update = theano.shared(param.get_value() * 0.)  # nullify

        updates.append((param, param - learning_rate * param_update))  # main update
        updates.append((param_update, momentum * param_update + (1 - momentum) * g_param))  # include momentum

    # Calculate the sum of first layer gradient
    gradient_magnitude = 0

    for circuit in xrange(n_circuit):
        # Note the arrangement of circuit gradient
        gradient_magnitude += (numpy.abs(T.grad(cost, classifier.params[circuit * len(n_hidden_node)]))).sum()

    # Train function returning error rate of batch 'index'
    train_model = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=[classifier.errors(y), gradient_magnitude],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            train: numpy.cast['int32'](1),
            mask: srng.binomial(n=1, p=1 - probability, size=(n_circuit,))  # regenerate the mask
        }
    )

    # Minimum improvement to be considered as a step forward
    improvement_threshold = 0.995

    best_valid_loss = numpy.inf
    start_time = timeit.default_timer()
    train_data = []
    valid_data = []
    test_data = []
    gradient_data = []
    epoch = 0

    # Loop through 'n_epochs' iterations
    while epoch < n_epochs:
        epoch += 1

        # Cooling
        learning_rate *= 0.999

        # Shuffle batch index for STD
        index = [i for i in xrange(n_train_batches)]
        shuffle(index)

        # Calculate training error for all batch
        train_losses = []
        gradient = []

        for i in index:
            train_losses_temp, gradient_temp = train_model(i)
            train_losses.append(train_losses_temp)
            gradient.append(gradient_temp)

        this_train_loss = numpy.mean(train_losses)  # Average training error for this epoch
        train_data.append(this_train_loss)

        # Save the change in first layer gradient
        gradient = numpy.mean(gradient)
        gradient_data.append(numpy.mean(gradient))

        # Calculate validating error for all batch
        valid_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_valid_loss = numpy.mean(valid_losses)  # Average validating error for this epoch
        valid_data.append(this_valid_loss)

        # Display this epoch's stat
        print(
            'epoch %i, train error %f %%, validation error %f, gradient %f %%' %
            (
                epoch,
                this_train_loss * 100.,
                this_valid_loss * 100.,
                gradient
            )
        )

        # If validating shows acceptable improvement, calculate testing error
        if this_valid_loss < best_valid_loss * improvement_threshold:
            best_valid_loss = this_valid_loss

            # Calculate testing error for all batch
            test_losses = [test_model(i) for i in xrange(n_test_batches)]
            this_test_loss = numpy.mean(test_losses)  # Average testing error for this epoch
            test_data.append(this_test_loss)

            print(
                'epoch %i, test error of best model %f %%' %
                (
                    epoch,
                    this_test_loss * 100.)
            )

    end_time = timeit.default_timer()

    # Weight distribution
    weight_matrix = []
    for layer_index in xrange(len(classifier.params)):
        # Hard coded for 2 layers
        if layer_index % 2 == 0:
            weight_matrix.append(classifier.params[layer_index].get_value(borrow=True))

    output = [train_data,
              valid_data,
              test_data,
              gradient_data,
              end_time - start_time,
              weight_matrix
              ]

    return output
