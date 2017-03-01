import cPickle
from model import evaluate
import os

if __name__ == '__main__':
    """ Experiment parameters
    :param n_exp            : number of experiments
    """
    n_exp = 1

    """ Network parameters
    :param n_hidden_node    : network architectures
    :param n_circuit        : number of circuits
    """
    n_hidden_node = [100, 100]
    n_circuit = 10

    """ Training parameters
    :param n_epochs         : number of epochs
    :param learning_rate    : learning rate
    :param momentum         : momentum
    :param batch_size       : size of mini batch
    :param dataset          : name of the data file
    """
    n_epochs = 1
    learning_rate = 0.1
    momentum = 0.4
    batch_size = 100
    dataset = 'mnist.pkl.gz'

    """ Dropout parameters
    :param drop_type        : type of dropout
                                1 - Node Dropout
                                2 - Drop Circuit
                                3 - None
    :param probability      : dropping probability
    """
    drop_type = 2
    probability = [0.1,
                   0.2,
                   0.3,
                   0.4,
                   0.5,
                   0.6,
                   0.7,
                   0.8,
                   0.9]

    """ Sparsity parameters
    :param sparsity         : type of sparsity penalty
                                1 - All layer
                                2 - Penultimate layer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                3 - None
    :param varied_coef      : sparsity coefficient
                                1 - Random
                                2 - Fixed
    """
    sparsity = 1
    varied_coef = 1

    """ Result export
    :param name             : result data filename
    """
    name = ['100_100_PC_10_1.pkl',
            '100_100_PC_10_2.pkl',
            '100_100_PC_10_3.pkl',
            '100_100_PC_10_4.pkl',
            '100_100_PC_10_5.pkl',
            '100_100_PC_10_6.pkl',
            '100_100_PC_10_7.pkl',
            '100_100_PC_10_8.pkl',
            '100_100_PC_10_9.pkl',
            ]

    for index in xrange(len(name)):
        name[index] = os.path.join(os.path.split(__file__)[0], name[index])

    """ Main program
    """
    for setup_id in xrange(len(name)):
        result = []

        for test_id in xrange(n_exp):
            print(\
                     'Setup #%d, experiment #%d '
            ) % (setup_id, test_id)

            result.append(
                evaluate(n_hidden_node=n_hidden_node, n_circuit=n_circuit,
                         learning_rate=learning_rate, n_epochs=n_epochs, momentum=momentum, batch_size=batch_size,
                         dataset=dataset,
                         drop_type=drop_type, probability=probability[setup_id],
                         sparsity=sparsity, varied_coef=varied_coef
                         )
            )

            with open(name[setup_id], 'w') as f:
                cPickle.dump(result, f)
