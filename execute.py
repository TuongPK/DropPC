import cPickle

from Non_Fixed_DropCircuit import run_NFD

n_exp = 1

# Network params
n_hidden_node = [[1000, 1000]]
n_circuit = 1

# Training params
n_epochs = 100
learning_rate = 0.5
momentum = 0.4
batch_size = 100
dataset = 'mnist.pkl.gz'

# Drop params
probability = 0

# Sparsity params
# 1 - All layer
# 2 - Penultimate layer
# 3 - No
sparsity = 1

# Result files
name = ['sth_500.pkl']

result = []
for test_id in xrange(n_exp):
    result.append(
            run_NFD(dataset = dataset,
                    n_hidden_node = n_hidden_node[test_id], n_circuit = n_circuit,
                    learning_rate = learning_rate, n_epochs = n_epochs, momentum = momentum, batch_size = batch_size, 
                    probability = probability, sparsity = sparsity
                   )
    )

with open(name[0], 'w') as f:
    cPickle.dump(result, f)

