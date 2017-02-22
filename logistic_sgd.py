
# coding: utf-8

# In[15]:

import numpy
import theano
import theano.tensor as T
import timeit
import cPickle
import sys
import os

from sklearn import preprocessing

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
       
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )
        
        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        
        self.params = [self.W, self.b]
        
        self.input = input
        
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
            
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
def load_data(dataset):
    
    feature = numpy.genfromtxt(dataset, delimiter=',', dtype=theano.config.floatX, usecols=range(2, 11))
    target = numpy.genfromtxt(dataset, delimiter=',', dtype=theano.config.floatX, usecols=1) - 1
       
    scaler = preprocessing.MinMaxScaler()
    feature = scaler.fit_transform(feature)
    
    #feature = preprocessing.normalize(feature)
    
    train_set = (feature[0:40], target[0:40])
    valid_set = (feature[40:60], target[40:60])
    test_set = (feature[60:100], target[60:100])
    
    def shared_dataset(data_xy, borrow = True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype = 'int32'), borrow = borrow)
        
        return shared_x, shared_y
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval

def sgd_optimization(learning_rate = 0.13, momentum = 0.4, n_epochs = 100, dataset = 'processed_breast_tissue.data', batch_size = 10):
    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
   
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow = True).shape[0] / batch_size
    
    
    print train_set_x.get_value(borrow = True)
    
    print('... Creating model')
    
    index = T.lscalar()
    
    x = T.matrix('x')
    y = T.ivector('y')
    
      
    classifier = LogisticRegression(input = x, n_in = 9, n_out = 6)
    
    cost = classifier.negative_log_likelihood(y)
    
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: test_set_x[index * batch_size : (index + 1) * batch_size],
            y: test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: valid_set_x[index * batch_size : (index + 1) * batch_size],
            y: valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )
    
    
    W_update = theano.shared(classifier.W.get_value()*0.)
    b_update = theano.shared(classifier.b.get_value()*0.)
    
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)
    
    updates = [(classifier.W, classifier.W - learning_rate * W_update), 
               (W_update, momentum * W_update + (1 - momentum) * g_W), 
               (classifier.b, classifier.b - learning_rate * b_update),
               (b_update, momentum * b_update + (1 - momentum) * g_b)]
    
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        
        givens = {
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )
    
    print('... Training Model')
    
    patience = 5000
    patience_increase = 2
    
    improvement_threshold = 0.995
    
    validation_frequency = min(n_train_batches, patience / 2)
    
    best_validation_loss = numpy.inf
    test_score = 0
    start_time = timeit.default_timer()
    
    done_looping = False
    epoch = 0
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in xrange(n_train_batches):
            
            out = train_model(minibatch_index)

            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss
                    
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    
                    test_score = numpy.mean(test_losses)
                    
                    print(
                        'epoch %i, minibatch %i/%i, test score of best model %f %%' % 
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)
                        
            if patience <= iter:
                done_looping = True
                break
    
    end_time = timeit.default_timer()
                    
    print(
        'Training completed with best validatation score %f %%, best test performance %f %%' % 
        (
            best_validation_loss * 100.,
            test_score * 100.
        )
    )
        
    print(
        'The code run for %d epochs, with %f epochs/sec' % 
        (
            epoch,
            1. * epoch / (end_time - start_time)
        )
    )
    
    print >> sys.stderr, ('The code for file ' 
                          + os.path.split('__file__')[1] 
                          + ' ran for %.1fs' % (end_time - start_time))
    
def predict():
    classifier = cPickle.load(open('best_model.pkl'))
    
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs = [cost]
    )
    
    dataset = 'wdbc.data'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    
    predicted_values = predict_model(test_set_x)
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values
    
if __name__ == '__main__':
    sgd_optimization()




# In[ ]:




# In[ ]:



