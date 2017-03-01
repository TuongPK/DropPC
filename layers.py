import numpy
import theano
import theano.tensor as T


class NonLinearLayer(object):
    """ Non-linear Layer
    Layer consists of simple nodes
    """

    def __init__(self, input, n_in, n_out,
                 drop_type, drop, train, probability,
                 W=None, b=None, activation=T.tanh):
        self.input = input

        # If undefined then initialize
        if W is None:
            W_values = numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # Apply activation function on input sum
        lin_output = T.dot(input, self.W) + self.b
        output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # Drop if being trained
        # Reduced if being tested
        if drop_type == 1:
            srng = T.shared_randomstreams.RandomStreams()
            mask = srng.binomial(n=1, p=1 - probability, size=output.shape)

            self.output = T.switch(T.neq(train, 0),
                                   output * T.cast(mask, theano.config.floatX),  # drop according to mask
                                   (output * (1 - probability))
                                   )
        elif drop_type == 2:
            self.output = T.switch(T.neq(train, 0),
                                   output * T.cast(drop, theano.config.floatX),  # drop according to switch
                                   (output * (1 - probability))
                                   )
        else:
            self.output = output

        self.params = [self.W, self.b]
