import theano.tensor as T

def relu(x):
    return T.maximum(0.0, x)


def tanh(x):
    return T.tanh(x)


def elu(x):
    return x if T.gt(0.0, x) else T.exp(x) - 1


def sigmoid(x):
    return T.nnet.sigmoid(x)
