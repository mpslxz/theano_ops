import theano.tensor as T


def relu(x):
    return T.maximum(0.0, x)


def tanh(x):
    return T.tanh(x)


def elu(x):
    return T.nnet.elu(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)
