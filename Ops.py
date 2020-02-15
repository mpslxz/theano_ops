import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d as pool2d, pool_3d as pool3d
from theano.tensor import shared_randomstreams


def bn(inpt, scale=1.0, shift=0.0, layer_name='', init_params=None):
    if init_params is None:
        gamma = theano.shared(np.asarray(scale * np.ones_like(
            inpt), dtype=theano.config.floatX), name='gamma_bn_' + layer_name)
        beta = theano.shared(np.asarray(shift * np.ones_like(
            inpt), dtype=theano.config.floatX), name='beta_bn_' + layer_name)
    else:
        gamma = init_params[0]
        beta = init_params[1]
    mean = T.mean(inpt)
    std = T.std(inpt)
    return T.nnet.batch_normalization(inputs=inpt, gamma=gamma, beta=beta, mean=mean, std=std), [gamma, beta]
   


def conv_1d(inpt, filter_shapes, stride=1, layer_name='', mode='valid', init_params=None):
    stride = (1, stride)
    output_channel = filter_shapes[0]
    input_channel = filter_shapes[1]
    rows = 1
    columns = filter_shapes[2]
    if init_params is None:
        filter_shape = (output_channel, input_channel, rows, columns)
        receptive_field_size = rows * columns

        w = theano.shared(np.asarray(
            np.random.normal(
                loc=0, scale=np.sqrt(2. / ((input_channel + output_channel) * receptive_field_size)),
            size=filter_shape),
            dtype=theano.config.floatX), name='w_conv1d_' + layer_name, borrow=True)

        b = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=1.0, size=(
                        filter_shape[0],)), dtype=theano.config.floatX),
            name='b_conv1d_' + layer_name, borrow=True)
    else:
        w = init_params[0]
        b = init_params[1]
    inpt = inpt.dimshuffle(0, 1, 'x', 2)
    return (T.nnet.conv2d(input=inpt, filters=w, border_mode=mode, subsample=stride) + b.dimshuffle('x', 0, 'x', 'x'))[:, :, 0, :], [w, b]


def conv_3d(inpt, filter_shapes, stride=(1, 1, 1), layer_name='', mode='valid', init_params=None):
    output_channel = filter_shapes[0]
    input_channel = filter_shapes[1]
    depth = filter_shapes[2]
    rows = filter_shapes[3]
    columns = filter_shapes[4]
    if init_params is None:
        filter_shape = (output_channel, input_channel, depth, rows, columns)
        receptive_field_size = depth * rows * columns
        w = theano.shared(np.asarray(
            np.random.normal(
                loc=0, scale=np.sqrt(2. / ((input_channel + output_channel) * receptive_field_size)),
            size=filter_shape),
            dtype=theano.config.floatX), name='w_conv3d_' + layer_name, borrow=True)
        b = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=1.0, size=(
                        filter_shape[0],)), dtype=theano.config.floatX),
            name='b_conv3d_' + layer_name, borrow=True)
    else:
        w = init_params[0]
        b = init_params[1]

    return T.nnet.conv3d(inpt, w, border_mode=mode, subsample=stride) + b.dimshuffle('x', 0, 'x', 'x', 'x'), [w, b]


def conv_2d_transpose(inpt, filter_shapes, stride=(1, 1), layer_name='', mode='valid', init_params=None):
    output_channel = filter_shapes[0]
    input_channel = filter_shapes[1]
    rows = filter_shapes[2]
    columns = filter_shapes[3]
    if init_params is None:
        filter_shape = (output_channel, input_channel, rows, columns)
        output_shape = (output_channel, input_channel, rows + 2, columns + 2)
        receptive_field_size = rows * columns

        w = theano.shared(np.asarray(
            np.random.normal(
                loc=0, scale=np.sqrt(2. / ((input_channel + output_channel) * receptive_field_size)),
            size=filter_shape),
            dtype=theano.config.floatX), name='w_conv2d_' + layer_name, borrow=True)

        b = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=1.0, size=(
                        filter_shape[0],)), dtype=theano.config.floatX),
            name='b_conv2d_' + layer_name, borrow=True)
    else:
        w = init_params[0]
        b = init_params[1]
    return T.nnet.conv2d_transpose(input=inpt, filters=w, output_shape=())


def conv_2d(inpt, filter_shapes, stride=(1, 1), layer_name='', mode='valid', init_params=None):
    output_channel = filter_shapes[0]
    input_channel = filter_shapes[1]
    rows = filter_shapes[2]
    columns = filter_shapes[3]
    if init_params is None:
        filter_shape = (output_channel, input_channel, rows, columns)
        receptive_field_size = rows * columns

        w = theano.shared(np.asarray(
            np.random.normal(
                loc=0, scale=np.sqrt(2. / ((input_channel + output_channel) * receptive_field_size)),
            size=filter_shape),
            dtype=theano.config.floatX), name='w_conv2d_' + layer_name, borrow=True)

        b = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=1.0, size=(
                        filter_shape[0],)), dtype=theano.config.floatX),
            name='b_conv2d_' + layer_name, borrow=True)
    else:
        w = init_params[0]
        b = init_params[1]
    return T.nnet.conv2d(input=inpt, filters=w, border_mode=mode, subsample=stride) + b.dimshuffle('x', 0, 'x', 'x'), [w, b]


def upsample_3d(inpt, ds):
    first_dim = T.extra_ops.repeat(inpt, ds, 4)
    sec_dim = T.extra_ops.repeat(first_dim, ds, 3)
    return T.extra_ops.repeat(sec_dim, ds, 2)


def upsample_2d(inpt, ds):
    one_dim = T.extra_ops.repeat(inpt, ds, 3)
    return T.extra_ops.repeat(one_dim, ds, 2)


def flatten(inpt, ndim=2):
    return T.flatten(inpt, ndim), []


def dense(inpt, nb_in, nb_out, layer_name='', init_params=None):
    if init_params is None:
        w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0, scale=np.sqrt(1. / nb_out), size=[nb_in, nb_out]),
                dtype=theano.config.floatX),
            name='w_dense_' + layer_name, borrow=True)
        b = theano.shared(np.asarray(np.random.normal(
            loc=0.0, scale=1.0, size=[nb_out]),
            dtype=theano.config.floatX),
            name='b_dense_' + layer_name, borrow=True)
    else:
        w = init_params[0]
        b = init_params[1]
    return T.dot(inpt, w) + b, [w, b]


def pool_2d(input, ws, ignore_border=False, mode='max'):
    return pool2d(input=input, ws=ws, ignore_border=ignore_border, mode=mode)


def pool_3d(input, ws, ignore_border=False, mode='max'):
    return pool3d(input=input, ws=ws, ignore_border=ignore_border, mode=mode)


def dropout(inpt, prob=0.25):
    rng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(int(9e+5)))
    mask = rng.binomial(
        n=1, p=1 - prob, size=inpt.shape, dtype=theano.config.floatX)
    return T.mul(inpt, mask), []


def scale(inpt, scale=1.0, shift=0.0, layer_name='', init_params=None):
    """Elemwise multiplication by gamma, add beta.
    Perhaps works when initialized as scale=1 and shift=0
    """
    if init_params is None:
        gamma = scale * T.ones_like(inpt)
        beta = shift * T.ones_like(inpt)
    else:
        gamma = init_params[0]
        beta = init_params[1]
    return T.mul(inpt, gamma) + beta, [gamma, beta]


def zero_pad_3d(inpt, padding=(1, 1, 1)):
    input_shape = inpt.shape
    output_shape = (input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2 * padding[0],
                    input_shape[3] + 2 * padding[1],
                    input_shape[4] + 2 * padding[2])
    output = T.zeros(output_shape)
    indices = (slice(None),
               slice(None),
               slice(padding[0], input_shape[2] + padding[0]),
               slice(padding[1], input_shape[3] + padding[1]),
               slice(padding[2], input_shape[4] + padding[2]))
    return T.set_subtensor(output[indices], inpt)


def zero_pad_2d(inpt, padding=(1, 1)):
    input_shape = inpt.shape
    output_shape = (input_shape[0],
                    input_shape[1],
                    input_shape[2] + 2 * padding[0],
                    input_shape[3] + 2 * padding[1])
    output = T.zeros(output_shape)
    indices = (slice(None),
               slice(None),
               slice(padding[0], input_shape[2] + padding[0]),
               slice(padding[1], input_shape[3] + padding[1]))
    return T.set_subtensor(output[indices], inpt)
