import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor import shared_randomstreams


def bn(inpt, scale=1.0, shift=0.0):
    gamma = scale * T.ones_like(inpt)
    beta = shift * T.ones_like(inpt)
    mean = T.mean(inpt)
    std = T.std(inpt)
    return T.nnet.batch_normalization(inputs=inpt, gamma=gamma, beta=beta, mean=mean, std=std)


def conv_3d(inpt, (output_channel, input_channel, depth, rows, columns), stride=(1, 1, 1), layer_name='', mode='valid'):
    filter_shape = (output_channel, input_channel, depth, rows, columns)
    receptive_field_size = depth * rows * columns
    w = theano.shared(np.asarray(
        np.random.normal(loc=0, scale=np.sqrt(2. / ((input_channel + output_channel) * receptive_field_size)),
                         size=filter_shape),
        dtype=theano.config.floatX), name='w_conv3d_' + layer_name, borrow=True)
    b = theano.shared(
        np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(filter_shape[0],)), dtype=theano.config.floatX),
        name='b_conv3d_' + layer_name, borrow=True)


    return T.nnet.conv3d(inpt, w, border_mode=mode, subsample=stride) + b.dimshuffle('x', 0, 'x', 'x', 'x'), [w, b]


def conv_2d(inpt, (output_channel, input_channel, rows, columns), stride=(1,1), layer_name='', mode='valid'):
    filter_shape = (output_channel, input_channel, rows, columns)
    receptive_field_size = rows* columns

    w = theano.shared(np.asarray(
        np.random.normal(loc=0, scale=np.sqrt(2. / ((input_channel + output_channel) * receptive_field_size)),
                         size=filter_shape),
        dtype=theano.config.floatX), name='w_conv2d_' + layer_name, borrow=True)
    b = theano.shared(
        np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(filter_shape[0],)), dtype=theano.config.floatX),
        name='b_conv2d_' + layer_name, borrow=True)

    return T.nnet.conv2d(input=inpt, filters=w, border_mode=mode, subsample=stride) + b.dimshuffle('x', 0, 'x', 'x'), [w, b]


def upsample_3d(inpt, ds):
    return T.tile(inpt, ds)


def flatten(inpt, ndim=2):
    return T.flatten(inpt, ndim)


def dense(inpt, nb_in, nb_out, layer_name=''):
    w = theano.shared(np.asarray(np.random.normal(loc=0, scale=np.sqrt(1. / nb_out), size=[nb_in, nb_out]),
                                 dtype=theano.config.floatX),
                      name='w_dense_' + layer_name, borrow=True)

    b = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=1.0, size=[nb_out]), dtype=theano.config.floatX),
                      name='b_dense_' + layer_name, borrow=True)
    return T.dot(inpt, w) + b, [w, b]


def max_pool_2d(input, ds, ignore_border=False):
    return pool_2d(input=input, ds=ds, ignore_border=ignore_border)


def max_pool_3d(input, ds, ignore_border=False):
    """
    Takes as input a N-D tensor, where N >= 3. It downscales the input video by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1],ds[2]) (time, height, width)
    :type input: N-D theano tensor of input images.
    :param input: input images. Max pooling will be done over the 3 last dimensions.
    :type ds: tuple of length 3
    :param ds: factor by which to downscale. (2,2,2) will halve the video in each dimension.
    :param ignore_border: boolean value. When True, (5,5,5) input with ds=(2,2,2) will generate a
      (2,2,2) output. (3,3,3) otherwise.
    """

    if input.ndim < 3:
        raise NotImplementedError('max_pool_3d requires a dimension >= 3')

    # extract nr dimensions
    vid_dim = input.ndim
    # max pool in two different steps, so we can use the 2d implementation of
    # downsamplefactormax. First maxpool frames as usual.
    # Then maxpool the time dimension. Shift the time dimension to the third
    # position, so rows and cols are in the back

    # extract dimensions
    frame_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    # store as 4D T with shape: (batch_size,1,height,width)
    new_shape = T.cast(T.join(0, batch_size,
                              T.as_tensor([1, ]),
                              frame_shape), 'int32')
    input_4D = T.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of videos in rows and cols

    output = pool_2d(input_4D, (ds[1], ds[2]), ignore_border=ignore_border)
    # restore to original shape
    outshape = T.join(0, input.shape[:-2], output.shape[-2:])
    out = T.reshape(output, outshape, ndim=input.ndim)

    # now maxpool time

    # output (time, rows, cols), reshape so that time is in the back
    shufl = (list(range(vid_dim - 3)) + [vid_dim - 2] + [vid_dim - 1] + [vid_dim - 3])
    input_time = out.dimshuffle(shufl)
    # reset dimensions
    vid_shape = input_time.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input_time.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,width,time)
    new_shape = T.cast(T.join(0, batch_size,
                              T.as_tensor([1, ]),
                              vid_shape), 'int32')
    input_4D_time = T.reshape(input_time, new_shape, ndim=4)
    # downsample mini-batch of videos in time

    outtime = pool_2d(input_4D_time, (1, ds[0]), ignore_border=ignore_border)
    # output
    # restore to original shape (xxx, rows, cols, time)
    outshape = T.join(0, input_time.shape[:-2], outtime.shape[-2:])

    return T.reshape(outtime, outshape, ndim=input.ndim)


def dropout(inpt, prob=0.25):
    rng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(int(9e+5)))
    mask = rng.binomial(n=1, p=1 - prob, size=inpt.shape, dtype=theano.config.floatX)
    return T.mul(inpt, mask)
