import theano.tensor as T
from Ops import flatten


def dice(output, label):
    smooth = 0.005
    pred = flatten(output)
    target = flatten(label)
    intersection = T.sum(pred * target)
    dice_value = (2. * intersection + smooth) / (T.sum(pred) + T.sum(target) + smooth)
    return dice_value

def log_likelihood(output, label):
    return T.mean(T.log(output)[T.arange(label.shape[0]), label])