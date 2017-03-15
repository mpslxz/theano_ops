import numpy as np
import cv2


def downsample_volume(volume, ratio=0.5, axis=0):
    downsampled = []
    if axis == 0:
        slice_shape = (int(volume.shape[2]*ratio), int(volume.shape[1]*ratio))
        downsampled += [cv2.resize(volume[i,:,:],slice_shape) for i in range(volume.shape[0])]

    if axis == 1:
        slice_shape = (int(volume.shape[2] * ratio), int(volume.shape[0] * ratio))
        downsampled += [cv2.resize(volume[:,i,:], slice_shape) for i in range(volume.shape[1])]

    if axis == 2:
        slice_shape = (int(volume.shape[1] * ratio), int(volume.shape[0] * ratio))
        downsampled += [cv2.resize(volume[:,:,i], slice_shape) for i in range(volume.shape[2])]

    downsampled = np.array(downsampled)
    if axis == 1:
        return downsampled.swapaxes(0,1)
    if axis == 2:
        return downsampled.swapaxes(0,1).swapaxes(1,2)
    return np.array(downsampled)

def normalize(nd_array):
    return (nd_array - nd_array.mean())/nd_array.std()
    # return (nd_array - nd_array.min()) / (nd_array.max() - nd_array.min())

def to_categorical(y, nb_classes=None):
    """From Keras
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical

class BatchFactory(object):
    def __init__(self, batch_size, nb_samples, iterations, randomizer=True):
        self.BATCH_SIZE = batch_size
        self.nb_samples = nb_samples
        self.iterations = iterations
        self.randomizer = randomizer

    def _index_generator(self):
        for i in range(self.iterations):
            if self.randomizer:
                indices = np.random.permutation(self.nb_samples)
            else:
                indices = np.arange(self.nb_samples)
            for indexer in range(self.nb_samples/self.BATCH_SIZE):
                yield indices[slice(indexer*self.BATCH_SIZE, (indexer+1)*self.BATCH_SIZE)]
            if self.nb_samples % self.BATCH_SIZE != 0:
                indexer = self.nb_samples / self.BATCH_SIZE
                yield indices[slice(indexer*self.BATCH_SIZE, self.nb_samples)]

    def generate_batch(self, X, Y=None):
        for samples in self._index_generator():
            if Y is None:
                yield X[samples]
            else:
                yield X[samples], Y[samples]
