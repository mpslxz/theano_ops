import theano
import pickle
import sys, os
import theano.tensor as T
import numpy as np
from abc import abstractmethod
from progressbar import ProgressBar, Bar, ETA, Percentage

import config


class TheanoModel(object):
    
    def __init__(self, batch_size, input_shape, optimizer, metrics):
        if not os.path.exists(config.ckpt_dir):
            os.mkdir(config.ckpt_dir)
        sys.setrecursionlimit(0x100000)
        self.INPUT_SHAPE = input_shape
        self.BATCH_SIZE = batch_size
        self.params = []
        self.history = []
        self.optimizer = optimizer
        self.metrics = metrics
        self.mode = T.compile.get_default_mode()

        self._def_tensors()
        self._def_arch()
        self._def_cost_acc()
        self._def_outputs()
        self._def_functions()


    def _def_outputs(self):
        self.output_metrics = []
        if 'acc' in self.metrics:
            self.output_metrics.insert(self.metrics.index('acc'), self.acc)
        if 'loss' in self.metrics:
            self.output_metrics.insert(self.metrics.index('loss'), self.cost)


    @abstractmethod
    def _def_tensors(self):

        self.x = None
        self.y = None
        self.indexer = T.lscalar()
        raise NotImplementedError


    @abstractmethod
    def _def_arch(self):
        self.outputs = None
        raise NotImplementedError


    @abstractmethod
    def _def_cost_acc(self):
        self.cost = None
        self.acc = None
        raise NotImplementedError


    def _def_functions(self):
        self.batch_test_fcn = theano.function([self.x, self.y], outputs=self.acc, mode=self.mode)
        self.batch_train_fcn = theano.function([self.x, self.y],
                                          outputs=self.output_metrics,
                                          updates=self.optimizer.updates(cost=self.cost, params=self.params),
                                               mode=self.mode)
        self.predict_fcn = theano.function([self.x], outputs=self.outputs, mode=self.mode)


    def get_shape(self, layer):
        dummy = np.random.random(self.INPUT_SHAPE).astype(np.float32)
        _get_shape = theano.function([self.x], layer)
        return _get_shape(dummy).shape


    def _get_batch(self, x, y, iteration, randomizer):
        return x[randomizer[iteration*self.BATCH_SIZE: (iteration + 1) * self.BATCH_SIZE]], \
               y[randomizer[iteration*self.BATCH_SIZE: (iteration + 1) * self.BATCH_SIZE]]


    def train(self, x_train, y_train, x_validation=None, y_validation=None, nb_epochs=100, verbose=True, gpu_memory=False):

        nb_batches = len(x_train) / self.BATCH_SIZE
        nb_samples = len(x_train)

        if gpu_memory:
            x_train_shared= theano.shared(x_train)
            y_train_shared = theano.shared(y_train)
            self._train_fcn_gpu = theano.function([self.indexer],
                                              outputs=self.output_metrics,
                                              updates=self.optimizer.updates(cost=self.cost, params=self.params),
                                              givens={self.x: x_train_shared[self.indexer*self.BATCH_SIZE:
                                                      (1+self.indexer)*self.BATCH_SIZE],
                                                      self.y: y_train_shared[self.indexer*self.BATCH_SIZE:
                                                      (1+self.indexer)*self.BATCH_SIZE]})

        for i in range(nb_epochs):
            print "\niteration: {} of {}".format(i + 1, nb_epochs)
            self.randomizer = np.random.permutation(nb_samples)
            vals = []

            global_steps = range(nb_batches)
            if verbose:
                pbar = ProgressBar(widgets=[Percentage(), ' ', Bar('='), ' ', ETA()], maxval=nb_batches)
                global_steps = pbar(global_steps)
            for step in global_steps:
                if gpu_memory:
                    vals += [self._train_fcn_gpu(step)]
                else:
                    x, y = self._get_batch(x_train, y_train, step, self.randomizer)
                    vals += [self.batch_train_fcn(x, y)]

            train_vals = [(name, "{:.4f}".format(val)) for name, val in zip(self.metrics, list(np.array(vals).mean(axis=0)))]
            self.history += train_vals
            for res in train_vals:
                print "train", res[0], res[1]
            if x_validation is not None:
                validation_acc = self.test(x_validation, y_validation)
                self.history += [('val_acc', "{:.4f}".format(validation_acc))]
                print "validation acc {:.4f}".format(validation_acc)
            if i % 10 == 0:
                print "writing model checkpoint to " + config.ckpt_dir
                self.freeze()


    def test(self, x_test, y_test):
        nb_batches = len(x_test) / self.BATCH_SIZE
        randomizer = np.arange(len(x_test))
        vals = []
        for j in range(nb_batches):
            x_, y_ = self._get_batch(x_test, y_test, j,randomizer)
            vals += [self.batch_test_fcn(x_, y_)]
        return np.array(vals).mean()


    def predict(self, x):
        return self.predict_fcn(x)


    def freeze(self):
        with open(config.ckpt_dir+'model_snapshot.ckpt', 'wb') as checkpoint_file:
            pickle.dump(self, checkpoint_file)