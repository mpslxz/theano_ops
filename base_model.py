import theano
import pickle
import sys, os
import numpy as np
import theano.tensor as T
import numpy as np
from abc import abstractmethod
from progressbar import ProgressBar, Bar, ETA, Percentage

import config
from utils import BatchFactory


class TheanoModel(object):
    def __init__(self, batch_size, input_shape, optimizer, metrics, lmbd=0, init_params=None):
        print("initializing model")
        if not os.path.exists(config.ckpt_dir):
            os.mkdir(config.ckpt_dir)
        sys.setrecursionlimit(0x100000)
        self.INPUT_SHAPE = input_shape
        self.BATCH_SIZE = batch_size
        self.params = []
        self.history = [] if init_params is None else init_params[1]
        self.optimizer = optimizer
        self.metrics = metrics
        self.mode = T.compile.get_default_mode()
        self.lmbd = lmbd
        self.to_regularize = []
        
        self._def_tensors()
        params = init_params[0] if init_params is not None else init_params
        self._def_arch(params)
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
    def _def_arch(self, init_params=None):
        self.outputs = None
        raise NotImplementedError

    @abstractmethod
    def _def_cost_acc(self):
        self.cost = None
        self.acc = None
        raise NotImplementedError

    def _def_functions(self):
        print("compiling model")
        self.batch_test_fcn = theano.function([self.x, self.y], outputs=self.acc, mode=self.mode)
        self.batch_train_fcn = theano.function([self.x, self.y],
                                          outputs=self.output_metrics,
                                          updates=self.optimizer.updates(cost=self.cost, params=self.params),
                                               mode=self.mode)
        self.predict_fcn = theano.function([self.x], outputs=self.outputs, mode=self.mode)

    def get_shape(self, layer):
        dummy = np.random.random(self.INPUT_SHAPE).astype(theano.config.floatX)
        _get_shape = theano.function([self.x], layer)
        return _get_shape(dummy).shape

    def get_params(self, layer_name, param_list):
        if param_list is None:
            return None
        params = [i for i in param_list if i.name.split('_')[2] == layer_name]
        return None if len(params) == 0 else params

    def train(self, x_train, y_train, x_validation=None, y_validation=None, nb_epochs=100, overwrite=True):

        nb_samples = len(x_train)
        nb_batches = np.ceil(1.*nb_samples / self.BATCH_SIZE)

        batch_engine = BatchFactory(batch_size=self.BATCH_SIZE, nb_samples=nb_samples, iterations=nb_epochs)
        batcher = batch_engine.generate_batch(X=x_train, Y=y_train)

        vals = []
        pbar = ProgressBar(widgets=[Percentage(), ' ', Bar('='), ' ', ETA()], maxval=nb_batches)
        print "\niteration {} of {}".format(1, nb_epochs)
        pbar.start()
        iteration = 0
        for ind, (x, y) in enumerate(batcher):
            vals += [self.batch_train_fcn(x, y)]
            pbar.update((ind + 1) % nb_batches)
            if (ind + 1) % nb_batches == 0:
                iteration += 1
                pbar.finish()
                train_vals = [(name, "{:.4f}".format(val)) for name, val in
                              zip(self.metrics, list(np.array(vals).mean(axis=0)))]
                self.history += train_vals
                for res in train_vals:
                    print "train", res[0], res[1]
                if x_validation is not None:
                    validation_acc = self.test(x_validation, y_validation)
                    self.history += [('val_acc', "{:.4f}".format(validation_acc))]
                    print "validation acc {:.4f}".format(validation_acc)
                vals =[]
                if (iteration + 1) % 10 == 0:
                    if overwrite:
                        self.freeze()
                    else:
                        self.freeze(iteration + 1)
                if ind != nb_epochs * nb_batches - 1:
                    print "\niteration {} of {}".format(iteration + 1, nb_epochs)
                    pbar.start()

        self.freeze()

    def test(self, x_test, y_test):
        batch_engine = BatchFactory(batch_size=self.BATCH_SIZE, nb_samples=len(x_test), iterations=1)
        batcher = batch_engine.generate_batch(X=x_test, Y=y_test)
        vals = []
        for idx, (x_, y_) in enumerate(batcher):
            vals += [self.batch_test_fcn(x_, y_)]
        return np.array(vals).mean()

    def predict(self, x):
        print "Predict"
        batch_engine = BatchFactory(batch_size=self.BATCH_SIZE, nb_samples=len(x), iterations=1, randomizer=False)
        batcher = batch_engine.generate_batch(X=x)
        predictions = []
        nb_batches = np.ceil(1.*len(x)/self.BATCH_SIZE)
        pbar = ProgressBar(widgets=[Percentage(), ' ', Bar('=')], maxval=nb_batches)
        pbar.start()
        for idx, x_ in enumerate(batcher):
            predictions.extend(self.predict_fcn(x_))
            pbar.update(idx)
        pbar.finish()
        return np.array(predictions)

    def freeze(self, idx=None):
        file_name = config.ckpt_dir+'model_snapshot' if idx is None else config.ckpt_dir+'model_snapshot{}'.format(idx)
        np.save(file_name, (self.params, self.history))

    @staticmethod
    def restore_params(file_name):
        return list(np.load(file_name))
