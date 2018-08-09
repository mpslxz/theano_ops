import numpy as np
import config as CONF
from proc_utils import normalize


class ScheduleFactory(object):

    def __init__(self,
                 BatchFactory,
                 SamplingFactoryObject,
                 DataGenFactoryObject,
                 iter_list):

        self.iter_list = iter_list
        self.BatchFactory = BatchFactory
        self.SamplingFactoryObject = SamplingFactoryObject
        self.DataGenFactoryObject = DataGenFactoryObject
        self.__dist_params = []
        self.__iter_ptr = 0
        self.__gen_costs = None

    def __update_dist_params(self):
        if self.__iter_ptr == 0:
            self.__dist_params.append((None, None))
        else:
            self.__dist_params.append(
                self.SamplingFactoryObject.new_sampling_params(
                    self.__gen_costs,
                    self.DataGenFactoryObject.code))

    def __augment_data(self, x, y, gen_x, gen_y):
        if gen_x is None:
            return x, y, None, None
        samples = normalize(
            gen_x.reshape(len(gen_x), 1, CONF.INPUT_SIZE[2], CONF.INPUT_SIZE[3]))
        distorted_labels = gen_y.reshape(len(samples),
                                         1,
                                         CONF.INPUT_SIZE[2], CONF.INPUT_SIZE[3])

        return np.vstack((x, samples.astype('float32'))), np.vstack((y, distorted_labels.astype('int8'))), samples.astype('float32'), distorted_labels.astype('int8')

    def __make_batcher(self, x, y, batch_size):
        nb_samples = len(x)
        self.nb_batches = np.ceil(1. * nb_samples / batch_size)

        batch_engine = self.BatchFactory(
            batch_size=batch_size, nb_samples=nb_samples, iterations=self.iter_list[self.__iter_ptr])
        return batch_engine.generate_batch(X=x, Y=y)

    def get_next_batcher(self, x, y, batch_size, fold):
        """Function to return the next batch generator

        :param x: data
        :param y: labels
        :returns: None if at the end of the iter_list otherwise next batch generator
        :rtype:

        """
        batcher = None
        if self.__iter_ptr == len(self.iter_list):
            return batcher, (None, None)
        self.__update_dist_params()
        mu, sigma = self.__dist_params[-1]
        gen_x, gen_y = self.DataGenFactoryObject.generate_data(mu,
                                                               sigma,
                                                               y,
                                                               fold)
        x_aug, y_aug, gen_x, gen_y = self.__augment_data(x, y, gen_x, gen_y)

        batcher = self.__make_batcher(x_aug, y_aug, batch_size)
        self.__iter_ptr += 1
        return batcher, (gen_x, gen_y)

    def set_gen_costs(self, costs):
        self.__gen_costs = costs

    def get_gen_costs(self):
        return self.__gen_costs

    def get_nb_batches(self):
        return self.nb_batches

    def get_nb_epochs(self):
        return self.iter_list[self.__iter_ptr]

    def get_dist_params(self):
        return self.__dist_params
