import numpy as np


class BiasedSampleFactory(object):

    def __init__(self, biased_sampling=False, init_mean=0, init_std=50):
        self.biased_sampling = biased_sampling
        self.init_mean = init_mean
        self.init_std = init_std

    def __get_new_dist_params(self, costs, samples):
        old_mean = np.mean(costs)
        contrib_samples = samples[np.where(costs > old_mean)[0]]
        return np.mean(contrib_samples), np.std(contrib_samples)

    def new_sampling_params(self, costs, samples):
        if samples is None or costs is None or not self.biased_sampling:
            return self.init_mean, self.init_std
        return self.__get_new_dist_params(costs, samples)
