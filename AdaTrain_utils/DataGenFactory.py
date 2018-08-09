import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def resize_images(v, size):
    res = []
    for i in range(len(v)):
        res.append(cv2.resize(v[i], size))
    return np.array(res)


def normalize(v):
    for i in range(len(v)):
        v[i] = v[i] / v[i].max()
    return v


class DataGenFactory(object):

    def __init__(self, generative_model, alpha=110, sigma=11, area_threshold=None, deform_labels=True):
        self.generative_model = generative_model
        self.alpha = alpha
        self.sigma = sigma
        self.area_threshold = area_threshold
        self.kernel_sizes = [(9, 9), (11, 11), (13, 13)]
        self.deform_labels = deform_labels
        self.code = None

    def __elastic_transform(self, image):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        assert len(image.shape) == 2
        random_state = np.random.RandomState(None)

        shape = image.shape

        dx = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        return map_coordinates(image, indices, order=1).reshape(shape)

    def __morph_erode(self, image):
        kernel = np.ones(self.kernel_sizes[np.random.randint(3)], np.uint8)
        return cv2.erode(image, kernel)

    def __morph_dilate(self, image):
        kernel = np.ones(self.kernel_sizes[np.random.randint(3)], np.uint8)
        return cv2.dilate(image, kernel)

    def __flip_a_coin(self):
        return np.random.binomial(1, 0.5, 1)[0]

    def __apply_random_transform_labels(self, img, fold=10):
        areas = img.sum(-1).sum(-1)

        res = []
        for j in range(fold):
            for i in img:
                i = self.__elastic_transform(i)
                if i.sum() > areas.mean():  # self.area_threshold * areas.max():
                    i = self.__morph_erode(i)
                if i.sum() < areas.mean():  # (2 - self.area_threshold) * areas.min():
                    i = self.__morph_dilate(i)
                res.append(i)
        return np.array(res)

    def generate_data(self, gen_mu, gen_sigma, labels, fold=1):
        if gen_mu is None:
            return None, None
        if self.deform_labels:
            labels = self.__apply_random_transform_labels(
                labels.squeeze().astype('uint8'), fold).squeeze().astype('uint8')
        if self.area_threshold is not None:
            threshold_idxs = np.where(
                labels.sum(-1).sum(-1) > self.area_threshold)[0]
            labels = labels[threshold_idxs]
        labels = labels.reshape((labels.shape[0], -1))
        self.code = np.random.normal(gen_mu,
                                     gen_sigma,
                                     (labels.shape[0],
                                      self.generative_model.model.nb_latent))
        samples = self.generative_model.model.generate(
            self.code.astype('float32'),
            labels.astype('uint8'))
        samples = normalize(samples)
        return samples, labels
