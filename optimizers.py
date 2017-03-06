import numpy as np
import theano
import theano.tensor as T


class RMSProp(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, cost, params, lr=1e-4, momentum=0.0, rescale=5.):
        grads = T.grad(cost, params)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - lr * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * lr * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class SGDNesterov(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, cost, params, lr=1e-4, momentum=0.0):
        grads = T.grad(cost, params)
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - lr * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * lr * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class SGD(object):
    # Only here for API conformity with other optimizers
    def __init__(self):
        pass

    def updates(self, cost, params, lr=1e-4):
        grads = T.grad(cost, params)
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - lr * grad))
        return updates


class Adam(object):

    def updates(self, cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8,
             gamma=1 - 1e-8):

        updates = []
        grads = theano.grad(cost, params)
        alpha = lr
        t = theano.shared(np.float32(1))
        b1_t = b1 * gamma ** (t - 1)  # (Decay the first moment running average coefficient)

        for param, grad in zip(params, grads):
            m_previous = theano.shared(np.zeros(param.get_value().shape,
                                                dtype=theano.config.floatX))
            v_previous = theano.shared(np.zeros(param.get_value().shape,
                                                dtype=theano.config.floatX))

            m = b1_t * m_previous + (1 - b1_t) * grad  # (Update biased first moment estimate)
            v = b2 * v_previous + (1 - b2) * grad ** 2  # (Update biased second raw moment estimate)
            m_hat = 1.*m / (1 - b1 ** t)  # (Compute bias-corrected first moment estimate)
            v_hat = 1.*v / (1 - b2 ** t)  # (Compute bias-corrected second raw moment estimate)
            theta = param - (alpha * m_hat) / (T.sqrt(v_hat) + e)  # (Update parameters)

            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((param, theta))
        updates.append((t, t + 1.))
        return updates

