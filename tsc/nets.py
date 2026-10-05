"""Minimal numpy MLP + Adam (no deep-learning framework needed).

Q-network as in the paper's Section 6.1: 3 hidden layers x 64 units,
Leaky-ReLU, Adam (lr 1e-3, beta1 0.9, beta2 0.999), Huber loss.
(The paper's first layer is a 2x2 convolution grouping lanes of the same
road; we use a dense first layer instead.)
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _lrelu(x):
    return np.where(x > 0, x, 0.01 * x)


class MLP:
    def __init__(self, n_in, n_out, hidden=(64, 64, 64), rng=None):
        rng = rng or np.random.default_rng(0)
        sizes = [n_in, *hidden, n_out]
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), (a, b)))
            self.b.append(np.zeros(b))
        self.params = [*self.W, *self.b]

    def forward(self, x, keep=False):
        acts, pre = [x], []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pre.append(z)
            h = z if i == len(self.W) - 1 else _lrelu(z)
            acts.append(h)
        if keep:
            self._cache = (acts, pre)
        return h

    __call__ = forward

    def backward(self, dout):
        """Gradients w.r.t. params (and input) for the last keep=True forward."""
        acts, pre = self._cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            if i != len(self.W) - 1:
                d = d * np.where(pre[i] > 0, 1.0, 0.01)
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            d = d @ self.W[i].T
        self.input_grad = d
        return [*gW, *gb]

    def copy_from(self, other):
        for p, q in zip(self.params, other.params):
            p[...] = q


def huber_grad(err, delta=1.0):
    return np.clip(err, -delta, delta)
