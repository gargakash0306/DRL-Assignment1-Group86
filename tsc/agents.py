"""Controllers: fixed-time, actuated, DQN and the three regulatable DQN variants
(DRQ, DRSQ, DRHQ) exactly following Algorithm 1 of the paper.
"""
import numpy as np

from .env import ACTIONS, N_ACTIONS, STEP
from .nets import MLP, Adam, huber_grad
from .regulatable import Regulatable

# protected lefts followed by throughs, EW barrier then NS barrier
SEQUENCE = [0, 3, 4, 7]  # (phi1+phi5) -> (phi2+phi6) -> (phi3+phi7) -> (phi4+phi8)


class FixedTime:
    name = "Fixed-time"

    def __init__(self, greens=(15, 35, 15, 25)):
        self.greens, self.k, self.timer = greens, 0, 0

    def reset(self):
        self.k, self.timer = 0, 0

    def act(self, obs, env):
        if self.timer >= self.greens[self.k]:
            self.k, self.timer = (self.k + 1) % 4, 0
        self.timer += STEP
        return SEQUENCE[self.k]


class Actuated:
    """Gap-out actuated control (the 'commonly deployed' baseline): fixed phase
    order, extend green while served queues remain, min/max green limits,
    skip phases with no demand."""
    name = "Actuated"

    def __init__(self, min_green=10, max_green=60):
        self.min_green, self.max_green = min_green, max_green
        self.k, self.timer = 0, 0

    def reset(self):
        self.k, self.timer = 0, 0

    def _demand(self, env, a):
        return sum(len(env.queue[m]) + len(env.approach[m]) for m in ACTIONS[a])

    def act(self, obs, env):
        a = SEQUENCE[self.k]
        served_q = sum(len(env.queue[m]) for m in ACTIONS[a])
        if self.timer >= self.min_green and (served_q == 0 or self.timer >= self.max_green):
            for _ in range(4):
                self.k = (self.k + 1) % 4
                if self._demand(env, SEQUENCE[self.k]) > 0:
                    break
            self.timer = 0
        self.timer += STEP
        return SEQUENCE[self.k]


class Replay:
    def __init__(self, cap, n_flat, rng):
        self.cap, self.rng, self.n, self.i = cap, rng, 0, 0
        self.flat = np.zeros((cap, n_flat))
        self.phase = np.zeros((cap, 8, 6))
        self.flags = np.zeros((cap, N_ACTIONS, 4))
        self.a = np.zeros(cap, int)
        self.r = np.zeros(cap)
        self.flat2 = np.zeros((cap, n_flat))
        self.done = np.zeros(cap)

    def add(self, o, a, r, o2, d):
        i = self.i
        self.flat[i], self.phase[i], self.flags[i] = o["flat"], o["phase"], o["flags"]
        self.a[i], self.r[i], self.flat2[i], self.done[i] = a, r, o2["flat"], d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, k):
        j = self.rng.integers(0, self.n, k)
        return (self.flat[j], self.phase[j], self.flags[j], self.a[j],
                self.r[j], self.flat2[j], self.done[j])


def _softmax(x):
    z = x - x.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class DQNFamily:
    """variant in {"DQN", "DRQ", "DRSQ", "DRHQ"} -- Algorithm 1.

    DQN : act with argmax_a Q(s,a)                         (line 9)
    D*Q : act with argmax_a G(s,a) (regulatable, line 10); Q is still trained
          off-policy (lines 13-16) and G is trained from Q every step
          over N minibatches (lines 17-29).
    """

    def __init__(self, variant, n_flat, gamma, seed=0, lr=1e-3, batch=32,
                 replay=100_000, target_every=500, n_g_updates=2):
        self.name = variant
        self.variant = variant
        self.rng = np.random.default_rng(seed)
        self.q = MLP(n_flat, N_ACTIONS, rng=self.rng)
        self.qt = MLP(n_flat, N_ACTIONS, rng=self.rng)
        self.qt.copy_from(self.q)
        self.opt = Adam(self.q.params, lr)
        self.G = Regulatable()
        self.gopt = Adam(self.G.params, lr)
        self.D = Replay(replay, n_flat, self.rng)
        self.gamma, self.batch, self.target_every, self.N = gamma, batch, target_every, n_g_updates
        self.eps, self.steps = 0.05, 0

    def reset(self):
        pass

    # ------------------------------------------------------------------
    def act(self, obs, env=None):
        if self.rng.random() < self.eps:                       # line 8
            return int(self.rng.integers(N_ACTIONS))
        if self.variant == "DQN":                              # line 9
            return int(np.argmax(self.q(obs["flat"][None])[0]))
        return int(np.argmax(self.G(obs["phase"][None], obs["flags"][None])[0]))  # line 10

    def observe(self, o, a, r, o2, done):
        self.D.add(o, a, r, o2, done)                          # line 12
        self.steps += 1
        if self.D.n < self.batch:
            return
        self._train_q()                                        # lines 13-15
        if self.steps % self.target_every == 0:                # line 16
            self.qt.copy_from(self.q)
        if self.variant != "DQN":
            for _ in range(self.N):                            # lines 17-18
                self._train_g()

    # ------------------------------------------------------------------
    def _train_q(self):
        s, _, _, a, r, s2, d = self.D.sample(self.batch)
        y = r + self.gamma * (1 - d) * self.qt(s2).max(1)      # line 14
        qv = self.q.forward(s, keep=True)
        err = qv[np.arange(len(a)), a] - y
        dout = np.zeros_like(qv)
        dout[np.arange(len(a)), a] = huber_grad(err) / len(a)  # line 15 (Huber)
        self.opt.step(self.q.backward(dout))

    def _train_g(self):
        s, ph, fl, a, r, s2, d = self.D.sample(self.batch)
        G = self.G(ph, fl)
        B = len(a)
        if self.variant == "DRQ":                              # lines 19-21
            y = r + self.gamma * (1 - d) * self.qt(s2).max(1)
            dG = np.zeros_like(G)
            dG[np.arange(B), a] = 2 * (G[np.arange(B), a] - y) / B
        else:
            Q = self.q(s)
            if self.variant == "DRSQ":                         # lines 22-25
                X = _softmax(Q)
            else:                                              # DRHQ, lines 26-29
                X = np.zeros_like(Q)
                X[np.arange(B), Q.argmax(1)] = 1.0
            Z = _softmax(G)
            dG = (Z - X) / B                                   # d(-sum X log Z)/dG
        self.gopt.step(self.G.grads(ph, fl, dG))
        self.G.clamp()
