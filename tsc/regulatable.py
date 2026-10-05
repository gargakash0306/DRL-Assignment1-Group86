"""The designed regulatable precedence function (Eq. 1 of the paper).

    g(s, PHI; theta') = [ sum_{phi in PHI} sum_{i=1..6} w_i^{PHI,phi} * s_phi[i]^{p_i^{PHI,phi}} ]
                        * [ sum_{j=1..4} w'_j^{PHI} * f_j^{p'_j^{PHI}} ]

theta' = {w, p, w', p'}: per action (phase pair) 2 phases x 6 vars x (w, p) = 24
plus 4 flags x (w', p') = 8  ->  8 actions x 32 = 256 tunable parameters,
exactly the count derived in Section 4 of the paper.

Lemma 1 (monotonicity): because every s[i] >= 0,
    dg/ds[i] = w_i p_i s[i]^{p_i - 1} * (sum_j w'_j f_j^{p'_j})
has a sign that does not depend on s, so the policy argmax_PHI g is
"regulatable": each input can only ever push an action's priority one way.
"""
import numpy as np

from .env import ACTIONS, N_ACTIONS, N_FLAGS, N_VARS

EPS_S = 1e-2      # keeps s^p differentiable at s = 0
P_MIN, P_MAX = 0.05, 4.0


class Regulatable:
    def __init__(self):
        # Algorithm 1, line 4: initialise all weights to 1
        self.w = np.ones((N_ACTIONS, 2, N_VARS))
        self.p = np.ones((N_ACTIONS, 2, N_VARS))
        self.wf = np.ones((N_ACTIONS, N_FLAGS))
        self.pf = np.ones((N_ACTIONS, N_FLAGS))
        self.params = [self.w, self.p, self.wf, self.pf]
        self.idx = np.array(ACTIONS)  # (8, 2) phase indices per action

    @property
    def n_params(self):
        return sum(p.size for p in self.params)

    # ------------------------------------------------------------------
    def _terms(self, phase, flags):
        """phase: (B, 8, 6), flags: (B, 8 actions, 4).
        Returns x (B,A,2,6), x^p, A-term (B,A), B-term (B,A), f^p'."""
        x = phase[:, self.idx, :] + EPS_S              # (B, A, 2, 6)
        xp = x ** self.p[None]
        A = (self.w[None] * xp).sum((2, 3))           # (B, A)
        fp = np.where(flags > 0, flags ** self.pf[None], 0.0)
        Bt = (self.wf[None] * fp).sum(2)              # (B, A)
        return x, xp, A, Bt, fp

    def __call__(self, phase, flags):
        """G(s, .) for a batch: returns (B, 8)."""
        _, _, A, Bt, _ = self._terms(phase, flags)
        return A * Bt

    def grads(self, phase, flags, dG):
        """Back-propagate dL/dG (B, 8) to the 256 parameters."""
        x, xp, A, Bt, fp = self._terms(phase, flags)
        c = (dG * Bt)[:, :, None, None]
        gw = (c * xp).sum(0)
        gp = (c * self.w[None] * xp * np.log(x)).sum(0)
        gwf = ((dG * A)[:, :, None] * fp).sum(0)
        logf = np.where(flags > 0, np.log(np.maximum(flags, 1e-12)), 0.0)
        gpf = ((dG * A)[:, :, None] * self.wf[None] * fp * logf).sum(0)
        return [gw, gp, gwf, gpf]

    def clamp(self):
        np.clip(self.p, P_MIN, P_MAX, out=self.p)

    # ------------------------------------------------------------------
    def partials(self, phase, flags):
        """Analytic dG(s,a)/ds_phi[i] for each action's two phases: (B, A, 2, 6)."""
        x, _, _, Bt, _ = self._terms(phase, flags)
        return self.w[None] * self.p[None] * x ** (self.p[None] - 1) * Bt[:, :, None, None]

    def contributions(self, phase, flags, a):
        """Per-term breakdown of g(s, a) for explanation slides."""
        x, xp, A, Bt, _ = self._terms(phase[None], flags[None])
        return (self.w[a] * xp[0, a]) * Bt[0, a], Bt[0, a]
