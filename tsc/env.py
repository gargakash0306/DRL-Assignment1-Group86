"""Single 4-way signalised intersection simulator (simplified stand-in for SUMO).

Reproduces the problem definition of Ault, Hanna & Sharon (AAMAS 2020), Section 3:
  * 8 NEMA phases (Figure 1 of the paper), phi1 = Eastbound left.
  * Action = one of the 8 non-conflicting phase pairs
        {(1,5),(1,6),(2,5),(2,6)} U {(3,7),(3,8),(4,7),(4,8)}
    held for a minimum phase length (5 s here, 3 s in the paper).
  * Yellow / all-red clearance is inserted automatically by the environment
    (constraints are part of the transition function, as in the paper).
  * Inputs per phase (Section 4): stopped vehicles, approaching vehicles,
    cumulative stopped time, average stopped time, average queue length
    (stopped / lanes), average speed of approaching vehicles.
  * Reward = negative accumulated waiting (vehicle-seconds) during the step.

Simplifications vs. the paper (stated honestly on the slides):
  * point-queue model with Poisson arrivals instead of SUMO car-following;
  * synthetic 90-minute demand profile (off-peak -> peak -> off-peak) whose
    shape mimics the UDOT data, instead of 14 h of real UDOT counts;
  * no permissive-left phases, so clearance flag f3 never fires.
"""
from collections import deque

import numpy as np

# ---- intersection layout -------------------------------------------------
# index: 0..7  ==  NEMA phi1..phi8
PHASE_NAMES = ["EB-Left", "WB-Thru", "SB-Left", "NB-Thru",
               "WB-Left", "EB-Thru", "NB-Left", "SB-Thru"]
LANES = np.array([1, 2, 1, 2, 1, 2, 1, 2])
SAT_FLOW = 0.5  # veh / s / lane  (1800 veh/h/lane)

# 8 non-conflicting phase pairs (0-indexed)
ACTIONS = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 6), (2, 7), (3, 6), (3, 7)]
ACTION_NAMES = [f"phi{a + 1}+phi{b + 1}" for a, b in ACTIONS]
N_ACTIONS = len(ACTIONS)
N_PHASES = 8
N_VARS = 6
VAR_NAMES = ["Stopped", "Approaching", "Cum. stopped time",
             "Avg stopped time", "Avg queue / lane", "Approach speed"]
N_FLAGS = 4
FLAG_NAMES = ["Full clearance", "Partial clearance",
              "Permissive clearance", "No clearance"]

# feature normalisation so every input is O(1) and non-negative
_NORM = np.array([20.0, 20.0, 1000.0, 60.0, 10.0, 1.0])

# share of total demand per movement (major EW arterial, minor NS road)
_SHARE = np.array([0.07, 0.19, 0.07, 0.14, 0.07, 0.19, 0.07, 0.14])
_SHARE = _SHARE / _SHARE.sum()

DEMAND_LEVELS = {"Low": 0.70, "Medium": 0.82, "High": 0.94}  # mean veh/s

STEP = 5             # seconds per decision (minimum phase length)
TRAVEL = 20          # seconds from spawn to stop line (approach zone)
YELLOW_ALLRED = 4    # lost green when every phase changes (full clearance)
PARTIAL_LOST = 3     # lost green for newly-started phase (partial clearance)
HORIZON = 5400       # 90-minute episode


def demand_profile(t, horizon=HORIZON):
    """Multiplier on mean demand: off-peak shoulders with a peak in the middle,
    plus a directional tidal swing (EB heavier early, WB heavier late)."""
    x = t / horizon
    peak = 0.75 + 0.55 * np.exp(-((x - 0.5) / 0.18) ** 2)
    tide = 0.25 * np.cos(2 * np.pi * x)
    m = np.full(8, peak)
    m[[0, 5]] *= 1 + tide   # EB movements
    m[[1, 4]] *= 1 - tide   # WB movements
    return m


class IntersectionEnv:
    def __init__(self, demand="Medium", seed=0, horizon=HORIZON):
        self.mean_rate = DEMAND_LEVELS[demand]
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.action = 3  # start with EW through (phi2+phi6)
        self.green_time = 0
        self.approach = [deque() for _ in range(8)]  # stop-line arrival times
        self.queue = [deque() for _ in range(8)]     # time joined queue
        self.acc = np.zeros(8)
        self.delays_sum = 0.0
        self.n_done = 0
        self.n_arrived = 0
        return self.observe()

    # ------------------------------------------------------------------
    def clearance_flags(self, new_action, cur_action=None):
        """One-hot clearance case f1..f4 for switching cur -> new (Section 4)."""
        cur = self.action if cur_action is None else cur_action
        f = np.zeros(N_FLAGS)
        if new_action == cur:
            f[3] = 1          # no clearance
        elif set(ACTIONS[new_action]) & set(ACTIONS[cur]):
            f[1] = 1          # partial clearance (one phase continues)
        else:
            f[0] = 1          # full clearance (all red)
        return f

    def all_flags(self):
        return np.stack([self.clearance_flags(a) for a in range(N_ACTIONS)])

    # ------------------------------------------------------------------
    def phase_state(self):
        """(8 phases x 6 vars) non-negative, normalised state variables."""
        s = np.zeros((8, N_VARS))
        for m in range(8):
            q = self.queue[m]
            nq = len(q)
            cum = sum(self.t - x for x in q)
            s[m, 0] = nq
            s[m, 1] = len(self.approach[m])
            s[m, 2] = cum
            s[m, 3] = cum / nq if nq else 0.0
            s[m, 4] = nq / LANES[m]
            # approaching vehicles slow down as the queue they join grows
            s[m, 5] = (1.0 / (1.0 + 0.1 * nq / LANES[m])) if self.approach[m] else 0.0
        return s / _NORM

    def observe(self):
        """Returns dict with the phase-state matrix, clearance flags for every
        candidate action, and the flat DQN input vector."""
        ps = self.phase_state()
        flags = self.all_flags()
        onehot = np.zeros(N_ACTIONS)
        onehot[self.action] = 1
        flat = np.concatenate([ps.ravel(), onehot])
        return {"phase": ps, "flags": flags, "flat": flat}

    # ------------------------------------------------------------------
    def step(self, action):
        prev = self.action
        f = self.clearance_flags(action)
        new_green = set(ACTIONS[action])
        lost = np.zeros(8)
        if f[0]:
            lost[:] = YELLOW_ALLRED
        elif f[1]:
            for m in new_green - set(ACTIONS[prev]):
                lost[m] = PARTIAL_LOST
        self.green_time = self.green_time + STEP if action == prev else 0
        self.action = action

        waiting = 0.0
        for k in range(STEP):
            t = self.t
            rates = self.mean_rate * _SHARE * demand_profile(t, self.horizon)
            arrivals = self.rng.poisson(rates)
            for m in range(8):
                for _ in range(arrivals[m]):
                    self.approach[m].append(t + TRAVEL)
                ap, q = self.approach[m], self.queue[m]
                while ap and ap[0] <= t:
                    ap.popleft()
                    q.append(t)
                    self.n_arrived += 1
                if m in new_green and k >= lost[m]:
                    self.acc[m] += LANES[m] * SAT_FLOW
                    while self.acc[m] >= 1 and q:
                        self.delays_sum += t - q.popleft()
                        self.n_done += 1
                        self.acc[m] -= 1
                    if not q:
                        self.acc[m] = min(self.acc[m], 1.0)
                else:
                    self.acc[m] = 0.0
                waiting += len(q)
            self.t += 1
        reward = -waiting / 100.0
        done = self.t >= self.horizon
        return self.observe(), reward, done

    # ------------------------------------------------------------------
    def average_delay(self):
        """Mean delay (s) per vehicle that reached the stop line; vehicles still
        queued at the end are charged the delay accumulated so far."""
        remaining = sum(self.t - x for q in self.queue for x in q)
        return (self.delays_sum + remaining) / max(self.n_arrived, 1)
