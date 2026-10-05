"""Run every controller on every demand profile over several random seeds.

Mirrors Section 6 of the paper: average delay per training episode,
95 % confidence interval over seeds; epsilon 0.05 -> 0 after EPS_OFF episodes;
gamma 0.8 (low / medium) and 0.9 (high).

Usage:  python3 run_experiments.py [--episodes 30] [--seeds 5] [--workers 10] [--out results]
Output: <out>/results.json, <out>/agents/*.npz, <out>/train_log.txt
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tsc.agents import Actuated, DQNFamily, FixedTime
from tsc.env import DEMAND_LEVELS, IntersectionEnv

HERE = os.path.dirname(os.path.abspath(__file__))
LEARNERS = ["DQN", "DRQ", "DRSQ", "DRHQ"]
GAMMA = {"Low": 0.8, "Medium": 0.8, "High": 0.9}


def run_episode(env, agent, seed, learn):
    o = env.reset(seed=seed)
    agent.reset()
    done = False
    while not done:
        a = agent.act(o, env)
        o2, r, done = env.step(a)
        if learn:
            agent.observe(o, a, r, o2, done)
        o = o2
    return env.average_delay()


def job(args):
    method, demand, seed, episodes, eps_off, out = args
    env = IntersectionEnv(demand)
    t0 = time.time()
    curve = []
    if method in LEARNERS:
        agent = DQNFamily(method, n_flat=56, gamma=GAMMA[demand], seed=seed)
        for ep in range(episodes):
            agent.eps = 0.05 if ep < eps_off else 0.0
            curve.append(run_episode(env, agent, 1000 * seed + ep, learn=True))
        np.savez(os.path.join(out, "agents", f"{method}_{demand}_{seed}.npz"),
                 w=agent.G.w, p=agent.G.p, wf=agent.G.wf, pf=agent.G.pf,
                 **{f"qW{i}": W for i, W in enumerate(agent.q.W)},
                 **{f"qb{i}": b for i, b in enumerate(agent.q.b)},
                 phase=agent.D.phase[:agent.D.n:50], flags=agent.D.flags[:agent.D.n:50],
                 flat=agent.D.flat[:agent.D.n:50])
    else:
        agent = FixedTime() if method == "Fixed-time" else Actuated()
        for ep in range(episodes):
            curve.append(run_episode(env, agent, 1000 * seed + ep, learn=False))
    line = (f"[{method:10s}|{demand:6s}|seed {seed}] "
            f"ep1={curve[0]:6.1f}s  last5={np.mean(curve[-5:]):6.1f}s  "
            f"({time.time() - t0:5.0f}s wall)")
    print(line, flush=True)
    return method, demand, seed, curve, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--eps-off", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="results", help="output folder (inside code/)")
    a = ap.parse_args()
    OUT = os.path.join(HERE, a.out)
    os.makedirs(os.path.join(OUT, "agents"), exist_ok=True)
    methods = ["Fixed-time", "Actuated", *LEARNERS]
    jobs = [(m, d, s, a.episodes, a.eps_off, OUT)
            for m in methods for d in DEMAND_LEVELS for s in range(a.seeds)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        out = pool.map(job, jobs)
    res = {}
    for m, d, s, curve, _ in out:
        res.setdefault(m, {}).setdefault(d, []).append(curve)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"config": vars(a), "results": res}, f)
    with open(os.path.join(OUT, "train_log.txt"), "w") as f:
        f.write("\n".join(o[4] for o in out))
        f.write(f"\nTotal wall time: {time.time() - t0:.0f}s\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
