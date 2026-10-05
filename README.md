# Learning an Interpretable Traffic Signal Control Policy: re-implementation

DRL Assignment 1, Problem III. Akash Garg (2025AG05687), Group 86, BITS Pilani WILP.

Paper: J. Ault, J. P. Hanna, G. Sharon, *Learning an Interpretable Traffic Signal Control Policy*, AAMAS 2020, [arXiv:1912.11023](https://arxiv.org/abs/1912.11023).
The authors' own code (SUMO based) is at https://github.com/jault/StateStreetSumo. This repo is my own re-implementation, written from the paper. It does not reuse their code.

## What is here

| File | What it does |
|---|---|
| `tsc/env.py` | Small single-intersection simulator. It has 8 phases (paper Fig. 1) and 8 phase-pair actions. Each phase has the paper's 6 state variables, and there are 4 clearance flags. Yellow/all-red is inserted by the environment. Reward is minus the waiting time. |
| `tsc/regulatable.py` | The designed precedence function (Eq. 1) with hand-written gradients. Also has the partial derivatives used to check Lemma 1. 256 parameters. |
| `tsc/agents.py` | Fixed-time and actuated (gap-out) baselines, plus DQN, DRQ, DRSQ and DRHQ following Algorithm 1. The line numbers of the algorithm are in the comments. |
| `tsc/nets.py` | numpy MLP (3×64, leaky ReLU), Adam, Huber loss |
| `run_experiments.py` | Runs every controller × demand level × seed, in parallel |
| `analyze.py` | Makes the figures in `figures/` and writes `results/summary.json` |

The only dependencies are numpy and matplotlib. There is no torch and no SUMO.

## How to run

```
python3 run_experiments.py --episodes 40 --eps-off 20 --seeds 5 --workers 11   # ~8 min on 11 cores
python3 analyze.py
```

For a quick check that does not overwrite the saved results:

```
python3 run_experiments.py --episodes 2 --seeds 1 --out demo_results
```

`results/` holds the results of the full run used in the slides. It includes the trained agents (`results/agents/*.npz`) that `analyze.py` reads.

## Setup compared with the paper

| | Paper | Here |
|---|---|---|
| Simulator | SUMO, State St & E 4500 S (Utah), 3 real days, 14 h each | Own point-queue simulator, 90-min episodes, 3 demand levels (0.70 / 0.82 / 0.94 veh/s) with a peak |
| Layout | 10 phases, 11 pairs, 352 params | 8 phases, 8 pairs, 256 params |
| Q-network | 3×64 with a 2×2 conv first layer | 3×64 dense |
| Same as paper | Adam 1e-3, Huber, replay 100k, batch 32, ε 0.05 → 0 after episode 20, γ 0.8 / 0.9 | |
| Seeds | 30 | 5 |
| Not done | | CMA-ES, PPO |

## Results (average delay in seconds, last 5 of 40 episodes, mean ± 95% CI over 5 seeds)

| | Low | Medium | High |
|---|---|---|---|
| Fixed-time | 71.7 ± 2.7 | 119.9 ± 4.6 | 178.2 ± 7.5 |
| Actuated | 23.0 ± 0.2 | 28.7 ± 0.6 | 40.8 ± 1.0 |
| DQN | 23.1 ± 0.4 | 38.0 ± 7.1 | 65.5 ± 11.5 |
| DRQ | ~2000 | ~2000 | ~2000 |
| DRSQ | 20.6 ± 0.5 | 31.4 ± 7.4 | 43.9 ± 6.8 |
| DRHQ | 20.5 ± 1.2 | 28.4 ± 3.9 | 39.6 ± 6.6 |

- Agrees with the paper:
  - DRQ fails.
  - DRHQ is the best regulatable variant and ends below actuated at all 3 levels (by 11.0%, 1.6% and 2.7%).
  - The learned G is monotonic in 100% of its inputs, against 24.7% for the DQN network.
- Does not agree with the paper:
  - My DQN is the slowest learner.
  - My agents need 19-38 episodes to beat actuated, while the paper's need 1. Part of this is episode length: a paper episode has about 16× more decisions than one of mine.
