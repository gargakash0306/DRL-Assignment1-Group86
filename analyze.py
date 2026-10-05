"""Turn results/ into figures + summary numbers used on the slides.

  figures/fig_delay_curves.png    -- my version of the paper's Figure 4
  figures/fig_final_bars.png      -- final delay per controller / demand
  figures/fig_monotonicity.png    -- Lemma 1 check: G vs DQN sign consistency
  figures/fig_weights.png         -- learned DRHQ weights (my version of Figure 2)
  figures/fig_explain.png         -- why DRHQ picked an action in a real state
  results/summary.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsc.env import ACTION_NAMES, ACTIONS, PHASE_NAMES, VAR_NAMES, FLAG_NAMES
from tsc.nets import MLP
from tsc.regulatable import Regulatable

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DEMANDS = ["Low", "Medium", "High"]
COLORS = {"Actuated": "#1f77b4", "Fixed-time": "#7f7f7f", "DQN": "#ff7f0e",
          "DRQ": "#8c564b", "DRSQ": "#9467bd", "DRHQ": "#2ca02c"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

data = json.load(open(os.path.join(RES, "results.json")))
R = {m: {d: np.array(v) for d, v in dd.items()} for m, dd in data["results"].items()}
EPS_OFF = data["config"]["eps_off"]
n_seeds = data["config"]["seeds"]


def ci95(x):  # x: (seeds, episodes)
    return 1.96 * x.std(0, ddof=1) / np.sqrt(x.shape[0])


# ---------------------------------------------------------------- Figure 4
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, d in zip(axes, DEMANDS):
    act = R["Actuated"][d].mean()
    for m in ["DQN", "DRSQ", "DRHQ"]:
        x = R[m][d]
        ep = np.arange(1, x.shape[1] + 1)
        mu = x.mean(0)
        ax.plot(ep, mu, color=COLORS[m], label=m, lw=2)
        lo = np.maximum(mu - ci95(x), 1)
        ax.fill_between(ep, lo, mu + ci95(x), color=COLORS[m], alpha=0.12)
    ax.axhline(act, color=COLORS["Actuated"], lw=2, ls="--", label="Actuated")
    ax.axvline(EPS_OFF + 0.5, color="k", lw=0.8, ls=":")
    ax.set_title(f"{d} demand")
    ax.set_xlabel("Training episode (90-min traffic period)")
    ax.set_yscale("log")
    ax.set_ylim(act * 0.6, 1000)
    ax.set_yticks([20, 30, 50, 100, 200, 500, 1000], ["20", "30", "50", "100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.text(EPS_OFF + 1, 700, "epsilon = 0", fontsize=9)
axes[0].set_ylabel("Average delay per vehicle (s)")
axes[0].legend(loc="upper right", fontsize=9)
fig.suptitle(f"Average delay per episode, mean and 95% CI over {n_seeds} seeds (log scale). "
             "DRQ stays above 1000 s so it is left out, same as the paper.", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_delay_curves.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- summary numbers
summary = {"final": {}, "first_ep_beating_actuated": {}, "improvement_vs_actuated_pct": {}}
for d in DEMANDS:
    act = R["Actuated"][d].mean()
    for m in R:
        x = R[m][d]
        final = x[:, -5:].mean(1)  # per seed, last 5 episodes
        summary["final"].setdefault(m, {})[d] = [float(final.mean()),
                                                float(1.96 * final.std(ddof=1) / np.sqrt(len(final)))]
        if m in ("DQN", "DRQ", "DRSQ", "DRHQ"):
            mu = x.mean(0)
            hit = np.where(mu <= act)[0]
            summary["first_ep_beating_actuated"].setdefault(m, {})[d] = int(hit[0] + 1) if len(hit) else None
            summary["improvement_vs_actuated_pct"].setdefault(m, {})[d] = float(100 * (act - final.mean()) / act)

# ---------------------------------------------------------------- final bars
fig, ax = plt.subplots(figsize=(10, 4.2))
ms = ["Actuated", "DQN", "DRSQ", "DRHQ"]
wbar = 0.2
for i, m in enumerate(ms):
    mu = [summary["final"][m][d][0] for d in DEMANDS]
    er = [summary["final"][m][d][1] for d in DEMANDS]
    xs = np.arange(3) + (i - 1.5) * wbar
    ax.bar(xs, mu, wbar, yerr=er, color=COLORS[m], label=m, capsize=3)
    for x_, v in zip(xs, mu):
        ax.text(x_, v + 1, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(3), [f"{d} demand" for d in DEMANDS])
ax.set_ylabel("Avg delay, last 5 episodes (s)")
ax.legend(ncol=4, fontsize=9, loc="upper left")
ax.set_title("Average delay over the last 5 of 40 episodes (lower is better)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_final_bars.png"), dpi=150)
plt.close(fig)


# ---------------------------------------------------------------- load agents
def load(method, demand, seed):
    z = np.load(os.path.join(RES, "agents", f"{method}_{demand}_{seed}.npz"))
    G = Regulatable()
    G.w[...], G.p[...], G.wf[...], G.pf[...] = z["w"], z["p"], z["wf"], z["pf"]
    q = MLP(56, 8)
    for i in range(len(q.W)):
        q.W[i][...], q.b[i][...] = z[f"qW{i}"], z[f"qb{i}"]
    return G, q, z["phase"], z["flags"], z["flat"]


# ---------------------------------------------------------------- Lemma 1 check
mono = {"DRHQ G (regulatable)": [], "DQN Q-network (black box)": []}
for d in DEMANDS:
    for s in range(n_seeds):
        G, _, ph, fl, _ = load("DRHQ", d, s)
        P = G.partials(ph, fl)                               # (B, A, 2, 6)
        consistent = (np.all(P >= 0, 0) | np.all(P <= 0, 0)).mean()
        mono["DRHQ G (regulatable)"].append(100 * consistent)
        _, q, ph, fl, flat = load("DQN", d, s)
        q.forward(flat, keep=True)
        ok = []
        for a, (m1, m2) in enumerate(ACTIONS):
            dout = np.zeros((len(flat), 8))
            dout[:, a] = 1
            q.backward(dout)
            g = q.input_grad[:, :48].reshape(-1, 8, 6)[:, [m1, m2], :]  # inputs of served phases
            tol = 1e-6
            ok.append((np.all(g >= -tol, 0) | np.all(g <= tol, 0)))
        mono["DQN Q-network (black box)"].append(100 * np.mean(ok))
summary["monotonic_pct"] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in mono.items()}

fig, ax = plt.subplots(figsize=(9, 3.4))
names = list(mono)
vals = [np.mean(mono[n]) for n in names]
ax.barh(names, vals, color=["#2ca02c", "#ff7f0e"])
for i, v in enumerate(vals):
    ax.text(v - 2 if v > 90 else v + 1, i, f"{v:.1f} %", va="center", ha="right" if v > 90 else "left", fontsize=12, fontweight="bold", color="white" if v > 90 else "black")
ax.set_xlim(0, 105)
ax.set_xlabel("% of inputs whose effect on the action score never changes sign")
ax.set_title(f"Monotonicity check, {len(mono[names[0]])} trained agents each")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_monotonicity.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- learned weights
G, q, ph, fl, flat = load("DRHQ", "Medium", 0)
rows, W, Pm = [], [], []
for a, (m1, m2) in enumerate(ACTIONS):
    for k, m in enumerate((m1, m2)):
        rows.append(f"{ACTION_NAMES[a]}: {PHASE_NAMES[m]}")
        W.append(G.w[a, k])
        Pm.append(G.p[a, k])
W, Pm = np.array(W), np.array(Pm)
fig, ax = plt.subplots(figsize=(11, 8))
lim = np.abs(W).max()
im = ax.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        ax.text(j, i, f"{W[i, j]:.2f}\n(p={Pm[i, j]:.2f})", ha="center", va="center", fontsize=7)
ax.set_xticks(range(6), VAR_NAMES, rotation=15)
ax.set_yticks(range(len(rows)), rows, fontsize=8)
for i in range(1, 8):
    ax.axhline(2 * i - 0.5, color="k", lw=0.8)
fig.colorbar(im, ax=ax, label="weight w (sign = direction of influence)")
ax.set_title("Learned weights w (exponent p in brackets), DRHQ, medium demand, seed 0")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_weights.png"), dpi=150)
plt.close(fig)
summary["clearance_weights_medium_seed0"] = {ACTION_NAMES[a]: dict(zip(FLAG_NAMES, map(float, G.wf[a])))
                                            for a in range(8)}
summary["n_params"] = G.n_params

# ---------------------------------------------------------------- explain one decision
Gv = G(ph, fl)
gap = np.sort(Gv, 1)
idx = int(np.argmax(np.where(ph.sum((1, 2)) > 1.5, gap[:, -1] - gap[:, -2], -1)))
best, second = np.argsort(Gv[idx])[::-1][:2]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, a in zip(axes, (best, second)):
    contrib, flagterm = G.contributions(ph[idx], fl[idx], a)
    labels, vals = [], []
    for k, m in enumerate(ACTIONS[a]):
        for i in range(6):
            labels.append(f"{PHASE_NAMES[m]}: {VAR_NAMES[i]}")
            vals.append(contrib[k, i])
    order = np.argsort(np.abs(vals))[::-1][:6]
    ax.barh([labels[o] for o in order][::-1], [vals[o] for o in order][::-1],
            color=["#2ca02c" if vals[o] > 0 else "#d62728" for o in order][::-1])
    ax.set_title(f"{'CHOSEN' if a == best else 'Runner-up'}: {ACTION_NAMES[a]}  g = {Gv[idx, a]:.2f}\n"
                 f"(clearance multiplier {flagterm:.2f})", fontsize=10)
    ax.set_xlabel("contribution w * s^p * clearance term")
fig.suptitle("One decision from the replay buffer: largest terms of g(s, PHI) for the top two actions")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_explain.png"), dpi=150)
plt.close(fig)

json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps(summary["final"], indent=1))

