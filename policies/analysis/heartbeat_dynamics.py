"""
Empirical heartbeat-age diagnostic.

Runs random rollouts under one or more (delay, dropout-mode) configs and
records the heartbeat-age values that *receivers* see for each
(sender, observer) pair, tagged with whether the sender was actually
alive at the time of the reading.

The goal is to give a visual answer to the question that drove the
v2 vs v3 pilot results::

    "Why does comm not help at D=5 but starts helping at D=30?"

Hypothesis (which this script confirms or refutes empirically):

    * At small D, the alive-sender age distribution is tightly bounded
      around D, while dead-sender ages climb past D within ~D steps and
      stay there forever -- so age alone is a near-perfect dropout
      detector outside a tiny ambiguity window.
    * At large D, the alive-sender age distribution spreads up to D+1,
      *overlapping* the dead-sender distribution for the entire post-
      death period; age alone is no longer a sufficient signal, and
      learned comm has a meaningful information gap to fill.

We do NOT need a trained policy for this -- the age dynamics are a
property of the wrapper plus the env physics, not of the policy. We use
a uniform-random policy over available actions so we get representative
trajectories quickly.

Output: one PNG per delay setting, with two side-by-side histograms
(alive-sender ages, dead-sender ages) and a stacked summary panel
showing the empirical "ambiguity window" -- the range of ages where
both distributions have non-negligible mass.

Usage
-----

::

    uv run python -m policies.analysis.heartbeat_dynamics \\
        --env rware-tiny-4ag-v2 \\
        --delays 5 30 \\
        --episodes 30 \\
        --max-steps 500 \\
        --dropout-window-start 200 --dropout-window-end 350 \\
        --out runs/figures/heartbeat_dynamics.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policies.wrappers.unified import (
    DropoutConfig,
    HeartbeatConfig,
    make_unified_env,
)


# ---------------------------------------------------------------------------
# Random-policy rollout
# ---------------------------------------------------------------------------


@dataclass
class AgePoint:
    """One recorded (sender, observer, t) heartbeat reading."""
    sender: int
    observer: int
    age: int
    sender_alive: bool   # ground truth at the moment of reading
    t: int               # step index since reset


def _random_joint_action(
    avail: np.ndarray, n_msg_tokens: int, rng: np.random.Generator
) -> np.ndarray:
    """
    avail has shape (n_agents, n_env_actions). Pick a uniformly random
    legal action per agent and a uniform random message token.
    """
    n_agents = avail.shape[0]
    out = np.zeros((n_agents, 2), dtype=np.int64)
    for i in range(n_agents):
        legal = np.where(avail[i] > 0)[0]
        if legal.size == 0:
            out[i, 0] = 0
        else:
            out[i, 0] = int(rng.choice(legal))
        out[i, 1] = int(rng.integers(0, max(n_msg_tokens, 1)))
    return out


def collect_age_points(
    env_id: str,
    *,
    delay: int,
    dropout_window: tuple[int, int] | None,
    n_episodes: int,
    max_steps: int,
    seed: int,
    n_msg_tokens: int = 8,
) -> list[AgePoint]:
    rng = np.random.default_rng(seed)
    hb_cfg = HeartbeatConfig(enabled=True, period=1, delay=delay)
    if dropout_window is not None:
        ws, we = dropout_window
        do_cfg = DropoutConfig(
            enabled=True, window_start=ws, window_end=we
        )
    else:
        do_cfg = DropoutConfig(enabled=False)

    env = make_unified_env(
        env_id,
        n_msg_tokens=n_msg_tokens,
        dropout_cfg=do_cfg,
        heartbeat_cfg=hb_cfg,
    )

    points: list[AgePoint] = []
    for ep in range(n_episodes):
        out = env.reset(seed=seed + ep)
        avail = out["available_actions"]
        info = out["info"]
        true_alive = info["debug_true_alive"]
        for t in range(max_steps):
            action = _random_joint_action(avail, n_msg_tokens, rng)
            out = env.step(action)
            avail = out["available_actions"]
            info = out["info"]
            ages = info["debug_heartbeat_age"]   # (n_agents, n_agents)
            true_alive = info["debug_true_alive"]
            # Record the off-diagonal entries: ages are (observer, sender)
            # in our wrapper convention. Skip self-readings (diagonal).
            n = ages.shape[0]
            for obs_i in range(n):
                for snd_j in range(n):
                    if obs_i == snd_j:
                        continue
                    points.append(AgePoint(
                        sender=snd_j, observer=obs_i,
                        age=int(ages[obs_i, snd_j]),
                        sender_alive=bool(true_alive[snd_j]),
                        t=t,
                    ))
            if out["done"]:
                break
    env.close()
    return points


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _stacked_histograms(
    ax,
    points: list[AgePoint],
    *,
    title: str,
    bin_max: int,
):
    alive_ages = np.array([p.age for p in points if p.sender_alive])
    dead_ages = np.array([p.age for p in points if not p.sender_alive])
    bins = np.arange(0, bin_max + 2)
    if alive_ages.size:
        ax.hist(
            alive_ages, bins=bins,
            density=True, alpha=0.55, color="#1f77b4",
            label=f"alive sender (n={alive_ages.size:,})",
        )
    if dead_ages.size:
        ax.hist(
            dead_ages, bins=bins,
            density=True, alpha=0.55, color="#d62728",
            label=f"dead sender (n={dead_ages.size:,})",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("heartbeat age (steps since last received ping)", fontsize=8)
    ax.set_ylabel("density (per age bin)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, loc="best")


def _ambiguity_summary(
    ax,
    delay_to_points: dict[int, list[AgePoint]],
    *,
    bin_max: int,
):
    """
    Overlap panel: for each delay, plot the *fraction of dead-sender age
    bins that are also visited by alive senders*. Bigger = more ambiguity
    available for comm to resolve.
    """
    bins = np.arange(0, bin_max + 2)
    for d, points in sorted(delay_to_points.items()):
        alive = np.array([p.age for p in points if p.sender_alive])
        dead = np.array([p.age for p in points if not p.sender_alive])
        if alive.size == 0 or dead.size == 0:
            continue
        h_alive, _ = np.histogram(alive, bins=bins, density=True)
        h_dead, _ = np.histogram(dead, bins=bins, density=True)
        # For each age bin, "ambiguity weight" = min(p_alive, p_dead).
        # Sum across bins gives total overlap mass (between 0 and 1) -- a
        # principled scalar version of "if you saw this age, how often
        # could it plausibly come from EITHER an alive or a dead sender?"
        overlap_per_bin = np.minimum(h_alive, h_dead)
        ax.plot(
            bins[:-1], overlap_per_bin,
            marker="o", markersize=3, linewidth=1.6,
            label=f"D={d}  total overlap mass = {overlap_per_bin.sum():.2f}",
        )
    ax.set_title(
        "ambiguity window: per-age min(p_alive, p_dead)\n"
        "[higher curve = more ages where comm could plausibly disambiguate]",
        fontsize=10,
    )
    ax.set_xlabel("heartbeat age", fontsize=8)
    ax.set_ylabel("min density across alive/dead", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, loc="best")


def make_figure(
    delay_to_points: dict[int, list[AgePoint]],
    *,
    out_path: Path,
    title: str,
    bin_max: int,
    max_age_clip: int = 32,
):
    delays = sorted(delay_to_points)
    n = len(delays)
    # Use GridSpec so the bottom panel actually spans the full figure width
    # regardless of how many delays we have on top.
    fig = plt.figure(figsize=(5.2 * max(n, 1), 8.5))
    gs = fig.add_gridspec(2, max(n, 1), height_ratios=[1.0, 1.1])
    top_axes = [fig.add_subplot(gs[0, j]) for j in range(max(n, 1))]
    bottom_ax = fig.add_subplot(gs[1, :])

    for j, d in enumerate(delays):
        ax = top_axes[j]
        _stacked_histograms(
            ax,
            delay_to_points[d],
            title=f"D={d}  |  alive vs dead heartbeat ages",
            bin_max=bin_max,
        )
        # Annotate the saturation cap; once dead ages cross it the wrapper
        # clips. The closer alive's plateau is to this cap, the smaller the
        # gap that distinguishes alive from dead.
        ax.axvline(
            max_age_clip, color="k", linewidth=0.7,
            linestyle="--", alpha=0.5,
        )
        ax.text(
            max_age_clip - 0.3, ax.get_ylim()[1] * 0.93,
            f"max_age_clip={max_age_clip}",
            rotation=90, va="top", ha="right", fontsize=7, color="k",
        )
        ax.axvline(
            d, color="#1f77b4", linewidth=0.6,
            linestyle=":", alpha=0.7,
        )
        ax.text(
            d + 0.3, ax.get_ylim()[1] * 0.93,
            f"alive plateau ≈ D = {d}\ngap to cap = {max_age_clip - d}",
            rotation=90, va="top", ha="left", fontsize=7, color="#1f77b4",
        )

    _ambiguity_summary(bottom_ax, delay_to_points, bin_max=bin_max)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[heartbeat] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Empirical heartbeat-age dynamics under random rollouts."
    )
    p.add_argument("--env", default="rware-tiny-4ag-v2")
    p.add_argument("--delays", nargs="+", type=int, default=[5, 30],
                   help="Heartbeat delays to compare.")
    p.add_argument("--episodes", type=int, default=30,
                   help="Number of episodes per delay setting.")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-msg-tokens", type=int, default=8,
                   help="Only used to construct the env; messages are random.")
    p.add_argument("--dropout-window-start", type=int, default=200)
    p.add_argument("--dropout-window-end",   type=int, default=350)
    p.add_argument("--no-dropout", action="store_true",
                   help="Disable dropout entirely (then there are no dead "
                        "senders to compare against; the figure becomes a "
                        "validation of the alive-only age distribution).")
    p.add_argument("--bin-max", type=int, default=None,
                   help="Truncate histogram x-axis at this age (default = "
                        "max(delays)+10).")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bin_max = int(args.bin_max) if args.bin_max else max(args.delays) + 10
    dropout_window = (
        None
        if args.no_dropout
        else (int(args.dropout_window_start), int(args.dropout_window_end))
    )
    delay_to_points: dict[int, list[AgePoint]] = {}
    for d in args.delays:
        print(f"[heartbeat] collecting D={d}, episodes={args.episodes} ...")
        delay_to_points[int(d)] = collect_age_points(
            env_id=args.env,
            delay=int(d),
            dropout_window=dropout_window,
            n_episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            n_msg_tokens=int(args.n_msg_tokens),
        )
        n_pts = len(delay_to_points[int(d)])
        n_dead = sum(1 for p in delay_to_points[int(d)] if not p.sender_alive)
        print(f"             collected {n_pts:,} (sender, observer) readings, "
              f"{n_dead:,} from dead senders")

    title = args.title or (
        f"heartbeat-age dynamics  |  env={args.env}  "
        f"|  dropout window=[{dropout_window[0]}, {dropout_window[1]})"
        if dropout_window else
        f"heartbeat-age dynamics  |  env={args.env}  |  no dropout"
    )
    make_figure(
        delay_to_points,
        out_path=Path(args.out),
        title=title,
        bin_max=bin_max,
    )


if __name__ == "__main__":
    main()
