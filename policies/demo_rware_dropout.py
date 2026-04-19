"""
Single deterministic rollout demonstrating the dropout moment.

Usage examples::

    # Default heuristic policy on the main env, with dropout at step 50 and a
    # 5-step heartbeat delay:
    uv run python -m policies.demo_rware_dropout \\
        --env rware-tiny-4ag-v2 --max-steps 200 \\
        --dropout --dropout-agent 0 --dropout-time 50 \\
        --heartbeat --heartbeat-delay 5

    # Same demo but using a trained MAPPO checkpoint:
    uv run python -m policies.demo_rware_dropout \\
        --env rware-tiny-4ag-v2 --checkpoint runs/.../ckpt.pt \\
        --max-steps 200 --dropout --dropout-time 50 --heartbeat --heartbeat-delay 5

The demo prints one line per step around the dropout moment, showing:

  * agent ids that are alive vs (presumed) dead
  * heartbeat ages (per agent)
  * the env action picked by each agent
  * the message token each agent emitted (mappo only)
  * cumulative team return so far

It is intentionally low-fi: presentation-grade in plain text, no GUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# Policy backends
# --------------------------------------------------------------------------


def _load_mappo(checkpoint: str, env, n_msg_tokens: int):
    """Return a callable ``act(state) -> joint_action`` backed by MAPPO."""
    import torch

    from policies.mappo import MAPPOConfig, MAPPOTrainer

    cfg = MAPPOConfig()
    trainer = MAPPOTrainer(
        obs_dim=env.spec.obs_dim,
        n_agents=env.spec.n_agents,
        n_env_actions=env.spec.n_env_actions,
        n_msg_tokens=n_msg_tokens,
        config=cfg,
    )
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trainer.load_state_dict(sd)
    print(f"[demo] loaded MAPPO checkpoint: {checkpoint}")

    def _act(state):
        out = trainer.act(
            obs=state["obs"],
            messages=state["messages"],
            alive=state["alive"],
            avail=state["available_actions"],
            deterministic=True,
        )
        return np.stack(
            [out["env_actions"], out["msg_actions"]], axis=-1
        ).astype(np.int64), out["msg_actions"]

    return _act


def _make_heuristic(env):
    """Return a callable backed by ``RwareHeuristicPolicy``."""
    from policies.baselines.rware_heuristic import (
        HeuristicConfig,
        RwareHeuristicPolicy,
    )

    policy = RwareHeuristicPolicy(
        n_agents=env.spec.n_agents,
        n_msg_tokens=1,
        config=HeuristicConfig(stale_threshold=4),
    )
    print("[demo] using RwareHeuristicPolicy (no checkpoint).")

    def _act(state):
        joint = policy.act(
            alive=state["alive"], info=state["info"], env=env
        )
        return joint, joint[:, 1]

    return _act


# --------------------------------------------------------------------------
# Pretty-print helpers
# --------------------------------------------------------------------------


def _reduce_hb_age(hb: np.ndarray | None, n_agents: int) -> np.ndarray:
    """
    Reduce ``info['debug_heartbeat_age']`` to a per-sender ``(n_agents,)`` vector.

    The wrapper publishes a 2D ``(recipient, sender)`` matrix; we min-reduce
    over recipients (excluding the diagonal, which is always 0) so each entry
    represents "the freshest report any teammate has of agent s".
    Falls back to a zero vector if the key is missing or has an unexpected
    shape.
    """
    if hb is None:
        return np.zeros(n_agents, dtype=np.int64)
    arr = np.asarray(hb, dtype=np.int64)
    if arr.ndim == 2 and arr.shape == (n_agents, n_agents):
        masked = arr.copy()
        np.fill_diagonal(masked, np.iinfo(np.int64).max)
        return masked.min(axis=0).astype(np.int64)
    if arr.ndim == 1 and arr.shape == (n_agents,):
        return arr
    return np.zeros(n_agents, dtype=np.int64)


def _fmt_int_arr(a: np.ndarray, width: int = 2) -> str:
    return "[" + " ".join(f"{int(x):{width}d}" for x in a) + "]"


def _fmt_bool_arr(a: np.ndarray) -> str:
    return "[" + " ".join("T" if bool(x) else "F" for x in a) + "]"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-rollout demo of the dropout regime.")
    p.add_argument("--env", default="rware-tiny-4ag-v2")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a MAPPOTrainer.state_dict() file. If omitted, the heuristic baseline is used.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-msg-tokens", type=int, default=8,
                   help="Must match what the checkpoint was trained with (mappo only).")
    p.add_argument("--no-comm", action="store_true",
                   help="Force n_msg_tokens=1 for the env (mirrors policies.train).")
    # Dropout / heartbeat knobs forwarded to the wrapper. Defaults intentionally
    # match `policies.train` so a demo invocation reproduces the same regime.
    p.add_argument("--heartbeat", action="store_true")
    p.add_argument("--heartbeat-period", type=int, default=1)
    p.add_argument("--heartbeat-delay", type=int, default=0)
    p.add_argument("--dropout", action="store_true")
    p.add_argument("--dropout-agent", type=int, default=None)
    p.add_argument("--dropout-time", type=int, default=None)
    p.add_argument("--dropout-window-start", type=int, default=None)
    p.add_argument("--dropout-window-end", type=int, default=None)
    p.add_argument("--print-window", type=int, default=8,
                   help="Print this many steps before+after the dropout moment.")
    p.add_argument("--render", action="store_true",
                   help="Open the RWARE pyglet window and visualise each step.")
    p.add_argument("--render-pause", type=float, default=0.15,
                   help="Sleep this many seconds between rendered steps (use a "
                        "larger value to slow the visualisation down).")
    return p.parse_args()


def _render_env(env) -> None:
    """Best-effort call into the underlying RWARE pyglet renderer."""
    try:
        inner = env.adapter._env.unwrapped
        inner.render(mode="human")
    except Exception as exc:
        print(f"[demo] render failed: {exc}")


def _make_env(args: argparse.Namespace, n_msg_tokens: int):
    """Build a ``UnifiedMARLEnv`` with dropout + heartbeat configs from CLI."""
    from policies.wrappers import DropoutConfig, HeartbeatConfig, make_unified_env

    dropout_cfg = DropoutConfig(
        enabled=bool(args.dropout),
        agent=args.dropout_agent,
        time=args.dropout_time,
        window_start=args.dropout_window_start,
        window_end=args.dropout_window_end,
    )
    heartbeat_cfg = HeartbeatConfig(
        enabled=bool(args.heartbeat),
        period=max(1, int(args.heartbeat_period or 1)),
        delay=max(0, int(args.heartbeat_delay or 0)),
    )
    return make_unified_env(
        args.env,
        n_msg_tokens=n_msg_tokens,
        dropout_cfg=dropout_cfg,
        heartbeat_cfg=heartbeat_cfg,
    )


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    n_msg_tokens = 1 if args.no_comm else int(args.n_msg_tokens)
    env = _make_env(args, n_msg_tokens=n_msg_tokens)
    print(f"[demo] env={args.env} spec={env.spec} n_msg_tokens={n_msg_tokens}")

    if args.checkpoint and Path(args.checkpoint).exists():
        act = _load_mappo(args.checkpoint, env, n_msg_tokens=n_msg_tokens)
        backend = "mappo"
    else:
        act = _make_heuristic(env)
        backend = "heuristic"

    state = env.reset(seed=args.seed)
    cum_return = 0.0
    dropout_step: int | None = None
    history: list[dict[str, Any]] = []  # ring buffer of all steps for printing

    print(
        f"[demo] backend={backend}  max_steps={args.max_steps}  "
        f"dropout_agent={args.dropout_agent}  dropout_time={args.dropout_time}"
    )
    if args.render:
        print(
            f"[demo] rendering with the RWARE pyglet window "
            f"(pause={args.render_pause}s/step). Close the window to exit early."
        )
        _render_env(env)

    import time as _time
    for t in range(args.max_steps):
        joint, msg_tokens = act(state)
        out = env.step(joint)
        cum_return += float(np.asarray(out["reward"]).sum())
        if args.render:
            _render_env(env)
            if args.render_pause > 0:
                _time.sleep(args.render_pause)

        info = out.get("info", {}) or {}
        true_alive = np.asarray(
            info.get("debug_true_alive", out["alive"]), dtype=bool
        )
        hb_age = _reduce_hb_age(info.get("debug_heartbeat_age"), env.spec.n_agents)
        fired = bool(info.get("debug_dropout_fired", False))
        if dropout_step is None and fired:
            dropout_step = t
        # Fallback: detect a non-fired-but-alive-mask drop. Skip when the
        # wrapper publishes the "no dropout yet" sentinel (-1).
        failed_idx = int(info.get("debug_failed_agent", -1))
        if (
            dropout_step is None
            and failed_idx >= 0
            and not bool(out["alive"][failed_idx])
        ):
            dropout_step = t

        history.append({
            "t": t,
            "alive": np.asarray(out["alive"], dtype=bool).copy(),
            "true_alive": true_alive.copy(),
            "hb": hb_age.copy(),
            "env_actions": joint[:, 0].copy(),
            "msg_tokens": np.asarray(msg_tokens, dtype=np.int64).copy(),
            "reward": np.asarray(out["reward"], dtype=np.float32).copy(),
            "cum_return": cum_return,
            "dropout_fired": fired,
            "failed_agent": info.get("debug_failed_agent"),
        })

        state = out
        if out["done"]:
            print(f"[demo] env returned done at t={t}")
            break

    env.close()

    # ---- Pretty print the dropout window ---------------------------------
    if dropout_step is None:
        print(
            "[demo] no dropout event detected via info['debug_dropout_fired']; "
            "printing the first + last few steps instead."
        )
        focus = list(range(min(args.print_window, len(history)))) + list(
            range(max(0, len(history) - args.print_window), len(history))
        )
        focus = sorted(set(focus))
    else:
        lo = max(0, dropout_step - args.print_window)
        hi = min(len(history), dropout_step + args.print_window + 1)
        focus = list(range(lo, hi))
        print(f"[demo] dropout fired at t={dropout_step}")
        failed = history[dropout_step].get("failed_agent")
        if failed is not None:
            print(f"[demo] failed agent id = {failed}")

    n = env.spec.n_agents
    bool_w = max(len("alive"), 2 * n + 1)
    int_w = max(len("hb_age"), 3 * n + 1)
    msg_w = max(len("msg_tok"), 3 * n + 1)
    print(
        f"\n  {'t':>3} {'alive':<{bool_w}} {'true_alive':<{bool_w}} "
        f"{'hb_age':<{int_w}} {'env_act':<{int_w}} {'msg_tok':<{msg_w}} "
        f"{'step_R':>7}  {'cum_R':>8}"
    )
    for idx in focus:
        h = history[idx]
        first_fire = h["dropout_fired"] and (
            idx == 0 or not history[idx - 1]["dropout_fired"]
        )
        marker = "  <-- DROPOUT" if first_fire else ""
        print(
            f"  {h['t']:>3d} "
            f"{_fmt_bool_arr(h['alive']):<{bool_w}} "
            f"{_fmt_bool_arr(h['true_alive']):<{bool_w}} "
            f"{_fmt_int_arr(h['hb']):<{int_w}} "
            f"{_fmt_int_arr(h['env_actions']):<{int_w}} "
            f"{_fmt_int_arr(h['msg_tokens']):<{msg_w}} "
            f"{float(h['reward'].sum()):7.2f}  {h['cum_return']:8.2f}{marker}"
        )

    # ---- Summary --------------------------------------------------------
    print(
        f"\n[demo] total steps={len(history)}  cum_return={cum_return:.3f}  "
        f"dropout_step={dropout_step}"
    )
    if dropout_step is not None:
        post = [h["reward"].sum() for h in history[dropout_step + 1:]]
        if post:
            print(
                f"[demo] post-dropout reward sum = {float(np.sum(post)):.3f} "
                f"over {len(post)} steps "
                f"(team kept earning -> recovery)"
            )


if __name__ == "__main__":
    main()
