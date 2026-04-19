"""
Entry point for training Flat MAPPO + RIAL-style communication on either the
RWARE or MultiGrid family of environments exposed through
``UnifiedMARLEnv``.

Usage (from the CS2860 project root):

    uv run python -m policies.train --env rware-tiny-2ag-v2 --updates 50
    uv run python -m policies.train --env MultiGrid-Empty-8x8-v0 --n-agents 2 \
            --updates 100 --rollout 512

See ``--help`` for the full list of knobs.

Logging:
  By default each run writes ``metrics.csv`` and TensorBoard events under
  ``runs/{run_name}/``. Pass ``--run-name`` to control the directory name,
  ``--log-dir`` to change the root, or ``--no-log`` to disable both.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .logger import RunLogger
from .mappo import MAPPOConfig, MAPPOTrainer, Runner, RolloutBuffer
from .wrappers import make_unified_env
from .wrappers.unified import DropoutConfig
from .wrappers.heartbeat import HeartbeatConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flat MAPPO + communication baseline")
    p.add_argument("--env", type=str, required=True, help="Registered env id")
    p.add_argument("--n-agents", type=int, default=None,
                   help="Number of agents (MultiGrid only; rware bakes it in).")
    p.add_argument("--n-msg-tokens", type=int, default=8)
    p.add_argument("--no-comm", action="store_true",
                   help="Ablation: force n_msg_tokens=1 (no usable comm channel).")
    p.add_argument("--rollout", type=int, default=256, help="Steps per rollout")
    p.add_argument("--updates", type=int, default=100, help="Number of PPO updates")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--save", type=str, default=None, help="Path to save the final checkpoint")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subdirectory under --log-dir. Defaults to a timestamp.")
    p.add_argument("--no-log", action="store_true",
                   help="Disable CSV + TensorBoard logging entirely.")

    # --- Ambiguity mechanism: controlled permanent dropout ---
    p.add_argument("--dropout", action="store_true",
                   help="Enable permanent teammate dropout mid-episode.")
    p.add_argument("--dropout-agent", type=int, default=None,
                   help="Which agent index to drop (fixed mode).")
    p.add_argument("--dropout-time", type=int, default=None,
                   help="Episode step at which dropout fires (fixed mode).")
    p.add_argument("--dropout-window-start", type=int, default=None,
                   help="Window mode: inclusive start of uniform-random dropout step.")
    p.add_argument("--dropout-window-end", type=int, default=None,
                   help="Window mode: exclusive end of uniform-random dropout step.")

    # --- Ambiguity mechanism: delayed heartbeats ---
    p.add_argument("--heartbeat", action="store_true",
                   help="Enable delayed-heartbeat freshness features in obs.")
    p.add_argument("--heartbeat-period", type=int, default=1,
                   help="Emit a heartbeat every N steps (>=1).")
    p.add_argument("--heartbeat-delay", type=int, default=0,
                   help="Heartbeats arrive N steps after they were produced.")

    # --- Optional reward shaping (RWARE only) ---
    p.add_argument("--shape-rewards", action="store_true",
                   help="(RWARE only) add a small bonus on requested-shelf "
                        "pickup events to densify the otherwise sparse "
                        "delivery-only reward.")
    p.add_argument("--pickup-bonus", type=float, default=0.5,
                   help="Per-agent reward added on a False->True pickup of a "
                        "currently-requested shelf when --shape-rewards is on.")
    p.add_argument("--step-penalty", type=float, default=0.0,
                   help="Per-step reward subtracted from every agent when "
                        "--shape-rewards is on (0 disables).")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    n_msg_tokens = 1 if args.no_comm else args.n_msg_tokens

    dropout_cfg = DropoutConfig(
        enabled=args.dropout,
        agent=args.dropout_agent,
        time=args.dropout_time,
        window_start=args.dropout_window_start,
        window_end=args.dropout_window_end,
    )
    heartbeat_cfg = HeartbeatConfig(
        enabled=args.heartbeat,
        period=max(1, args.heartbeat_period),
        delay=max(0, args.heartbeat_delay),
    )

    adapter_kwargs: dict = {}
    if args.env.startswith("rware-") and args.shape_rewards:
        adapter_kwargs.update(
            shape_rewards=True,
            pickup_bonus=args.pickup_bonus,
            step_penalty=args.step_penalty,
        )

    env = make_unified_env(
        args.env,
        n_agents=args.n_agents,
        n_msg_tokens=n_msg_tokens,
        dropout_cfg=dropout_cfg,
        heartbeat_cfg=heartbeat_cfg,
        **adapter_kwargs,
    )
    spec = env.spec
    print(f"[env] {spec}  (n_msg_tokens={n_msg_tokens})")

    cfg = MAPPOConfig(
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        update_epochs=args.update_epochs,
        minibatches=args.minibatches,
        hidden=args.hidden,
        depth=args.depth,
        device=args.device,
    )
    trainer = MAPPOTrainer(
        obs_dim=spec.obs_dim,
        n_agents=spec.n_agents,
        n_env_actions=spec.n_env_actions,
        n_msg_tokens=n_msg_tokens,
        config=cfg,
    )
    buffer = RolloutBuffer(
        rollout_len=args.rollout,
        n_agents=spec.n_agents,
        obs_dim=spec.obs_dim,
        n_env_actions=spec.n_env_actions,
        n_msg_tokens=n_msg_tokens,
    )
    runner = Runner(env, trainer)

    print(f"[cfg] {asdict(cfg)}")

    logger: RunLogger | None = None
    if not args.no_log:
        run_config = {
            **vars(args),
            "n_msg_tokens_effective": n_msg_tokens,
            "env_spec": repr(spec),
        }
        logger = RunLogger(
            log_dir=args.log_dir,
            run_name=args.run_name,
            config=run_config,
            use_tensorboard=True,
        )
        print(f"[log] writing to {logger.run_dir}")

    t0 = time.time()
    total_steps = 0
    try:
        for update in range(1, args.updates + 1):
            rollout_stats = runner.collect(buffer, seed=args.seed if update == 1 else None)
            total_steps += buffer.ptr
            losses = trainer.update(buffer)

            elapsed = time.time() - t0
            sps = total_steps / max(elapsed, 1e-6)
            train_summary = rollout_stats.summary()

            scalars = {
                "train/ep_return_mean": train_summary.get("ep_return_mean", 0.0),
                "train/ep_length_mean": train_summary.get("ep_length_mean", 0.0),
                "train/n_episodes": train_summary.get("n_episodes", 0),
                "loss/policy": losses["policy_loss"],
                "loss/value": losses["value_loss"],
                "loss/entropy": losses["entropy"],
                "loss/approx_kl": losses["approx_kl"],
                "loss/clip_frac": losses["clip_frac"],
                "time/elapsed_s": elapsed,
                "time/sps": sps,
                "time/update": update,
            }

            line = (
                f"[upd {update:4d} | steps {total_steps:7d} | t {elapsed:6.1f}s | "
                f"sps {sps:5.0f}] "
                f"ret_mean={scalars['train/ep_return_mean']:.3f} "
                f"ep_len={scalars['train/ep_length_mean']:.1f} "
                f"pi_loss={losses['policy_loss']:.3f} v_loss={losses['value_loss']:.3f} "
                f"ent={losses['entropy']:.3f} kl={losses['approx_kl']:.4f}"
            )
            print(line, flush=True)

            if update % args.eval_every == 0:
                eval_stats = runner.evaluate(
                    n_episodes=args.eval_episodes, seed=args.seed + update
                )
                eval_summary = eval_stats.summary()
                scalars["eval/ep_return_mean"] = eval_summary["ep_return_mean"]
                scalars["eval/ep_length_mean"] = eval_summary["ep_length_mean"]
                print(
                    f"        [eval] ret_mean={eval_summary['ep_return_mean']:.3f} "
                    f"ep_len={eval_summary['ep_length_mean']:.1f}",
                    flush=True,
                )

            if logger is not None:
                logger.log_scalars(total_steps, scalars)

        if args.save is not None:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.state_dict(), args.save)
            print(f"[ckpt] saved to {args.save}")
    finally:
        if logger is not None:
            logger.close()
        env.close()


if __name__ == "__main__":
    main()
