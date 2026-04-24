"""
CLI entry point for the paper-faithful hierarchical baseline.

    uv run python -m policies.hierarchical.train \
        --env rware-tiny-2ag-easy-v2 --episodes 50 --max-steps 500

    uv run python -m policies.hierarchical.train \
        --env rware-tiny-4ag-v2 --episodes 200 --max-steps 500 --comm

Two variants:
  * default:            Cooperative-HRL (no explicit inter-agent comm).
                        Each agent conditions its Q on its own stale
                        estimate of teammates' selected subtasks.
  * --comm:             COM-Cooperative-HRL. At every subtask boundary
                        an agent broadcasts its new cooperative
                        subtask, so every teammate sees an up-to-date
                        assignment vector.

Compatible with the same ``--dropout`` / ``--heartbeat`` flags as
``policies.train`` so side-by-side matrix experiments are possible.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from policies.logger import RunLogger
from policies.wrappers import DropoutConfig, HeartbeatConfig, make_unified_env

from .controller import ControllerConfig, HierarchicalController
from .learner import CooperativeSMDPQLearner, QLearnerConfig
from .runner import HRLRunner
from .subtasks import MAX_SLOTS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Hierarchical Multi-Agent RL baseline (Ghavamzadeh et al., "
            "Coop-HRL / COM-Coop-HRL) for RWARE."
        )
    )
    p.add_argument("--env", type=str, default="rware-tiny-4ag-v2")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    # Algorithm-level knobs.
    p.add_argument("--comm", action="store_true",
                   help="Enable COM-Cooperative-HRL: broadcast new "
                        "cooperative subtasks at every boundary.")
    p.add_argument("--no-cooperative", action="store_true",
                   help="Ablation: drop teammate conditioning from Q key "
                        "(reduces to independent hierarchical Q-learning).")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes", type=int, default=200)
    p.add_argument("--optimistic-init", type=float, default=0.0)
    p.add_argument("--max-option-steps", type=int, default=60)
    p.add_argument("--stale-threshold", type=int, default=4)
    p.add_argument("--request-queue-size", type=int, default=None,
                   help="Override the inferred request_queue_size; "
                        "defaults to the env's configured value.")
    p.add_argument("--save-q", type=str, default=None,
                   help="Path to save the final Q-table (JSON).")

    # Wrapper-side mechanism knobs (mirror policies.train / heuristic).
    p.add_argument("--heartbeat", action="store_true")
    p.add_argument("--heartbeat-period", type=int, default=1)
    p.add_argument("--heartbeat-delay", type=int, default=0)
    p.add_argument("--dropout", action="store_true")
    p.add_argument("--dropout-agent", type=int, default=None)
    p.add_argument("--dropout-time", type=int, default=None)
    p.add_argument("--dropout-window-start", type=int, default=None)
    p.add_argument("--dropout-window-end", type=int, default=None)

    # Logging.
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-log", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

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

    env = make_unified_env(
        args.env,
        n_msg_tokens=1,
        dropout_cfg=dropout_cfg,
        heartbeat_cfg=heartbeat_cfg,
    )
    spec = env.spec
    base = env.adapter._env.unwrapped  # type: ignore[attr-defined]
    req_size = args.request_queue_size or int(getattr(base, "request_queue_size", spec.n_agents))
    req_size = max(1, min(req_size, MAX_SLOTS))

    print(f"[env] {spec}  request_queue_size={req_size}  comm={args.comm}")

    q_cfg = QLearnerConfig(
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay_episodes=int(args.epsilon_decay_episodes),
        cooperative=not args.no_cooperative,
        optimistic_init=float(args.optimistic_init),
    )
    # n_subtasks = MAX_SLOTS (indices 0..7) + IDLE(8) + RECOVER(9)
    learner = CooperativeSMDPQLearner(
        q_cfg, n_subtasks=MAX_SLOTS + 2, rng_seed=args.seed
    )

    ctrl_cfg = ControllerConfig(
        n_agents=spec.n_agents,
        request_queue_size=req_size,
        max_option_steps=int(args.max_option_steps),
        comm=bool(args.comm),
        stale_threshold=int(args.stale_threshold),
    )
    controller = HierarchicalController(ctrl_cfg, learner)
    runner = HRLRunner(controller)

    print(
        f"[cfg] alpha={q_cfg.alpha} gamma={q_cfg.gamma} eps=[{q_cfg.epsilon_start}->"
        f"{q_cfg.epsilon_end}] over {q_cfg.epsilon_decay_episodes} eps  "
        f"coop={q_cfg.cooperative} comm={ctrl_cfg.comm}"
    )
    print(
        f"[hb] heartbeat={heartbeat_cfg.enabled} "
        f"(period={heartbeat_cfg.period}, delay={heartbeat_cfg.delay})  "
        f"dropout={dropout_cfg.enabled} "
        f"(agent={dropout_cfg.agent}, time={dropout_cfg.time})"
    )

    logger: RunLogger | None = None
    if not args.no_log:
        run_name = args.run_name or (
            f"hrl{'-comm' if args.comm else ''}-{args.env}-s{args.seed}"
        )
        config = {
            "method": "hrl-paper",
            "comm": bool(args.comm),
            **vars(args),
            "env_spec": repr(spec),
        }
        logger = RunLogger(
            log_dir=args.log_dir,
            run_name=run_name,
            config=config,
            use_tensorboard=True,
        )
        print(f"[log] writing to {logger.run_dir}")

    t0 = time.time()
    total_steps = 0
    try:
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            ep_r, ep_l, ep_deliv = runner.run_episode(
                env,
                seed=ep_seed,
                max_steps=args.max_steps,
                training=True,
            )
            total_steps += ep_l

            elapsed = time.time() - t0
            sps = total_steps / max(elapsed, 1e-6)
            print(
                f"[ep {ep:4d} | steps {total_steps:7d} | t {elapsed:6.1f}s | "
                f"sps {sps:5.0f}] "
                f"ret={ep_r:.3f} ep_len={ep_l} deliv={ep_deliv} "
                f"eps={learner.epsilon:.3f} |Q|={learner.table_size()}",
                flush=True,
            )

            scalars = {
                "train/ep_return": float(ep_r),
                "train/ep_length": float(ep_l),
                "train/ep_deliveries": float(ep_deliv),
                "train/epsilon": float(learner.epsilon),
                "hrl/q_table_size": float(learner.table_size()),
                "time/elapsed_s": float(elapsed),
                "time/sps": float(sps),
                "time/update": float(ep + 1),
            }

            if (ep + 1) % max(1, args.eval_every) == 0:
                eval_returns = []
                eval_lengths = []
                eval_delivs = []
                for ei in range(args.eval_episodes):
                    r, l, d = runner.run_episode(
                        env,
                        seed=10_000 + args.seed + ep * args.eval_episodes + ei,
                        max_steps=args.max_steps,
                        training=False,
                    )
                    eval_returns.append(r)
                    eval_lengths.append(l)
                    eval_delivs.append(d)
                scalars["eval/ep_return_mean"] = float(np.mean(eval_returns))
                scalars["eval/ep_length_mean"] = float(np.mean(eval_lengths))
                scalars["eval/ep_deliveries_mean"] = float(np.mean(eval_delivs))
                print(
                    f"        [eval] ret_mean={scalars['eval/ep_return_mean']:.3f} "
                    f"ep_len={scalars['eval/ep_length_mean']:.1f} "
                    f"deliv={scalars['eval/ep_deliveries_mean']:.2f}",
                    flush=True,
                )

            if logger is not None:
                logger.log_scalars(total_steps, scalars)

        if args.save_q is not None:
            learner.save(args.save_q)
            print(f"[ckpt] Q-table saved to {args.save_q}")
    finally:
        if logger is not None:
            logger.close()
        env.close()


if __name__ == "__main__":
    main()
