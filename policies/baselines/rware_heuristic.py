"""
Paper-inspired hierarchical-style heuristic baseline for RWARE.

This is intentionally a *coarse* reactive baseline, not full HRL. The high
level is a fixed allocation rule; the low level is greedy directional
stepping toward an assigned cell. Its only purpose is to serve as a
sanity-check reference point that the MAPPO runs are compared against.

High-level logic per agent (each step, recomputed):

  1. If the agent is already carrying a shelf that is currently requested,
     its intent is DELIVER -> nearest goal.
  2. If the agent is carrying a shelf that is *not* requested (i.e. it was
     returned to it or it picked up the wrong one), its intent is RETURN
     -> any empty shelf cell.
  3. Otherwise the agent's intent is PICKUP -> the closest requested shelf
     that the agent has been assigned (see allocation below).
  4. If the agent has been assigned nothing, NOOP / WAIT.

Allocation rule (fixed, non-learning):

  * Collect the set of currently requested shelves.
  * For each requested shelf, compute Manhattan distance from every
    *participating* agent. An agent participates iff it is alive AND its
    heartbeat age is within ``stale_threshold`` (the heuristic cannot
    distinguish "gone" from "merely stale" any better than that).
  * Greedy round-robin: repeatedly pick the (agent, shelf) pair with the
    smallest distance that has not been assigned yet, until every
    participating agent has at most one assignment.
  * When a teammate is stale beyond threshold, it is dropped from
    participation -> its work is *reallocated* to a fresh teammate via
    this same rule on the next call.

The per-step CLI dropout / heartbeat behaviour (``--dropout``,
``--heartbeat-delay``, etc.) is assumed to be applied inside the
``UnifiedMARLEnv`` / its adapter on the mechanism branch. This module
only *consumes* ``info["debug_heartbeat_age"]`` and friends.

Limitations (documented honestly):
  * We do not respect RWARE's highway vs. shelf-row movement constraints
    when planning. The env cancels illegal moves on its own so we simply
    aim at the target and eat the cancellation cost.
  * We do not plan multi-step paths; it's a one-step greedy heading.
  * We do not reason about collisions between teammates; concurrent
    conflicting moves are resolved by the env.
  * We treat "stale" as binary via ``stale_threshold``. The MAPPO + comm
    policies can do much more nuanced inference; that's the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # imported for the Action/Direction enums; noqa if rware not installed.
    from rware.warehouse import Action, Direction
except Exception:  # pragma: no cover - only hit if rware not installed
    Action = None  # type: ignore[assignment]
    Direction = None  # type: ignore[assignment]


# Fallbacks so the module still imports if rware changes its enums.
_A_NOOP = 0
_A_FORWARD = 1
_A_LEFT = 2
_A_RIGHT = 3
_A_TOGGLE = 4

_D_UP = 0
_D_DOWN = 1
_D_LEFT = 2
_D_RIGHT = 3


@dataclass
class HeuristicConfig:
    """Knobs for the heuristic policy."""

    # A teammate whose heartbeat age exceeds this is treated as potentially
    # dropped and excluded from the allocator. Smaller -> more paranoid.
    stale_threshold: int = 4
    # When we do not know heartbeat ages (fields missing from info), assume
    # everyone is fresh. This is what we want for "no heartbeat/dropout" runs.
    default_heartbeat_age: int = 0
    # If True, when an agent has nothing assigned, it takes NOOP (wait) rather
    # than wandering. Wandering tends to cause collisions in rware-tiny maps.
    noop_when_idle: bool = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RwareHeuristicPolicy:
    """
    Callable that maps (unified_state, info, env) -> joint action (N, 2).

    The policy is stateless across episodes aside from per-agent bookkeeping
    used to break allocation ties consistently. Create one instance per env.
    """

    def __init__(
        self,
        n_agents: int,
        n_msg_tokens: int = 1,
        config: HeuristicConfig | None = None,
    ):
        self.n_agents = int(n_agents)
        self.n_msg_tokens = max(1, int(n_msg_tokens))
        self.cfg = config or HeuristicConfig()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Called by the runner at episode start. Nothing stateful yet."""
        return None

    def act(
        self,
        *,
        alive: np.ndarray,
        info: dict[str, Any],
        env: Any,
    ) -> np.ndarray:
        """
        Return an ``(n_agents, 2)`` int64 array ``[env_action, msg_token]``.

        ``env`` is the ``UnifiedMARLEnv`` instance -- we reach through to
        its adapter's underlying rware Warehouse via
        ``env.adapter._env.unwrapped`` to read positions, shelves, etc.
        """
        base = _unwrap_rware(env)
        if base is None:
            # Fallback: random-ish no-op if we can't introspect.
            return np.zeros((self.n_agents, 2), dtype=np.int64)

        hb = _heartbeat_ages(info, self.n_agents, self.cfg.default_heartbeat_age)
        alive = alive.astype(bool)

        participating = alive & (hb <= self.cfg.stale_threshold)

        assignments = _allocate_shelves(base, participating)
        env_actions = np.full(self.n_agents, _A_NOOP, dtype=np.int64)

        for i, agent in enumerate(base.agents):
            if not alive[i]:
                env_actions[i] = _A_NOOP
                continue

            intent, target = _plan_intent(base, agent, assignments.get(i))
            if intent is None:
                env_actions[i] = _A_NOOP if self.cfg.noop_when_idle else _A_FORWARD
                continue
            env_actions[i] = _step_toward(agent, target, intent)

        msg_tokens = np.zeros(self.n_agents, dtype=np.int64)
        return np.stack([env_actions, msg_tokens], axis=-1)


# ---------------------------------------------------------------------------
# Runner: run the heuristic against UnifiedMARLEnv just like MAPPO does.
# ---------------------------------------------------------------------------


@dataclass
class HeuristicRolloutStats:
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_deliveries: list[int]

    def summary(self) -> dict[str, float]:
        if not self.episode_returns:
            return {
                "ep_return_mean": 0.0,
                "ep_length_mean": 0.0,
                "ep_deliveries_mean": 0.0,
                "n_episodes": 0,
            }
        return {
            "ep_return_mean": float(np.mean(self.episode_returns)),
            "ep_return_std": float(np.std(self.episode_returns)),
            "ep_length_mean": float(np.mean(self.episode_lengths)),
            "ep_deliveries_mean": float(np.mean(self.episode_deliveries)),
            "n_episodes": len(self.episode_returns),
        }


def run_heuristic_episodes(
    env,
    policy: RwareHeuristicPolicy,
    *,
    n_episodes: int,
    seed: int | None = None,
    max_steps: int | None = None,
) -> HeuristicRolloutStats:
    """
    Run ``n_episodes`` full episodes of ``policy`` on ``env`` and return
    aggregated stats. ``env`` is a ``UnifiedMARLEnv``.
    """
    returns: list[float] = []
    lengths: list[int] = []
    deliveries: list[int] = []

    for ep in range(n_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        state = env.reset(seed=ep_seed)
        policy.reset()
        ep_r = 0.0
        ep_l = 0
        ep_deliv = 0
        info = state["info"]
        done = False
        while not done:
            joint = policy.act(alive=state["alive"], info=info, env=env)
            step_out = env.step(joint)
            # Team reward: sum over agents alive at step start (== previous alive)
            ep_r += float((step_out["reward"] * state["alive"]).sum())
            ep_deliv += int((step_out["reward"] > 0).sum())
            ep_l += 1
            info = step_out["info"]
            done = step_out["done"]
            state = {
                "obs": step_out["obs"],
                "available_actions": step_out["available_actions"],
                "alive": step_out["alive"],
                "messages": step_out["messages"],
                "info": info,
            }
            if max_steps is not None and ep_l >= max_steps:
                break

        returns.append(ep_r)
        lengths.append(ep_l)
        deliveries.append(ep_deliv)

    return HeuristicRolloutStats(returns, lengths, deliveries)


# ---------------------------------------------------------------------------
# Allocation + low-level controller internals
# ---------------------------------------------------------------------------


def _unwrap_rware(env) -> Any | None:
    """Reach through UnifiedMARLEnv -> adapter -> gym -> rware Warehouse."""
    try:
        inner = env.adapter._env.unwrapped
    except AttributeError:
        return None
    if not hasattr(inner, "agents") or not hasattr(inner, "request_queue"):
        return None
    return inner


def _heartbeat_ages(
    info: dict[str, Any],
    n_agents: int,
    default_age: int,
) -> np.ndarray:
    """
    Pull a per-sender heartbeat age vector ``(n_agents,)`` from ``info``.

    Mechanism branch may publish ``debug_heartbeat_age`` either as a 1D
    per-sender array or as a 2D ``(recipient, sender)`` matrix produced by
    ``HeartbeatTracker.ages()``. The centralized heuristic only needs the
    *team-best* knowledge per sender, so we min-reduce over recipients for
    the 2D case (= "freshest report anyone in the team has").
    """
    hb = info.get("debug_heartbeat_age")
    if hb is None:
        return np.full(n_agents, default_age, dtype=np.int64)
    arr = np.asarray(hb, dtype=np.int64)
    if arr.ndim == 2 and arr.shape == (n_agents, n_agents):
        # Drop the diagonal (self-freshness is always 0 and not informative
        # for "is teammate s presumed dropped"); take the min across rows.
        masked = arr.copy()
        np.fill_diagonal(masked, np.iinfo(np.int64).max)
        return masked.min(axis=0).astype(np.int64)
    if arr.shape == (n_agents,):
        return arr
    # Unknown shape: fall back to "all fresh" rather than crash.
    return np.full(n_agents, default_age, dtype=np.int64)


def _allocate_shelves(base, participating: np.ndarray) -> dict[int, Any]:
    """
    Return a dict ``{agent_idx: Shelf}`` picking the nearest participating
    agent for each requested shelf via a greedy smallest-distance rule.

    Agents that are already carrying a requested shelf do not need a
    pickup assignment and are skipped for allocation (they just head to
    the goal).
    """
    requests = list(base.request_queue)
    if not requests:
        return {}

    already_busy = set()
    for i, agent in enumerate(base.agents):
        shelf = agent.carrying_shelf
        if shelf is not None and any(s.id == shelf.id for s in requests):
            already_busy.add(i)

    candidate_agents = [
        i for i in range(len(base.agents))
        if participating[i] and i not in already_busy
    ]
    if not candidate_agents:
        return {}

    # Build all (dist, agent_idx, shelf_idx) triples, sort, greedy pick.
    triples: list[tuple[int, int, int, Any]] = []
    for ai in candidate_agents:
        ag = base.agents[ai]
        for si, shelf in enumerate(requests):
            d = abs(ag.x - shelf.x) + abs(ag.y - shelf.y)
            triples.append((d, ai, si, shelf))
    triples.sort(key=lambda t: (t[0], t[1], t[2]))

    assignments: dict[int, Any] = {}
    claimed_agents: set[int] = set()
    claimed_shelves: set[int] = set()
    for d, ai, si, shelf in triples:
        if ai in claimed_agents or si in claimed_shelves:
            continue
        assignments[ai] = shelf
        claimed_agents.add(ai)
        claimed_shelves.add(si)

    return assignments


def _plan_intent(base, agent, assigned_shelf) -> tuple[str | None, tuple[int, int] | None]:
    """
    Decide (intent, target_cell) for a single agent. Intents:

      * "DELIVER": agent carries a requested shelf -> go to nearest goal
      * "RETURN":  agent carries a non-requested shelf -> go to an empty cell
      * "PICKUP":  agent is unloaded and has an assigned shelf -> go to it
      * None:      no intent (wait)
    """
    requested_ids = {s.id for s in base.request_queue}
    shelf = agent.carrying_shelf

    if shelf is not None and shelf.id in requested_ids:
        goal = _nearest(base.goals, (agent.x, agent.y))
        return "DELIVER", goal

    if shelf is not None and shelf.id not in requested_ids:
        drop = _nearest_empty_shelf_cell(base, (agent.x, agent.y))
        if drop is None:
            return None, None
        return "RETURN", drop

    if assigned_shelf is not None:
        return "PICKUP", (assigned_shelf.x, assigned_shelf.y)

    return None, None


def _nearest(cells, pos):
    px, py = pos
    best = None
    best_d = None
    for c in cells:
        cx, cy = c
        d = abs(cx - px) + abs(cy - py)
        if best_d is None or d < best_d:
            best, best_d = (cx, cy), d
    return best


def _nearest_empty_shelf_cell(base, pos):
    """
    Any grid cell that is a shelf row and currently empty of shelves is a
    legal drop site. ``base.grid`` has two layers (agents, shelves) in
    rware 2.x; we use the shelf layer.
    """
    # rware's grid is (layers, H, W) with layer 1 = shelves (0 = agents).
    try:
        shelf_layer = base.grid[1]
    except Exception:
        return None

    highways = getattr(base, "highways", None)
    H, W = shelf_layer.shape
    px, py = pos
    best = None
    best_d = None
    for yy in range(H):
        for xx in range(W):
            if shelf_layer[yy, xx] != 0:
                continue
            if highways is not None:
                try:
                    if highways[yy, xx] != 0:
                        continue  # highway cells are not valid shelf spots
                except Exception:
                    pass
            d = abs(xx - px) + abs(yy - py)
            if best_d is None or d < best_d:
                best, best_d = (xx, yy), d
    return best


def _step_toward(agent, target, intent: str) -> int:
    """
    One-step greedy controller. Returns a rware action index.

    If already at the target cell, execute the intent's terminal action
    (TOGGLE_LOAD for all three intents). Otherwise aim at the target by
    rotating to face the larger component of the offset and then stepping
    forward.
    """
    tx, ty = target
    dx = tx - agent.x
    dy = ty - agent.y

    if dx == 0 and dy == 0:
        # On target cell; pickup / drop / deliver all use TOGGLE_LOAD.
        return _A_TOGGLE

    # Prefer the axis with the larger magnitude first.
    if abs(dx) >= abs(dy):
        want_dir = _D_RIGHT if dx > 0 else _D_LEFT
    else:
        want_dir = _D_DOWN if dy > 0 else _D_UP

    cur_dir = _dir_to_int(agent.dir)
    if cur_dir == want_dir:
        return _A_FORWARD
    return _rotation_action(cur_dir, want_dir)


def _dir_to_int(d) -> int:
    if isinstance(d, int):
        return d
    # rware Direction enum has .value
    v = getattr(d, "value", None)
    if v is not None:
        return int(v)
    # Fallback by name
    name = str(d).split(".")[-1].upper()
    return {"UP": _D_UP, "DOWN": _D_DOWN, "LEFT": _D_LEFT, "RIGHT": _D_RIGHT}.get(name, 0)


def _rotation_action(cur: int, want: int) -> int:
    """
    Table of left/right turns to go from ``cur`` to ``want``. The rware
    Directions form a cycle UP(0) -> RIGHT(3) -> DOWN(1) -> LEFT(2) -> UP
    in the env's own rotation semantics. Rather than recompute the cycle,
    we just enumerate the 12 off-diagonal cases explicitly.
    """
    if cur == want:
        return _A_FORWARD
    # Directions laid out on a compass: UP=0, DOWN=1, LEFT=2, RIGHT=3.
    # rware's Agent.dir rotation: LEFT action rotates counter-clockwise,
    # RIGHT rotates clockwise. We rely on either one turn being correct
    # and if not the next step's reassessment will fix it.
    cw = {
        _D_UP: _D_RIGHT,
        _D_RIGHT: _D_DOWN,
        _D_DOWN: _D_LEFT,
        _D_LEFT: _D_UP,
    }
    if cw[cur] == want:
        return _A_RIGHT
    return _A_LEFT


# ---------------------------------------------------------------------------
# CLI driver -- run the heuristic standalone, with the same env / dropout /
# heartbeat knobs used by `policies.train`. Logs CSV rows that the matrix
# aggregator can ingest alongside MAPPO runs.
# ---------------------------------------------------------------------------


def _build_dropout_cfg(args):
    """Construct a wrapper ``DropoutConfig`` from CLI args."""
    from policies.wrappers import DropoutConfig

    return DropoutConfig(
        enabled=bool(args.dropout),
        agent=args.dropout_agent,
        time=args.dropout_time,
        window_start=args.dropout_window_start,
        window_end=args.dropout_window_end,
    )


def _build_heartbeat_cfg(args):
    """Construct a wrapper ``HeartbeatConfig`` from CLI args."""
    from policies.wrappers import HeartbeatConfig

    return HeartbeatConfig(
        enabled=bool(args.heartbeat),
        period=max(1, int(args.heartbeat_period or 1)),
        delay=max(0, int(args.heartbeat_delay or 0)),
        max_age_clip=max(1, int(args.heartbeat_max_age_clip or 32)),
    )


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Run the rware hierarchical-style heuristic baseline."
    )
    p.add_argument("--env", default="rware-tiny-4ag-v2")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stale-threshold", type=int, default=4)
    p.add_argument(
        "--no-comm",
        action="store_true",
        help=(
            "Heuristic ignores comm tokens regardless; the flag is accepted "
            "for matrix-runner symmetry with MAPPO methods."
        ),
    )
    # Wrapper-side mechanism knobs (forwarded into DropoutConfig /
    # HeartbeatConfig). The defaults match `policies.train` exactly so a
    # heuristic run shares the same regime semantics as a MAPPO run.
    p.add_argument("--heartbeat", action="store_true",
                   help="Enable delayed-heartbeat freshness in info + obs.")
    p.add_argument("--heartbeat-period", type=int, default=1,
                   help="Emit a heartbeat every N steps (>=1).")
    p.add_argument("--heartbeat-delay", type=int, default=0,
                   help="Heartbeats arrive N steps after they were produced.")
    p.add_argument("--heartbeat-max-age-clip", type=int, default=32,
                   help="Clip heartbeat-age features at this value. Use a "
                        "larger value for delay scans beyond d=30.")
    p.add_argument("--dropout", action="store_true",
                   help="Enable permanent teammate dropout mid-episode.")
    p.add_argument("--dropout-agent", type=int, default=None,
                   help="Which agent index to drop (fixed mode).")
    p.add_argument("--dropout-time", type=int, default=None,
                   help="Episode step at which dropout fires (fixed mode).")
    p.add_argument("--dropout-window-start", type=int, default=None,
                   help="Window mode: inclusive start of random dropout step.")
    p.add_argument("--dropout-window-end", type=int, default=None,
                   help="Window mode: exclusive end of random dropout step.")

    p.add_argument("--log-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-log", action="store_true")
    return p.parse_args()


def main() -> None:
    """Stand-alone CLI: run N episodes of the heuristic, log per-episode rows."""
    args = _parse_args()

    np.random.seed(args.seed)
    from policies.wrappers import make_unified_env

    dropout_cfg = _build_dropout_cfg(args)
    heartbeat_cfg = _build_heartbeat_cfg(args)
    env = make_unified_env(
        args.env,
        n_msg_tokens=1,
        dropout_cfg=dropout_cfg,
        heartbeat_cfg=heartbeat_cfg,
    )

    # If heartbeat is *off*, the wrapper still publishes ages clipped at the
    # tracker's `max_age_clip` sentinel for every off-diagonal entry, which
    # would make the heuristic's stale-gating mark every teammate as dead and
    # NOOP forever. Disable the gate in that case so the baseline degrades
    # gracefully to "always treat everyone as fresh".
    if not heartbeat_cfg.enabled:
        effective_stale_threshold = 10**9
    else:
        effective_stale_threshold = int(args.stale_threshold)

    policy = RwareHeuristicPolicy(
        n_agents=env.spec.n_agents,
        n_msg_tokens=1,
        config=HeuristicConfig(stale_threshold=effective_stale_threshold),
    )
    print(
        f"[heuristic] env={args.env} n_agents={env.spec.n_agents} "
        f"episodes={args.episodes} max_steps={args.max_steps} "
        f"stale_threshold={args.stale_threshold} "
        f"(effective={effective_stale_threshold})"
    )
    print(
        f"[heuristic] heartbeat={heartbeat_cfg.enabled} "
        f"(period={heartbeat_cfg.period}, delay={heartbeat_cfg.delay})  "
        f"dropout={dropout_cfg.enabled} "
        f"(agent={dropout_cfg.agent}, time={dropout_cfg.time}, "
        f"window=[{dropout_cfg.window_start},{dropout_cfg.window_end}))"
    )

    logger = None
    if not args.no_log:
        from policies.logger import RunLogger

        run_name = args.run_name or f"heuristic-{args.env}-s{args.seed}"
        config = {
            "method": "heuristic",
            "env": args.env,
            **vars(args),
            "env_spec": repr(env.spec),
        }
        logger = RunLogger(
            log_dir=args.log_dir,
            run_name=run_name,
            config=config,
            use_tensorboard=False,
        )
        print(f"[log] writing to {logger.run_dir}")

    all_returns: list[float] = []
    all_lengths: list[int] = []
    all_deliveries: list[int] = []
    try:
        for ep in range(args.episodes):
            stats = run_heuristic_episodes(
                env,
                policy,
                n_episodes=1,
                seed=args.seed + ep,
                max_steps=args.max_steps,
            )
            ep_return = stats.episode_returns[0]
            ep_length = stats.episode_lengths[0]
            ep_deliv = stats.episode_deliveries[0]
            all_returns.append(ep_return)
            all_lengths.append(ep_length)
            all_deliveries.append(ep_deliv)

            print(
                f"[ep {ep:3d}] return={ep_return:.3f} "
                f"length={ep_length} deliveries={ep_deliv}",
                flush=True,
            )
            if logger is not None:
                step = (ep + 1) * args.max_steps
                logger.log_scalars(
                    step=step,
                    metrics={
                        # Schema kept compatible with `policies.train` so the
                        # aggregator can read both kinds of run with one path.
                        "train/ep_return_mean": float(ep_return),
                        "train/ep_length_mean": float(ep_length),
                        "train/n_episodes": 1.0,
                        "eval/ep_return_mean": float(ep_return),
                        "eval/ep_length_mean": float(ep_length),
                        "heuristic/deliveries": float(ep_deliv),
                        "time/update": float(ep + 1),
                    },
                )

        if all_returns:
            mean_r = float(np.mean(all_returns))
            std_r = float(np.std(all_returns))
            print(
                f"[heuristic] {len(all_returns)} eps: "
                f"return mean={mean_r:.3f} std={std_r:.3f} "
                f"deliveries mean={np.mean(all_deliveries):.2f}"
            )
    finally:
        if logger is not None:
            logger.close()
        env.close()


__all__ = [
    "HeuristicConfig",
    "RwareHeuristicPolicy",
    "HeuristicRolloutStats",
    "run_heuristic_episodes",
]


if __name__ == "__main__":
    main()
