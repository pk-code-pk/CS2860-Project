"""
Episode-driver for the hierarchical (Coop-HRL / COM-Coop-HRL) baseline.

The runner exists so that the HRL baseline plays nicely with the rest
of the repo's infrastructure (UnifiedMARLEnv contract, dropout /
heartbeat wrappers, CSV logger). It is intentionally structured to
resemble ``run_heuristic_episodes`` so the experiment scripts can swap
baselines with minimal ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .controller import HierarchicalController


@dataclass
class HRLRolloutStats:
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_deliveries: list[int]
    episode_td_mean: list[float]
    episode_table_size: list[int]

    def summary(self) -> dict[str, float]:
        if not self.episode_returns:
            return {
                "ep_return_mean": 0.0,
                "ep_length_mean": 0.0,
                "ep_deliveries_mean": 0.0,
                "n_episodes": 0,
                "hrl/q_table_size": 0,
            }
        return {
            "ep_return_mean": float(np.mean(self.episode_returns)),
            "ep_return_std": float(np.std(self.episode_returns)),
            "ep_length_mean": float(np.mean(self.episode_lengths)),
            "ep_deliveries_mean": float(np.mean(self.episode_deliveries)),
            "n_episodes": len(self.episode_returns),
            "hrl/q_table_size": float(self.episode_table_size[-1]) if self.episode_table_size else 0.0,
        }


class HRLRunner:
    """Stateless driver: one ``__call__`` per episode."""

    def __init__(self, controller: HierarchicalController):
        self.controller = controller

    def run_episode(
        self,
        env,
        *,
        seed: int | None = None,
        max_steps: int | None = None,
        training: bool = True,
    ) -> tuple[float, int, int]:
        """Run one episode. Returns ``(ep_return, ep_length, deliveries)``."""
        base = _unwrap_rware(env)
        if base is None:
            raise RuntimeError(
                "HRL baseline requires an RWARE UnifiedMARLEnv (got something else)."
            )

        state = env.reset(seed=seed)
        self.controller.reset()

        info = state["info"]
        alive = state["alive"]
        hb = _heartbeat_ages(info, self.controller.cfg.n_agents)

        ep_r = 0.0
        ep_l = 0
        ep_deliv = 0
        last_reward: Optional[np.ndarray] = None
        done = False

        while not done:
            prim_actions = self.controller.step(
                base,
                alive=alive,
                heartbeat_ages=hb,
                last_reward=last_reward,
                training=training,
            )
            # UnifiedMARLEnv expects (N, 2) = [env_action, msg_token].
            joint = np.stack(
                [prim_actions, np.zeros_like(prim_actions)], axis=-1
            ).astype(np.int64)

            step_out = env.step(joint)

            reward = np.asarray(step_out["reward"], dtype=np.float32)
            ep_r += float((reward * alive).sum())
            ep_deliv += int((reward > 0).sum())
            ep_l += 1

            alive = step_out["alive"]
            info = step_out["info"]
            hb = _heartbeat_ages(info, self.controller.cfg.n_agents)
            last_reward = reward
            done = bool(step_out["done"])

            if max_steps is not None and ep_l >= max_steps:
                break

        # Close out any still-running option with a terminal SMDP update.
        self.controller.on_episode_end(
            base,
            last_reward=last_reward,
            heartbeat_ages=hb,
            alive=alive,
            training=training,
        )
        return ep_r, ep_l, ep_deliv


# ---------------------------------------------------------------------------
# Helpers mirror those in the heuristic baseline.
# ---------------------------------------------------------------------------


def _unwrap_rware(env) -> Any | None:
    try:
        inner = env.adapter._env.unwrapped
    except AttributeError:
        return None
    if not hasattr(inner, "agents") or not hasattr(inner, "request_queue"):
        return None
    return inner


def _heartbeat_ages(info: dict[str, Any] | None, n_agents: int) -> np.ndarray:
    if not info:
        return np.zeros(n_agents, dtype=np.int64)
    hb = info.get("debug_heartbeat_age")
    if hb is None:
        return np.zeros(n_agents, dtype=np.int64)
    arr = np.asarray(hb, dtype=np.int64)
    if arr.ndim == 2 and arr.shape == (n_agents, n_agents):
        masked = arr.copy()
        np.fill_diagonal(masked, np.iinfo(np.int64).max)
        return masked.min(axis=0).astype(np.int64)
    if arr.shape == (n_agents,):
        return arr
    return np.zeros(n_agents, dtype=np.int64)


__all__ = ["HRLRunner", "HRLRolloutStats"]
