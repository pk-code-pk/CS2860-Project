"""
Rollout collection loop.

Single-env, single-threaded runner that fills a ``RolloutBuffer`` with exactly
``T`` steps of experience from a ``UnifiedMARLEnv``, auto-resetting on done.

Separate ``evaluate`` routine runs a handful of episodes with a greedy actor
for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..wrappers.unified import UnifiedMARLEnv
from .buffer import RolloutBuffer
from .mappo import MAPPOTrainer


@dataclass
class RolloutStats:
    episode_returns: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.episode_returns:
            return {"ep_return_mean": 0.0, "ep_length_mean": 0.0, "n_episodes": 0}
        return {
            "ep_return_mean": float(np.mean(self.episode_returns)),
            "ep_return_std": float(np.std(self.episode_returns)),
            "ep_length_mean": float(np.mean(self.episode_lengths)),
            "n_episodes": len(self.episode_returns),
        }


class Runner:
    def __init__(self, env: UnifiedMARLEnv, trainer: MAPPOTrainer):
        self.env = env
        self.trainer = trainer

        self._state: dict[str, np.ndarray] | None = None
        self._ep_return = 0.0
        self._ep_length = 0

    # ------------------------------------------------------------------

    def _ensure_reset(self, seed: int | None = None) -> None:
        if self._state is None:
            self._state = self.env.reset(seed=seed)
            self._ep_return = 0.0
            self._ep_length = 0

    def collect(self, buffer: RolloutBuffer, seed: int | None = None) -> RolloutStats:
        self._ensure_reset(seed=seed)
        buffer.reset()
        stats = RolloutStats()

        while not buffer.full:
            assert self._state is not None
            s = self._state
            out = self.trainer.act(
                obs=s["obs"],
                messages=s["messages"],
                alive=s["alive"],
                avail=s["available_actions"],
                deterministic=False,
            )
            env_a = out["env_actions"]
            msg_a = out["msg_actions"]

            joint = np.stack([env_a, msg_a], axis=-1).astype(np.int64)

            buffer.add(
                obs=s["obs"],
                messages=s["messages"],
                alive=s["alive"],
                avail=s["available_actions"],
                env_actions=env_a,
                msg_actions=msg_a,
                logp_env=out["logp_env"],
                logp_msg=out["logp_msg"],
                value=out["value"],
                reward=np.zeros(self.env.n_agents, dtype=np.float32),  # filled below
                done=False,
            )

            step_out = self.env.step(joint)
            t = buffer.ptr - 1
            buffer.rewards[t] = step_out["reward"]
            buffer.dones[t] = step_out["done"]

            reward_sum = float((step_out["reward"] * s["alive"]).sum())
            self._ep_return += reward_sum
            self._ep_length += 1

            if step_out["done"]:
                stats.episode_returns.append(self._ep_return)
                stats.episode_lengths.append(self._ep_length)
                self._state = self.env.reset()
                self._ep_return = 0.0
                self._ep_length = 0
            else:
                self._state = {
                    "obs": step_out["obs"],
                    "available_actions": step_out["available_actions"],
                    "alive": step_out["alive"],
                    "messages": step_out["messages"],
                    "info": step_out["info"],
                }

        # Bootstrap value for the (possibly non-terminal) final state.
        assert self._state is not None
        last_value = self.trainer.value(
            self._state["obs"], self._state["messages"], self._state["alive"]
        )
        buffer.compute_advantages(
            last_value=last_value,
            gamma=self.trainer.cfg.gamma,
            gae_lambda=self.trainer.cfg.gae_lambda,
        )
        return stats

    # ------------------------------------------------------------------

    def evaluate(
        self, n_episodes: int = 5, seed: int | None = None
    ) -> RolloutStats:
        stats = RolloutStats()
        for i in range(n_episodes):
            s = self.env.reset(seed=(seed + i) if seed is not None else None)
            ep_r, ep_l = 0.0, 0
            done = False
            while not done:
                out = self.trainer.act(
                    obs=s["obs"],
                    messages=s["messages"],
                    alive=s["alive"],
                    avail=s["available_actions"],
                    deterministic=True,
                )
                joint = np.stack([out["env_actions"], out["msg_actions"]], axis=-1)
                step_out = self.env.step(joint.astype(np.int64))
                ep_r += float((step_out["reward"] * s["alive"]).sum())
                ep_l += 1
                done = step_out["done"]
                s = {
                    "obs": step_out["obs"],
                    "available_actions": step_out["available_actions"],
                    "alive": step_out["alive"],
                    "messages": step_out["messages"],
                    "info": step_out["info"],
                }
            stats.episode_returns.append(ep_r)
            stats.episode_lengths.append(ep_l)
        # Reset the runner-side state so collect() starts a fresh episode.
        self._state = None
        return stats
