"""
Adapter for ``MultiGrid-*`` environments into the ``UnifiedMARLEnv`` contract.

Multigrid specifics we normalize away:
  * ``reset`` / ``step`` return dicts keyed by agent index (0..N-1).
  * Per-agent obs is ``{'image': (V, V, 3) int, 'direction': int, 'mission': ...}``;
    we flatten it to ``image.ravel()`` concatenated with a one-hot of direction
    and drop the mission string (the envs we care about have a fixed mission).
  * Multigrid does support per-agent termination, so the alive mask is real.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import multigrid.envs  # noqa: F401  (registers MultiGrid-* env ids)
import numpy as np
from multigrid.core.constants import Direction

from .unified import EnvSpec


_MULTIGRID_NUM_ACTIONS = 7  # left, right, forward, pickup, drop, toggle, done
_MULTIGRID_NOOP = 6  # "done" is a no-op in MiniGrid semantics


class MultigridAdapter:
    """Concrete adapter wrapping a single ``MultiGrid-*`` env instance."""

    spec: EnvSpec

    def __init__(self, env_id: str, n_agents: int | None = None, **make_kwargs: Any):
        if n_agents is not None:
            make_kwargs.setdefault("agents", int(n_agents))
        make_kwargs.setdefault("render_mode", None)
        self._env = gym.make(
            env_id,
            disable_env_checker=True,
            **make_kwargs,
        )
        base = self._env.unwrapped
        n = int(base.num_agents)

        # Compute the flat obs dim from the first agent's observation space.
        agent_space = base.observation_space[0]
        image_shape = agent_space["image"].shape
        image_dim = int(np.prod(image_shape))
        self._image_dim = image_dim
        self._n_dirs = len(Direction)
        flat_dim = image_dim + self._n_dirs

        self.spec = EnvSpec(
            env_id=env_id,
            n_agents=n,
            obs_dim=flat_dim,
            n_env_actions=_MULTIGRID_NUM_ACTIONS,
            noop_action=_MULTIGRID_NOOP,
            family="multigrid",
        )

    # ---- interface ----------------------------------------------------

    def reset(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        obs_dict, info = self._env.reset(seed=seed)
        obs = self._stack_obs(obs_dict)
        avail = self._available_actions()
        alive = np.ones(self.spec.n_agents, dtype=bool)
        return obs, avail, alive, dict(info)

    def step(
        self, env_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        action_dict = {i: int(env_actions[i]) for i in range(self.spec.n_agents)}
        obs_dict, rewards, terms, truncs, info = self._env.step(action_dict)

        obs = self._stack_obs(obs_dict)
        reward = np.asarray(
            [float(rewards[i]) for i in range(self.spec.n_agents)],
            dtype=np.float32,
        )
        terminated = np.asarray(
            [bool(terms.get(i, False)) for i in range(self.spec.n_agents)], dtype=bool
        )
        truncated = np.asarray(
            [bool(truncs.get(i, False)) for i in range(self.spec.n_agents)], dtype=bool
        )
        alive = ~(terminated | truncated)
        # Episode-level done = all agents are done, or any agent was truncated
        # (truncations in multigrid are global: step_count >= max_steps).
        done = bool(np.all(~alive)) or bool(truncated.any())
        avail = self._available_actions()
        return obs, avail, alive, reward, done, dict(info)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # ---- helpers ------------------------------------------------------

    def _stack_obs(self, obs_dict: dict) -> np.ndarray:
        out = np.zeros((self.spec.n_agents, self.spec.obs_dim), dtype=np.float32)
        for i in range(self.spec.n_agents):
            agent_obs = obs_dict[i]
            image = np.asarray(agent_obs["image"], dtype=np.float32).ravel()
            direction = int(agent_obs["direction"])
            out[i, : self._image_dim] = image / 255.0  # rough normalization
            out[i, self._image_dim + direction] = 1.0
        return out

    def _available_actions(self) -> np.ndarray:
        """All 7 multigrid actions are always legal in the base envs."""
        return np.ones(
            (self.spec.n_agents, self.spec.n_env_actions), dtype=np.uint8
        )
