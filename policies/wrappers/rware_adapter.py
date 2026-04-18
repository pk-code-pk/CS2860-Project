"""
Adapter for ``rware-*`` environments into the ``UnifiedMARLEnv`` contract.

RWARE specifics we normalize away:
  * ``reset`` / ``step`` return tuples of per-agent obs rather than dicts.
  * Observations are already flat ``float32`` vectors when the env is built
    with ``ObservationType.FLATTENED`` (the default for the ``rware-*``
    registered ids).
  * Rewards are returned as a list of floats.
  * No per-agent termination – only a global ``done`` flag.

We force ``msg_bits=0`` so RWARE's built-in per-agent communication bits do
not interfere with the communication channel we add on top of the wrapper.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import rware  # noqa: F401  (registers rware-* env ids)

from .unified import EnvSpec


# RWARE's Action enum has 5 entries (NOOP=0, FORWARD, LEFT, RIGHT, TOGGLE_LOAD).
_RWARE_NUM_ACTIONS = 5
_RWARE_NOOP = 0


class RwareAdapter:
    """Concrete adapter wrapping a single ``rware-*`` env instance."""

    spec: EnvSpec

    def __init__(self, env_id: str, **make_kwargs: Any):
        # Force msg_bits=0 so the env's action space is a plain Discrete(5)
        # per agent. The unified wrapper adds its own communication channel.
        make_kwargs.setdefault("msg_bits", 0)
        self._env = gym.make(
            env_id,
            disable_env_checker=True,
            render_mode=None,
            **make_kwargs,
        )
        base = self._env.unwrapped

        obs_space = base.observation_space
        assert isinstance(obs_space, gym.spaces.Tuple), (
            "rware adapter expects a Tuple observation space; ensure the env "
            "was registered in FLATTENED mode."
        )
        sample_space = obs_space.spaces[0]
        assert isinstance(sample_space, gym.spaces.Box) and len(sample_space.shape) == 1, (
            "rware observations must be flat Box vectors; use a FLATTENED env id."
        )

        self.spec = EnvSpec(
            env_id=env_id,
            n_agents=int(base.n_agents),
            obs_dim=int(sample_space.shape[0]),
            n_env_actions=_RWARE_NUM_ACTIONS,
            noop_action=_RWARE_NOOP,
            family="rware",
        )

    # ---- interface ----------------------------------------------------

    def reset(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        obs_tuple, info = self._env.reset(seed=seed)
        obs = self._stack_obs(obs_tuple)
        avail = self._available_actions()
        alive = np.ones(self.spec.n_agents, dtype=bool)
        return obs, avail, alive, info

    def step(
        self, env_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        actions = [int(a) for a in env_actions]
        obs_tuple, rewards, done, truncated, info = self._env.step(actions)
        obs = self._stack_obs(obs_tuple)
        reward = np.asarray(rewards, dtype=np.float32)
        episode_done = bool(done) or bool(truncated)
        # RWARE has no per-agent termination; agents are "alive" for the entire
        # episode. Episode-level termination is signalled via the `done` return
        # value. UnifiedMARLEnv handles the "force done when all agents dead"
        # corner case for envs that DO have per-agent termination.
        alive = np.ones(self.spec.n_agents, dtype=bool)
        avail = self._available_actions()
        return obs, avail, alive, reward, episode_done, info

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            # rware uses Pyglet; teardown can throw on some platforms.
            pass

    # ---- helpers ------------------------------------------------------

    def _stack_obs(self, obs_tuple: tuple) -> np.ndarray:
        return np.stack([np.asarray(o, dtype=np.float32) for o in obs_tuple], axis=0)

    def _available_actions(self) -> np.ndarray:
        """All 5 RWARE actions are always legal."""
        return np.ones(
            (self.spec.n_agents, self.spec.n_env_actions), dtype=np.uint8
        )
