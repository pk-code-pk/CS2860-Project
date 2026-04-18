"""
Unified MARL environment interface.

Normalizes heterogeneous multi-agent Gymnasium environments (e.g. ``multigrid``
and ``rware``) into a single, fixed-shape, numpy-friendly contract:

Per-agent outputs (stacked across ``n_agents``):
    obs              float32 (n_agents, obs_dim)      zeroed for dead agents
    available_actions uint8   (n_agents, n_env_actions) 1 = legal, 0 = masked
    alive            bool    (n_agents,)              False once agent terminated
    reward           float32 (n_agents,)              0 for agents dead before step
    done             bool                             episode-level done
    info             dict

Action input:
    actions          int64   (n_agents, 2)            [env_action, msg_token]
                                                     dead agents' rows are ignored
                                                     and replaced with a NOOP

Every concrete env adapter implements the ``BaseAdapter`` protocol below; the
``UnifiedMARLEnv`` class handles alive-mask bookkeeping, observation zeroing,
message routing and NOOP substitution for dead agents.

The communication channel itself (message tokens) is handled *outside* the
env – the wrapper just accepts a message per live agent per step and exposes
the previous-step messages so the policy can concatenate them with obs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    """Static description of a wrapped environment (shared by all episodes)."""

    env_id: str
    n_agents: int
    obs_dim: int
    n_env_actions: int
    noop_action: int
    # Name for book-keeping / logging.
    family: str  # "multigrid" or "rware"


class BaseAdapter(Protocol):
    """Minimal contract an adapter needs to satisfy."""

    spec: EnvSpec

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return (obs [N, D], available_actions [N, A], alive [N], info)."""
        ...

    def step(
        self, env_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """Return (obs, available_actions, alive, reward, done, info).

        ``env_actions`` is an int64 array of shape ``(n_agents,)`` containing an
        environment action for every agent – the adapter can assume entries
        for dead agents have already been replaced with ``spec.noop_action``.
        """
        ...

    def close(self) -> None:
        ...


class UnifiedMARLEnv:
    """
    Thin state machine on top of a ``BaseAdapter`` that:

      * maintains a canonical ``alive`` mask across steps
      * zeroes observations and messages for dead agents
      * substitutes NOOPs for dead agents before calling ``adapter.step``
      * returns the previous-timestep message vector so the policy can
        condition on it without the wrapper caring what "communication"
        means semantically

    Messages are a simple ``(n_agents, n_msg_tokens)`` one-hot matrix where
    each row is the one-hot encoding of the token each agent chose last step.
    Dead agents contribute an all-zero row.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        n_msg_tokens: int = 8,
        zero_obs_on_death: bool = True,
    ):
        self.adapter = adapter
        self.spec = adapter.spec
        self.n_msg_tokens = int(n_msg_tokens)
        self.zero_obs_on_death = bool(zero_obs_on_death)

        self._alive = np.ones(self.spec.n_agents, dtype=bool)
        self._last_messages = np.zeros(
            (self.spec.n_agents, self.n_msg_tokens), dtype=np.float32
        )

    # ---- properties ---------------------------------------------------

    @property
    def n_agents(self) -> int:
        return self.spec.n_agents

    @property
    def obs_dim(self) -> int:
        return self.spec.obs_dim

    @property
    def n_env_actions(self) -> int:
        return self.spec.n_env_actions

    @property
    def alive(self) -> np.ndarray:
        return self._alive.copy()

    @property
    def last_messages(self) -> np.ndarray:
        return self._last_messages.copy()

    # ---- lifecycle ----------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        obs, avail, alive, info = self.adapter.reset(seed=seed)
        self._alive = alive.astype(bool).copy()
        self._last_messages = np.zeros(
            (self.n_agents, self.n_msg_tokens), dtype=np.float32
        )
        obs = self._apply_death_mask_to_obs(obs, self._alive)
        return {
            "obs": obs.astype(np.float32, copy=False),
            "available_actions": avail.astype(np.uint8, copy=False),
            "alive": self._alive.copy(),
            "messages": self._last_messages.copy(),
            "info": info,
        }

    def step(self, joint_action: np.ndarray) -> dict[str, Any]:
        """
        ``joint_action`` is an int array of shape ``(n_agents, 2)``.

        Column 0 is the environment action, column 1 is the message token.
        """
        joint_action = np.asarray(joint_action, dtype=np.int64)
        assert joint_action.shape == (self.n_agents, 2), (
            f"expected joint_action shape {(self.n_agents, 2)}, got {joint_action.shape}"
        )
        env_actions = joint_action[:, 0].copy()
        msg_tokens = joint_action[:, 1].copy()

        # Dead agents act with a noop and emit no message.
        env_actions[~self._alive] = self.spec.noop_action
        msg_tokens[~self._alive] = 0

        obs, avail, alive_after, reward, done, info = self.adapter.step(env_actions)

        # Agents that were dead before this step should not receive any reward.
        reward = np.asarray(reward, dtype=np.float32).copy()
        reward[~self._alive] = 0.0

        # Compose canonical alive mask: once dead, stay dead.
        new_alive = self._alive & alive_after.astype(bool)
        obs = self._apply_death_mask_to_obs(obs, new_alive)

        # Messages for the next step: one-hot, zero rows for newly-dead agents.
        messages = np.zeros(
            (self.n_agents, self.n_msg_tokens), dtype=np.float32
        )
        alive_rows = np.where(new_alive)[0]
        tokens = np.clip(msg_tokens[alive_rows], 0, self.n_msg_tokens - 1)
        messages[alive_rows, tokens] = 1.0

        self._alive = new_alive
        self._last_messages = messages

        if not np.any(new_alive):
            # All agents dead -> force episode end.
            done = True

        return {
            "obs": obs.astype(np.float32, copy=False),
            "available_actions": avail.astype(np.uint8, copy=False),
            "alive": new_alive.copy(),
            "reward": reward,
            "messages": messages.copy(),
            "done": bool(done),
            "info": info,
        }

    def close(self) -> None:
        self.adapter.close()

    # ---- helpers ------------------------------------------------------

    def _apply_death_mask_to_obs(self, obs: np.ndarray, alive: np.ndarray) -> np.ndarray:
        if not self.zero_obs_on_death:
            return obs
        obs = obs.copy()
        obs[~alive] = 0.0
        return obs


def make_unified_env(
    env_id: str,
    *,
    n_agents: int | None = None,
    n_msg_tokens: int = 8,
    seed: int | None = None,
    **adapter_kwargs: Any,
) -> UnifiedMARLEnv:
    """
    Factory that picks the right adapter based on the env id prefix.

    ``env_id`` conventions:
      * ``rware-*`` -> RWARE via ``rware`` registration (FLATTENED obs).
      * ``MultiGrid-*`` -> multigrid, Gymnasium registry.

    ``n_agents`` only applies to multigrid (rware bakes it into the env id).
    """
    if env_id.startswith("rware-"):
        from .rware_adapter import RwareAdapter

        adapter = RwareAdapter(env_id, **adapter_kwargs)
    elif env_id.startswith("MultiGrid-"):
        from .multigrid_adapter import MultigridAdapter

        adapter = MultigridAdapter(env_id, n_agents=n_agents, **adapter_kwargs)
    else:
        raise ValueError(f"Unknown env_id prefix for {env_id!r}")

    env = UnifiedMARLEnv(adapter, n_msg_tokens=n_msg_tokens)
    if seed is not None:
        env.reset(seed=seed)
    return env
