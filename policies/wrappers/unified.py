"""
Unified MARL environment interface.

Normalizes heterogeneous multi-agent Gymnasium environments (e.g. ``multigrid``
and ``rware``) into a single, fixed-shape, numpy-friendly contract:

Per-agent outputs (stacked across ``n_agents``):
    obs              float32 (n_agents, obs_dim)      zeroed for dead agents;
                                                     obs_dim includes optional
                                                     heartbeat-freshness features
    available_actions uint8   (n_agents, n_env_actions) 1 = legal, 0 = masked
    alive            bool    (n_agents,)              True internal alive mask
    reward           float32 (n_agents,)              0 for agents dead before step
    done             bool                             episode-level done
    info             dict                             see debug keys below

Action input:
    actions          int64   (n_agents, 2)            [env_action, msg_token]
                                                     dead agents' rows are ignored
                                                     and replaced with a NOOP

Ambiguity mechanism (new, opt-in):

  * A ``DropoutConfig`` can force one agent to permanently fail mid-episode.
  * A ``HeartbeatConfig`` routes delayed per-agent heartbeats through a
    ``HeartbeatTracker`` and appends per-agent freshness features to each
    agent's obs vector. Dead agents stop producing new heartbeats; already
    in-flight heartbeats still arrive on schedule, which is what creates
    the ambiguity between "gone" and "merely stale".

The wrapper keeps the **true** alive mask for internal masking / NOOP
substitution / reward zeroing. The actor-visible obs pathway only carries
heartbeat-derived freshness, never an oracle teammate-death signal.

Stable debug keys always added to ``info`` (zeros/defaults when the
mechanism is off):

  info["debug_true_alive"]     bool (n_agents,)
  info["debug_dropout_fired"]  bool
  info["debug_failed_agent"]   int  (-1 if no dropout has fired)
  info["debug_heartbeat_age"]  int64 (n_agents, n_agents)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np

from .heartbeat import HeartbeatConfig, HeartbeatTracker


@dataclass(frozen=True)
class EnvSpec:
    """Static description of a wrapped environment (shared by all episodes)."""

    env_id: str
    n_agents: int
    obs_dim: int
    n_env_actions: int
    noop_action: int
    family: str  # "multigrid" or "rware"


class BaseAdapter(Protocol):
    """Minimal contract an adapter needs to satisfy."""

    spec: EnvSpec

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        ...

    def step(
        self, env_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        ...

    def close(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


@dataclass
class DropoutConfig:
    """
    Controlled permanent-dropout config. Two modes:

      * Fixed:   ``agent`` and ``time`` are both set -> deterministic.
      * Window:  only ``window_start`` / ``window_end`` set -> random agent
                 & random step inside the window, drawn with the wrapper's
                 reset seed.

    Leaving ``enabled`` False makes the wrapper behave exactly like the
    pre-mechanism baseline.
    """

    enabled: bool = False
    agent: int | None = None
    time: int | None = None
    window_start: int | None = None
    window_end: int | None = None

    def __post_init__(self) -> None:
        fixed_specified = self.agent is not None or self.time is not None
        window_specified = self.window_start is not None or self.window_end is not None
        if not self.enabled:
            return
        if fixed_specified and window_specified:
            raise ValueError(
                "DropoutConfig: provide either fixed (agent+time) OR a random "
                "window (window_start/end), not both."
            )
        if fixed_specified:
            if self.agent is None or self.time is None:
                raise ValueError(
                    "DropoutConfig: fixed mode needs both --dropout-agent and "
                    "--dropout-time."
                )
            if self.agent < 0 or self.time < 0:
                raise ValueError("DropoutConfig: agent/time must be non-negative.")
        elif window_specified:
            if self.window_start is None or self.window_end is None:
                raise ValueError(
                    "DropoutConfig: window mode needs both --dropout-window-start "
                    "and --dropout-window-end."
                )
            if self.window_end <= self.window_start:
                raise ValueError(
                    "DropoutConfig: window_end must be strictly greater than "
                    "window_start."
                )
        else:
            raise ValueError(
                "DropoutConfig: enabled but neither fixed nor window mode was "
                "specified."
            )


@dataclass
class _DropoutState:
    """Per-episode resolved dropout target."""

    agent: int | None = None
    time: int | None = None
    fired: bool = False


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class UnifiedMARLEnv:
    """
    Thin state machine on top of a ``BaseAdapter`` that:

      * maintains a canonical **true** ``alive`` mask across steps
      * zeroes observations and messages for dead agents
      * substitutes NOOPs for dead agents before calling ``adapter.step``
      * optionally applies permanent dropout to one agent mid-episode
      * optionally runs a delayed-heartbeat tracker and appends per-agent
        freshness features to each agent's obs vector

    The obs vector returned by this wrapper has shape ``(n_agents, obs_dim)``
    where ``obs_dim == adapter.spec.obs_dim + heartbeat_feature_width``.
    The ``spec`` property exposes that effective obs_dim so downstream
    networks size themselves correctly.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        n_msg_tokens: int = 8,
        zero_obs_on_death: bool = True,
        dropout_cfg: DropoutConfig | None = None,
        heartbeat_cfg: HeartbeatConfig | None = None,
    ):
        self.adapter = adapter
        self.n_msg_tokens = int(n_msg_tokens)
        if self.n_msg_tokens < 1:
            raise ValueError(
                f"n_msg_tokens must be >= 1 (got {self.n_msg_tokens}). "
                "Use n_msg_tokens=1 to disable the comm channel via the --no-comm ablation."
            )
        self.zero_obs_on_death = bool(zero_obs_on_death)

        self.dropout_cfg = dropout_cfg or DropoutConfig()
        self.heartbeat_cfg = heartbeat_cfg or HeartbeatConfig()

        self._n_agents = adapter.spec.n_agents
        self._base_obs_dim = adapter.spec.obs_dim

        # Heartbeat freshness contributes (n_agents,) features per agent.
        self._hb_feature_width = self._n_agents if self.heartbeat_cfg.enabled else 0
        self._tracker = HeartbeatTracker(self._n_agents, self.heartbeat_cfg)

        # Build an outer spec with the *effective* obs_dim so callers (trainer
        # ctor, spec printouts, etc.) see the real shape of the obs tensor.
        eff_obs_dim = self._base_obs_dim + self._hb_feature_width
        self.spec = replace(adapter.spec, obs_dim=eff_obs_dim)

        self._alive = np.ones(self._n_agents, dtype=bool)
        self._last_messages = np.zeros(
            (self._n_agents, self.n_msg_tokens), dtype=np.float32
        )

        # Per-episode mutable state.
        self._t = 0
        self._dropout_state = _DropoutState()
        # Dedicated RNG so dropout-window sampling is reproducible per reset seed.
        self._dropout_rng = np.random.default_rng()

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
            (self._n_agents, self.n_msg_tokens), dtype=np.float32
        )
        self._t = 0
        self._tracker.reset()

        # Reset-seeded RNG so window-mode dropout is reproducible per seed.
        if seed is not None:
            self._dropout_rng = np.random.default_rng(int(seed) + 0xB1B_EAD)

        self._dropout_state = self._resolve_dropout()

        obs = self._apply_death_mask_to_obs(obs, self._alive)
        obs_full = self._append_freshness(obs)
        info = self._with_debug_info(info)
        return {
            "obs": obs_full.astype(np.float32, copy=False),
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
        assert joint_action.shape == (self._n_agents, 2), (
            f"expected joint_action shape {(self._n_agents, 2)}, got {joint_action.shape}"
        )
        env_actions = joint_action[:, 0].copy()
        msg_tokens = joint_action[:, 1].copy()

        # Step-time dropout: flip a target agent to dead *before* its action
        # is applied so the failed agent never produces a real env action.
        alive_pre_step = self._apply_dropout_if_due()

        # Dead agents act with a noop and emit no message.
        env_actions[~alive_pre_step] = self.spec.noop_action
        msg_tokens[~alive_pre_step] = 0

        obs, avail, alive_after, reward, done, info = self.adapter.step(env_actions)

        # Agents that were dead before this step should not receive any reward.
        reward = np.asarray(reward, dtype=np.float32).copy()
        reward[~alive_pre_step] = 0.0

        # Compose canonical alive mask: once dead, stay dead. The adapter's
        # own alive_after may disagree if the env has no per-agent termination;
        # the pre-step dropout-adjusted mask is authoritative.
        new_alive = alive_pre_step & alive_after.astype(bool)
        obs = self._apply_death_mask_to_obs(obs, new_alive)

        # Messages for the next step: one-hot, zero rows for newly-dead agents.
        messages = np.zeros(
            (self._n_agents, self.n_msg_tokens), dtype=np.float32
        )
        alive_rows = np.where(new_alive)[0]
        tokens = np.clip(msg_tokens[alive_rows], 0, self.n_msg_tokens - 1)
        messages[alive_rows, tokens] = 1.0

        # Heartbeat: emissions come from agents who were alive AT THE START of
        # this step (alive_pre_step). That keeps a single in-flight heartbeat
        # from the just-failed agent's final live step, which is exactly the
        # ambiguity we want ("last echo before silence").
        self._tracker.step(alive_before=alive_pre_step)

        self._alive = new_alive
        self._last_messages = messages
        self._t += 1

        if not np.any(new_alive):
            done = True

        obs_full = self._append_freshness(obs)
        info = self._with_debug_info(info)

        return {
            "obs": obs_full.astype(np.float32, copy=False),
            "available_actions": avail.astype(np.uint8, copy=False),
            "alive": new_alive.copy(),
            "reward": reward,
            "messages": messages.copy(),
            "done": bool(done),
            "info": info,
        }

    def close(self) -> None:
        self.adapter.close()

    # ---- dropout helpers ---------------------------------------------

    def _resolve_dropout(self) -> _DropoutState:
        cfg = self.dropout_cfg
        if not cfg.enabled:
            return _DropoutState()
        if cfg.agent is not None and cfg.time is not None:
            # Fixed mode takes precedence (validated in __post_init__).
            agent = int(cfg.agent) % self._n_agents
            return _DropoutState(agent=agent, time=int(cfg.time), fired=False)
        # Window mode.
        assert cfg.window_start is not None and cfg.window_end is not None
        t = int(self._dropout_rng.integers(cfg.window_start, cfg.window_end))
        a = int(self._dropout_rng.integers(0, self._n_agents))
        return _DropoutState(agent=a, time=t, fired=False)

    def _apply_dropout_if_due(self) -> np.ndarray:
        """
        Returns the alive mask to use for *this* step's action masking,
        applying any scheduled dropout. Mutates internal state so the
        failed agent stays dead from this step onward.
        """
        alive = self._alive.copy()
        ds = self._dropout_state
        if (
            self.dropout_cfg.enabled
            and not ds.fired
            and ds.time is not None
            and ds.agent is not None
            and self._t >= ds.time
            and 0 <= ds.agent < self._n_agents
        ):
            alive[ds.agent] = False
            ds.fired = True
            # Persist into `_alive` as well so the failed agent cannot be
            # resurrected even if the adapter's alive_after claims otherwise.
            self._alive[ds.agent] = False
        return alive

    # ---- obs / info helpers ------------------------------------------

    def _apply_death_mask_to_obs(self, obs: np.ndarray, alive: np.ndarray) -> np.ndarray:
        if not self.zero_obs_on_death:
            return obs
        obs = obs.copy()
        obs[~alive] = 0.0
        return obs

    def _append_freshness(self, obs: np.ndarray) -> np.ndarray:
        """Append per-agent heartbeat freshness features to ``obs``."""
        if self._hb_feature_width == 0:
            return obs
        fresh = self._tracker.freshness_features()  # (N, N)
        assert fresh.shape == (self._n_agents, self._n_agents)
        # Each agent i sees the freshness row [i, :] (its own view of all senders).
        return np.concatenate([obs, fresh], axis=1)

    def _with_debug_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        info = dict(info) if info else {}
        info["debug_true_alive"] = self._alive.copy()
        info["debug_dropout_fired"] = bool(self._dropout_state.fired)
        info["debug_failed_agent"] = (
            int(self._dropout_state.agent)
            if self._dropout_state.fired and self._dropout_state.agent is not None
            else -1
        )
        info["debug_heartbeat_age"] = self._tracker.ages()
        return info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_unified_env(
    env_id: str,
    *,
    n_agents: int | None = None,
    n_msg_tokens: int = 8,
    seed: int | None = None,
    dropout_cfg: DropoutConfig | None = None,
    heartbeat_cfg: HeartbeatConfig | None = None,
    **adapter_kwargs: Any,
) -> UnifiedMARLEnv:
    """
    Factory that picks the right adapter based on the env id prefix.
    """
    if env_id.startswith("rware-"):
        from .rware_adapter import RwareAdapter

        adapter = RwareAdapter(env_id, **adapter_kwargs)
    elif env_id.startswith("MultiGrid-"):
        from .multigrid_adapter import MultigridAdapter

        adapter = MultigridAdapter(env_id, n_agents=n_agents, **adapter_kwargs)
    else:
        raise ValueError(f"Unknown env_id prefix for {env_id!r}")

    env = UnifiedMARLEnv(
        adapter,
        n_msg_tokens=n_msg_tokens,
        dropout_cfg=dropout_cfg,
        heartbeat_cfg=heartbeat_cfg,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


__all__ = [
    "EnvSpec",
    "BaseAdapter",
    "UnifiedMARLEnv",
    "DropoutConfig",
    "HeartbeatConfig",
    "make_unified_env",
]
