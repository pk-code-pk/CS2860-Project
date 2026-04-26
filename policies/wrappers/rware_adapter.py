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

Optional reward shaping (``shape_rewards=True``):
  * RWARE's stock reward is +1 on delivery, 0 otherwise. With 4 agents on a
    tiny warehouse this is brutally sparse and vanilla MAPPO struggles to
    bootstrap inside a CPU-friendly budget. The standard fix used in the
    RWARE literature is a small dense bonus on the pick-up step:
    ``+pickup_bonus`` added to an agent's reward the first step its
    ``carrying_shelf`` flips from ``None`` to a *currently-requested* shelf.
    Picking up a shelf that is not in the request queue gives no bonus
    (otherwise the agent could farm pickup bonus by toggling random
    shelves). An optional small per-step ``step_penalty`` is also exposed
    but defaults to 0.
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

    def __init__(
        self,
        env_id: str,
        *,
        shape_rewards: bool = False,
        pickup_bonus: float = 0.5,
        step_penalty: float = 0.0,
        **make_kwargs: Any,
    ):
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

        # Reward shaping config (see module docstring).
        self.shape_rewards = bool(shape_rewards)
        self.pickup_bonus = float(pickup_bonus)
        self.step_penalty = float(step_penalty)
        # Per-agent "was carrying a requested shelf last step" flag, used to
        # detect the False -> True pick-up transition. Initialised in reset().
        self._prev_carrying_requested = np.zeros(self.spec.n_agents, dtype=bool)

    # ---- interface ----------------------------------------------------

    def reset(
        self, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        obs_tuple, info = self._env.reset(seed=seed)
        obs = self._stack_obs(obs_tuple)
        avail = self._available_actions()
        alive = np.ones(self.spec.n_agents, dtype=bool)
        # Reset shaping bookkeeping.
        self._prev_carrying_requested = self._carrying_requested_mask()
        return obs, avail, alive, info

    def step(
        self, env_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        actions = [int(a) for a in env_actions]
        obs_tuple, rewards, done, truncated, info = self._env.step(actions)
        obs = self._stack_obs(obs_tuple)
        reward = np.asarray(rewards, dtype=np.float32)

        # Optional reward shaping. Done after the env step so we can read
        # post-step `carrying_shelf` state.
        if self.shape_rewards:
            cur_carrying = self._carrying_requested_mask()
            picked_up = cur_carrying & ~self._prev_carrying_requested
            if self.pickup_bonus != 0.0:
                reward = reward + picked_up.astype(np.float32) * self.pickup_bonus
            if self.step_penalty != 0.0:
                reward = reward - self.step_penalty
            self._prev_carrying_requested = cur_carrying

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

    def _carrying_requested_mask(self) -> np.ndarray:
        """
        Per-agent bool: ``True`` iff the agent is currently carrying a shelf
        whose ``id`` is in the env's request queue. Used by reward shaping
        to detect the False -> True pickup transition on a *requested*
        shelf (so agents can't farm pickup bonus on random shelves).
        """
        base = self._env.unwrapped
        try:
            requested_ids = {s.id for s in base.request_queue}
        except Exception:
            requested_ids = set()
        out = np.zeros(self.spec.n_agents, dtype=bool)
        for i, ag in enumerate(base.agents):
            shelf = getattr(ag, "carrying_shelf", None)
            if shelf is None:
                continue
            out[i] = shelf.id in requested_ids
        return out

    def message_intent_labels(
        self,
        *,
        n_msg_tokens: int,
        alive: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return supervised message labels for grounded intent communication.

        Labels are fixed semantic tokens, not learned language meanings:

        * 0: available / idle / no assigned request
        * 1: carrying a requested shelf toward delivery
        * 2: carrying a non-requested shelf
        * 3+k: assigned to request-queue slot k (clipped to available tokens)

        The assignment labels give the message channel information heartbeat
        cannot provide: which piece of work a teammate was handling before
        dropout. Dead agents are assigned 0 here; the wrapper masks them out
        of the auxiliary loss and continues echoing their last live message.
        """
        n_tokens = max(1, int(n_msg_tokens))
        labels = np.zeros(self.spec.n_agents, dtype=np.int64)
        if n_tokens <= 1:
            return labels

        base = self._env.unwrapped
        alive_mask = (
            np.ones(self.spec.n_agents, dtype=bool)
            if alive is None
            else np.asarray(alive, dtype=bool)
        )
        requests = list(getattr(base, "request_queue", []))
        requested_ids = {getattr(s, "id", None) for s in requests}

        already_busy: set[int] = set()
        for i, agent in enumerate(base.agents):
            if not alive_mask[i]:
                continue
            shelf = getattr(agent, "carrying_shelf", None)
            if shelf is None:
                continue
            if getattr(shelf, "id", None) in requested_ids:
                labels[i] = min(1, n_tokens - 1)
                already_busy.add(i)
            else:
                labels[i] = min(2, n_tokens - 1)
                already_busy.add(i)

        if not requests or n_tokens <= 3:
            return labels

        candidate_agents = [
            i for i in range(self.spec.n_agents)
            if alive_mask[i] and i not in already_busy
        ]
        triples: list[tuple[int, int, int]] = []
        for ai in candidate_agents:
            ag = base.agents[ai]
            for slot_idx, shelf in enumerate(requests):
                d = abs(int(ag.x) - int(shelf.x)) + abs(int(ag.y) - int(shelf.y))
                triples.append((d, ai, slot_idx))
        triples.sort()

        used_agents: set[int] = set()
        used_slots: set[int] = set()
        for _dist, agent_idx, slot_idx in triples:
            if agent_idx in used_agents or slot_idx in used_slots:
                continue
            token = min(3 + slot_idx, n_tokens - 1)
            labels[agent_idx] = token
            used_agents.add(agent_idx)
            used_slots.add(slot_idx)
            if len(used_agents) == len(candidate_agents):
                break

        return labels
