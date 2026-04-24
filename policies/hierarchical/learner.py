"""
Tabular Cooperative SMDP Q-learner.

Implements the high-level learner described in Ghavamzadeh et al.'s
Cooperative-HRL (and its COM-Cooperative-HRL variant):

  * The agent selects *cooperative subtasks* (options) at a temporally
    abstracted level and the Q-function is learned with the SMDP
    Bellman target

        Q(s,a) <- Q(s,a) + alpha * ( sum_{i=0..k-1} gamma^i r_{t+i}
                                     + gamma^k max_a' Q(s',a')
                                     - Q(s,a) )

    where k is the number of primitive steps the option actually took.

  * "Cooperative" means the Q-key incorporates teammates' currently
    selected cooperative subtasks, exactly as in the paper. Whether
    these teammate values are *correct* is determined by the COM
    variant:
      - with comm:       the agent has ground-truth teammate subtasks
                         at each boundary;
      - without comm:    the agent uses its locally-remembered copy
                         (updated only when it happens to replan at
                         the same time as a teammate).

  * Homogeneous agents share a single Q-table (matches the paper's
    cooperative-homogeneous assumption and makes the tabular learner
    actually trainable in a few thousand episodes).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .subtasks import AbstractState, HighSubtask


Key = tuple  # (carrying, closest_free_slot, teammate_assignments_tuple)


def _state_key(s: AbstractState) -> Key:
    return (int(s.carrying), int(s.closest_free_slot), tuple(s.teammate_assignments))


@dataclass
class QLearnerConfig:
    """SMDP Q-learner hyperparameters."""

    alpha: float = 0.3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    # Cooperative conditioning: if False, the Q-key ignores teammates
    # entirely -> reduces to independent hierarchical Q-learning. Kept
    # as a flag so we can ablate.
    cooperative: bool = True
    # When True, unknown (s,a) pairs default to an optimistic initial
    # value to encourage exploration. Matches the paper's "optimistic
    # initialization" remark for tabular cooperative learners.
    optimistic_init: float = 0.0


class CooperativeSMDPQLearner:
    """Tabular Q-learner with SMDP-style multi-step updates.

    The table is indexed by (AbstractState key, HighSubtask id). A
    single table is shared across agents (homogeneous policy).
    """

    def __init__(self, cfg: QLearnerConfig, n_subtasks: int, rng_seed: int = 0):
        self.cfg = cfg
        self.n_subtasks = int(n_subtasks)
        self.q: dict[Key, np.ndarray] = {}
        self._rng = random.Random(rng_seed)
        self._episode = 0

    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        c = self.cfg
        if c.epsilon_decay_episodes <= 0:
            return c.epsilon_end
        frac = min(1.0, self._episode / c.epsilon_decay_episodes)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * frac

    def on_episode_end(self) -> None:
        self._episode += 1

    # ------------------------------------------------------------------

    def _row(self, key: Key) -> np.ndarray:
        row = self.q.get(key)
        if row is None:
            row = np.full(self.n_subtasks, float(self.cfg.optimistic_init))
            self.q[key] = row
        return row

    def _prepare_key(self, s: AbstractState) -> Key:
        if self.cfg.cooperative:
            return _state_key(s)
        return (int(s.carrying), int(s.closest_free_slot), tuple())

    # ------------------------------------------------------------------

    def select(
        self,
        s: AbstractState,
        legal: list[HighSubtask],
        greedy: bool = False,
    ) -> HighSubtask:
        """Epsilon-greedy selection restricted to ``legal``.

        Ties are broken randomly (stable via the learner's RNG). When
        ``greedy=True`` (evaluation mode) we never take the epsilon
        branch even if still annealing.
        """
        assert legal, "legal subtask set is empty"
        key = self._prepare_key(s)
        row = self._row(key)
        if (not greedy) and self._rng.random() < self.epsilon:
            return self._rng.choice(legal)
        # Greedy over legal set.
        legal_ids = [int(x) for x in legal]
        legal_vals = [float(row[i]) for i in legal_ids]
        best_val = max(legal_vals)
        best_actions = [a for a, v in zip(legal, legal_vals) if v >= best_val - 1e-9]
        return self._rng.choice(best_actions)

    # ------------------------------------------------------------------

    def smdp_update(
        self,
        s_start: AbstractState,
        action: HighSubtask,
        reward_trace: list[float],
        s_end: Optional[AbstractState],
        legal_end: list[HighSubtask],
        terminal: bool,
    ) -> float:
        """Apply a single SMDP update for an option that just terminated.

        ``reward_trace`` is the list of per-primitive-step rewards the
        option collected from start to end (inclusive of the
        terminating step). ``s_end`` is the abstract state after the
        option finished; ``legal_end`` is the option set legal there;
        ``terminal`` is True if the episode ended during the option.

        Returns the TD error for logging.
        """
        gamma = self.cfg.gamma
        g = 0.0
        for i, r in enumerate(reward_trace):
            g += (gamma ** i) * float(r)
        k = len(reward_trace)

        if not terminal and s_end is not None and legal_end:
            next_key = self._prepare_key(s_end)
            next_row = self._row(next_key)
            next_legal_ids = [int(x) for x in legal_end]
            next_val = max(float(next_row[i]) for i in next_legal_ids)
        else:
            next_val = 0.0

        target = g + (gamma ** k) * next_val
        key = self._prepare_key(s_start)
        row = self._row(key)
        a = int(action)
        td = target - float(row[a])
        row[a] = float(row[a]) + self.cfg.alpha * td
        return float(td)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "config": self.cfg.__dict__,
            "n_subtasks": self.n_subtasks,
            "episode": self._episode,
            # JSON-compatible serialization of the Q dict.
            "q": [
                {"key": list(k), "values": list(map(float, v))}
                for k, v in self.q.items()
            ],
        }
        p.write_text(json.dumps(blob))

    @classmethod
    def load(cls, path: str) -> "CooperativeSMDPQLearner":
        p = Path(path)
        blob = json.loads(p.read_text())
        cfg = QLearnerConfig(**blob["config"])
        inst = cls(cfg, n_subtasks=int(blob["n_subtasks"]))
        inst._episode = int(blob.get("episode", 0))
        for entry in blob["q"]:
            key_raw = entry["key"]
            key = (
                int(key_raw[0]),
                int(key_raw[1]),
                tuple(int(v) for v in key_raw[2]),
            )
            inst.q[key] = np.asarray(entry["values"], dtype=float)
        return inst

    # Introspection helpers --------------------------------------------

    def table_size(self) -> int:
        return len(self.q)


__all__ = ["QLearnerConfig", "CooperativeSMDPQLearner"]
