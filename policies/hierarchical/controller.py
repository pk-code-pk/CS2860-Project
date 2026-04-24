"""
Per-agent hierarchical controller.

A ``HierarchicalController`` wraps the shared cooperative Q-learner and
the hand-coded low-level executor into one object per environment
(there is exactly one team controller; agents are homogeneous so they
share the learner). It is responsible for:

  * maintaining each agent's currently running cooperative subtask
    (the option currently in execution), including the shelf id it is
    committed to and the number of primitive steps it has taken;
  * computing abstract state keys at option boundaries and invoking the
    learner's ``select`` method;
  * issuing SMDP updates when an option terminates, carrying along the
    reward trace collected during that option's execution;
  * maintaining a shared "last known teammate assignment" table, whose
    freshness depends on the comm variant:
        - COM on:  updated every time any agent selects a new subtask
                   (broadcast at the boundary)
        - COM off: updated only when the observing agent itself replans
                   (own-stale-estimate).

The controller is deliberately *not* an nn.Module: the high level is
tabular (matches the paper) and the low level is hand-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .low_level import LowLevelExecutor
from .learner import CooperativeSMDPQLearner
from .subtasks import (
    AbstractState,
    HighSubtask,
    MAX_SLOTS,
    available_cooperative_subtasks,
    build_abstract_state,
    handle_slot_index,
    is_handle_slot,
    subtask_terminated,
)


_A_NOOP = 0


@dataclass
class ControllerConfig:
    """Knobs that are controller-level rather than learner-level."""

    n_agents: int
    request_queue_size: int
    max_option_steps: int = 60
    # Communication variant. If True, teammates' selected subtasks are
    # broadcast at every subtask boundary (COM-Cooperative-HRL). Else
    # each agent only sees its own stale estimate (Cooperative-HRL).
    comm: bool = False
    # Staleness threshold for labeling a teammate as presumed-dropped
    # (enables the RECOVER cooperative subtask). Only used when the env
    # exposes heartbeat ages in info["debug_heartbeat_age"].
    stale_threshold: int = 4


@dataclass
class _OptionState:
    """Per-agent run-time bookkeeping for its current option."""

    subtask: HighSubtask = HighSubtask.IDLE
    shelf_id_at_start: Optional[int] = None
    steps: int = 0
    reward_trace: list[float] = field(default_factory=list)
    # Abstract state at option start (for the SMDP update target).
    s_start: Optional[AbstractState] = None
    # Teammate-assignment snapshot at start (for replayability of keys).
    team_at_start: tuple[int, ...] = ()


class HierarchicalController:
    """Drives one RWARE team using Coop-HRL / COM-Coop-HRL."""

    def __init__(
        self,
        cfg: ControllerConfig,
        learner: CooperativeSMDPQLearner,
    ):
        self.cfg = cfg
        self.learner = learner
        self.low = LowLevelExecutor(n_slots=self._n_slots())
        self._options: list[_OptionState] = [
            _OptionState() for _ in range(cfg.n_agents)
        ]
        # Ground-truth board of what each agent is *actually* currently
        # executing. Updated whenever any agent replans, independent of
        # the comm variant.
        self._broadcasted: np.ndarray = np.full(
            cfg.n_agents, int(HighSubtask.IDLE), dtype=np.int64
        )
        # Per-agent *snapshot* of the broadcast board. Under
        # COM-Cooperative-HRL this gets refreshed to the ground truth
        # at every single boundary (so teammates always read the live
        # assignments). Under plain Cooperative-HRL without comm it is
        # only refreshed when *this specific agent* replans -- i.e.
        # every agent sees its teammates' choices only as of its own
        # last replan, which is the paper's "no communication" regime.
        self._view: np.ndarray = np.full(
            (cfg.n_agents, cfg.n_agents), int(HighSubtask.IDLE), dtype=np.int64
        )

    # ------------------------------------------------------------------

    def _n_slots(self) -> int:
        return min(int(self.cfg.request_queue_size), MAX_SLOTS)

    def reset(self) -> None:
        self._options = [_OptionState() for _ in range(self.cfg.n_agents)]
        self._broadcasted = np.full(
            self.cfg.n_agents, int(HighSubtask.IDLE), dtype=np.int64
        )
        self._view = np.full(
            (self.cfg.n_agents, self.cfg.n_agents), int(HighSubtask.IDLE),
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Main per-step API

    def step(
        self,
        base,
        *,
        alive: np.ndarray,
        heartbeat_ages: np.ndarray,
        last_reward: Optional[np.ndarray],
        training: bool,
    ) -> np.ndarray:
        """Return primitive joint action ``(n_agents,)`` for this step.

        ``last_reward`` is the per-agent reward vector produced by the
        *previous* env step (None on the first step of an episode),
        used to append to each agent's in-progress option reward trace.
        """
        n = self.cfg.n_agents

        if last_reward is not None:
            for i in range(n):
                # Credit only live agents whose option is currently running.
                if not alive[i]:
                    continue
                self._options[i].reward_trace.append(float(last_reward[i]))

        # Identify which agents need to replan on this step. Two cases:
        #   (a) first step (steps == 0 and subtask defaulted to IDLE),
        #   (b) the current option's termination predicate fires.
        actions = np.full(n, _A_NOOP, dtype=np.int64)

        for i in range(n):
            if not alive[i]:
                # Dead agents NOOP; keep their option state so SMDP
                # bookkeeping can still be closed out at episode end.
                continue

            opt = self._options[i]
            need_replan = self._should_replan(base, i, opt)

            if need_replan:
                # Close out the running option (if it actually ran).
                if opt.steps > 0 and opt.s_start is not None:
                    self._finalize_option(
                        base, i, opt, heartbeat_ages, terminal=False, training=training
                    )
                self._start_new_option(base, i, heartbeat_ages, training=training)

            opt = self._options[i]
            prim = self.low.act(
                base, i, opt.subtask, opt.shelf_id_at_start
            )
            opt.steps += 1
            actions[i] = int(prim)

        return actions

    # ------------------------------------------------------------------
    # Episode boundary

    def on_episode_end(
        self,
        base,
        *,
        last_reward: Optional[np.ndarray],
        heartbeat_ages: np.ndarray,
        alive: np.ndarray,
        training: bool,
    ) -> None:
        """Close out every in-flight option with the terminal SMDP update."""
        if last_reward is not None:
            for i in range(self.cfg.n_agents):
                if alive[i]:
                    self._options[i].reward_trace.append(float(last_reward[i]))
        for i in range(self.cfg.n_agents):
            opt = self._options[i]
            if opt.steps > 0 and opt.s_start is not None and training:
                self._finalize_option(
                    base, i, opt, heartbeat_ages, terminal=True, training=True
                )
        self.learner.on_episode_end()
        self.reset()

    # ------------------------------------------------------------------
    # Internal: replan / finalize

    def _should_replan(self, base, agent_idx: int, opt: _OptionState) -> bool:
        if opt.steps == 0 and opt.s_start is None:
            # No option has ever been started for this agent yet.
            return True
        return subtask_terminated(
            base,
            agent_idx=agent_idx,
            subtask=opt.subtask,
            slot_shelf_id_at_start=opt.shelf_id_at_start,
            steps_since_start=opt.steps,
            max_option_steps=self.cfg.max_option_steps,
        )

    def _teammate_view(self, agent_idx: int) -> tuple[int, ...]:
        """The tuple of teammate assignments visible to ``agent_idx``.

        Freshness depends on the comm variant:
          * COM on: refresh this agent's row from the live broadcast
            board right now, so we return the up-to-date ground truth.
          * COM off: we return this agent's *own last snapshot* of the
            board -- stale, updated only when this agent itself replans.
        """
        if self.cfg.comm:
            self._view[agent_idx, :] = self._broadcasted
        row = self._view[agent_idx]
        return tuple(int(row[j]) for j in range(self.cfg.n_agents) if j != agent_idx)

    def _build_state(
        self, base, agent_idx: int, teammate_tuple: tuple[int, ...]
    ) -> AbstractState:
        return build_abstract_state(
            base,
            agent_idx=agent_idx,
            teammate_assignments=teammate_tuple,
            n_slots=self._n_slots(),
        )

    def _legal_set(
        self,
        base,
        agent_idx: int,
        teammate_tuple: tuple[int, ...],
        heartbeat_ages: np.ndarray,
    ) -> list[HighSubtask]:
        presumed_dropped = bool(np.any(heartbeat_ages > self.cfg.stale_threshold))
        return available_cooperative_subtasks(
            base,
            agent_idx=agent_idx,
            teammate_assignments=teammate_tuple,
            n_slots=self._n_slots(),
            any_teammate_presumed_dropped=presumed_dropped,
        )

    def _start_new_option(
        self,
        base,
        agent_idx: int,
        heartbeat_ages: np.ndarray,
        training: bool,
    ) -> None:
        teammate_tuple = self._teammate_view(agent_idx)
        s = self._build_state(base, agent_idx, teammate_tuple)
        legal = self._legal_set(base, agent_idx, teammate_tuple, heartbeat_ages)
        a = self.learner.select(s, legal, greedy=not training)

        shelf_id = self._shelf_id_for_subtask(base, agent_idx, a)

        self._options[agent_idx] = _OptionState(
            subtask=a,
            shelf_id_at_start=shelf_id,
            steps=0,
            reward_trace=[],
            s_start=s,
            team_at_start=teammate_tuple,
        )

        # Commit the new assignment to the ground-truth board.
        self._broadcasted[agent_idx] = int(a)
        # Snapshot the *current* ground truth into this agent's own view.
        # Under COM this view will be refreshed again the next time any
        # *other* agent broadcasts; under no-comm it will only be
        # refreshed the next time this agent itself replans, which is
        # exactly the "stale teammate knowledge" the paper describes
        # for Cooperative-HRL without COM.
        self._view[agent_idx, :] = self._broadcasted

    def _finalize_option(
        self,
        base,
        agent_idx: int,
        opt: _OptionState,
        heartbeat_ages: np.ndarray,
        terminal: bool,
        training: bool,
    ) -> None:
        if not training:
            # Evaluation-only: we still close bookkeeping but skip the
            # learner update.
            return
        teammate_tuple_end = self._teammate_view(agent_idx)
        s_end = self._build_state(base, agent_idx, teammate_tuple_end) if not terminal else None
        legal_end = (
            self._legal_set(base, agent_idx, teammate_tuple_end, heartbeat_ages)
            if not terminal
            else []
        )
        self.learner.smdp_update(
            s_start=opt.s_start,  # type: ignore[arg-type]
            action=opt.subtask,
            reward_trace=list(opt.reward_trace),
            s_end=s_end,
            legal_end=legal_end,
            terminal=bool(terminal),
        )

    # ------------------------------------------------------------------

    def _shelf_id_for_subtask(
        self, base, agent_idx: int, subtask: HighSubtask
    ) -> Optional[int]:
        """Bind the option to a concrete shelf id at start time.

        For HANDLE_SLOT_k we pick the k-th request_queue entry. For
        RECOVER we bind to the *closest* still-requested shelf. For
        IDLE there is nothing to bind.
        """
        if subtask == HighSubtask.IDLE:
            return None

        reqs = list(base.request_queue)
        if subtask == HighSubtask.RECOVER:
            ag = base.agents[agent_idx]
            best, best_d = None, None
            for s in reqs:
                d = abs(s.x - ag.x) + abs(s.y - ag.y)
                if best_d is None or d < best_d:
                    best, best_d = s, d
            return int(best.id) if best is not None else None

        if is_handle_slot(subtask):
            k = handle_slot_index(subtask)
            if 0 <= k < len(reqs):
                return int(reqs[k].id)
        return None


__all__ = ["ControllerConfig", "HierarchicalController"]
