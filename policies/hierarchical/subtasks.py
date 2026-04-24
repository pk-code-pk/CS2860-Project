"""
Cooperative subtask abstraction for the RWARE hierarchy.

Paper correspondence
--------------------
Ghavamzadeh et al.'s Cooperative-HRL decomposes the task into a DAG of
subtasks with primitive actions at the leaves and *cooperative subtasks*
(over which Q-values condition on teammates' choices) at an intermediate
level. For RWARE the natural cooperative level is:

    HANDLE_SLOT_k  -- take care of the k-th request-queue entry
    IDLE           -- explicitly wait / yield
    RECOVER        -- cooperatively cover a slot whose assigned teammate
                      appears to have been dropped (heartbeat stale)

These are coarse, temporally extended options. Each one terminates only
when a well-defined initiation/termination condition fires (shelf picked
up + delivered, or the request queue shifts, or the option was
preempted).

The primitive-action level is left to a hand-coded low-level executor
(see ``low_level.py``) for tractability. This is an approximation of the
paper's tree: the paper's primitive level is also learned, but the
paper's focus is explicitly on the cooperative level, and their
primitive subtasks (Navigate, Put, etc.) are already near-deterministic
given local geometry in the warehouse-like domain they use.

State abstraction
-----------------
A compact per-agent abstract state is used as the Q-learning key:

  * ``carrying``: 0 = unloaded, 1 = carrying requested, 2 = carrying
                  non-requested (needs return)
  * ``closest_free_slot``: index of the request-queue slot that is both
                  (a) not currently claimed by a fresher teammate and
                  (b) closest to this agent, or -1 if none.
  * ``teammate_assignments``: tuple of length n_agents-1 giving each
                  teammate's *currently selected* cooperative subtask
                  as seen by this agent (exact with COM, best-estimate
                  without).

This keeps the Q-table small enough to be actually trainable in a few
thousand episodes on a CPU while still conditioning on teammate
coordination, which is the whole point of the paper's "Cooperative"
qualifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import numpy as np


class HighSubtask(IntEnum):
    """Cooperative / high-level subtask identifiers.

    The first ``MAX_SLOTS`` entries correspond to ``HANDLE_SLOT_k``;
    the remaining two are IDLE and RECOVER. ``MAX_SLOTS`` is a static
    cap so the enum is finite -- environments with fewer request slots
    simply mask the tail (see ``available_cooperative_subtasks``).
    """

    HANDLE_SLOT_0 = 0
    HANDLE_SLOT_1 = 1
    HANDLE_SLOT_2 = 2
    HANDLE_SLOT_3 = 3
    HANDLE_SLOT_4 = 4
    HANDLE_SLOT_5 = 5
    HANDLE_SLOT_6 = 6
    HANDLE_SLOT_7 = 7
    IDLE = 8
    RECOVER = 9


MAX_SLOTS = 8
IDLE = HighSubtask.IDLE
RECOVER = HighSubtask.RECOVER


def is_handle_slot(st: HighSubtask) -> bool:
    return 0 <= int(st) < MAX_SLOTS


def handle_slot_index(st: HighSubtask) -> int:
    assert is_handle_slot(st), f"{st!r} is not a HANDLE_SLOT_* subtask"
    return int(st)


# ---------------------------------------------------------------------------
# Abstract state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstractState:
    """Hashable abstract state used as a Q-table key."""

    carrying: int                       # 0 / 1 / 2
    closest_free_slot: int              # -1 if none
    teammate_assignments: tuple[int, ...]  # length n_agents-1


def _carrying_code(base, agent_idx: int) -> int:
    ag = base.agents[agent_idx]
    shelf = getattr(ag, "carrying_shelf", None)
    if shelf is None:
        return 0
    try:
        requested_ids = {s.id for s in base.request_queue}
    except Exception:
        requested_ids = set()
    return 1 if shelf.id in requested_ids else 2


def _claimed_slot_mask(
    n_slots: int,
    teammate_assignments: tuple[int, ...],
    own_idx_excluded: bool = True,
) -> np.ndarray:
    """Bool mask of length n_slots marking slots claimed by *anyone else*.

    ``teammate_assignments`` is already excluding the agent itself (the
    controller stores its own subtask separately), so every entry of
    the tuple is a teammate's choice.
    """
    del own_idx_excluded  # kept for API symmetry; teammate tuple already excludes self
    mask = np.zeros(n_slots, dtype=bool)
    for v in teammate_assignments:
        if 0 <= v < n_slots:
            mask[v] = True
    return mask


def _closest_free_slot(
    base,
    agent_idx: int,
    n_slots: int,
    teammate_assignments: tuple[int, ...],
) -> int:
    """Return slot index closest to agent among request-queue slots that
    no fresh teammate is currently claiming, or -1 if all slots are
    either missing or already assigned to a teammate."""
    reqs = list(base.request_queue)
    # Pad request_queue entries into slot-indexed view. If the queue is
    # shorter than MAX_SLOTS, trailing slots are treated as empty.
    if not reqs:
        return -1
    claimed = _claimed_slot_mask(n_slots, teammate_assignments)
    ag = base.agents[agent_idx]
    best_d: Optional[int] = None
    best_i = -1
    for i, shelf in enumerate(reqs):
        if i >= n_slots:
            break
        if claimed[i]:
            continue
        d = abs(ag.x - shelf.x) + abs(ag.y - shelf.y)
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return best_i


def build_abstract_state(
    base,
    agent_idx: int,
    teammate_assignments: tuple[int, ...],
    n_slots: int,
) -> AbstractState:
    carrying = _carrying_code(base, agent_idx)
    closest = _closest_free_slot(base, agent_idx, n_slots, teammate_assignments)
    return AbstractState(
        carrying=int(carrying),
        closest_free_slot=int(closest),
        teammate_assignments=tuple(int(v) for v in teammate_assignments),
    )


# ---------------------------------------------------------------------------
# Availability / termination
# ---------------------------------------------------------------------------


def available_cooperative_subtasks(
    base,
    agent_idx: int,
    teammate_assignments: tuple[int, ...],
    n_slots: int,
    any_teammate_presumed_dropped: bool,
) -> list[HighSubtask]:
    """
    Compute the set of cooperative subtasks legal for ``agent_idx`` at a
    boundary, mirroring the paper's notion of initiation conditions
    combined with cooperative deconfliction.

    Rules:
      * IDLE is always available.
      * HANDLE_SLOT_k is available if request_queue has a k-th entry
        AND no other teammate is currently claiming slot k (deconflict).
      * If the agent is already *carrying* a requested shelf (carrying
        code 1), only HANDLE_SLOT_k for that shelf is available -- once
        committed the option must run to delivery. We model this by
        matching the carried shelf's id against request_queue order.
      * RECOVER is available iff some teammate is presumed dropped AND
        that teammate's previously-claimed slot is still open.
    """
    reqs = list(base.request_queue)
    n_req = min(len(reqs), n_slots)
    claimed = _claimed_slot_mask(n_slots, teammate_assignments)

    ag = base.agents[agent_idx]
    carrying_shelf = getattr(ag, "carrying_shelf", None)

    # If already carrying a requested shelf, force HANDLE_SLOT of that shelf.
    if carrying_shelf is not None:
        for i, s in enumerate(reqs[:n_slots]):
            if s.id == carrying_shelf.id:
                return [HighSubtask(i)]

    avail: list[HighSubtask] = [HighSubtask.IDLE]
    for i in range(n_req):
        if not claimed[i]:
            avail.append(HighSubtask(i))

    if any_teammate_presumed_dropped and n_req > 0:
        avail.append(HighSubtask.RECOVER)

    return avail


def subtask_terminated(
    base,
    agent_idx: int,
    subtask: HighSubtask,
    slot_shelf_id_at_start: Optional[int],
    steps_since_start: int,
    max_option_steps: int = 60,
) -> bool:
    """
    Termination test for a running cooperative option. This is the
    ``termination predicate'' in options terminology.

    Termination fires on the earliest of:
      * IDLE: after a single time step (makes the high level re-plan
        often; preserves the paper's SMDP structure).
      * HANDLE_SLOT_k / RECOVER:
          - the *shelf the option committed to* (tracked by id at
            option start) is no longer in the request queue (delivered
            or the queue shifted), OR
          - the agent is not carrying that shelf anymore after having
            been carrying it (task complete), OR
          - the step budget ``max_option_steps`` has elapsed.
      * ``max_option_steps`` hard cap protects us from infinite loops
        when the env cancels moves (e.g. collisions in tiny warehouses).
    """
    if subtask == HighSubtask.IDLE:
        return steps_since_start >= 1

    if steps_since_start >= max_option_steps:
        return True

    if slot_shelf_id_at_start is None:
        # No shelf bound to this option -- shouldn't normally happen.
        return True

    try:
        requested_ids = {s.id for s in base.request_queue}
    except Exception:
        requested_ids = set()

    if slot_shelf_id_at_start not in requested_ids:
        # Delivered or queue shifted; option done.
        return True

    return False


__all__ = [
    "HighSubtask",
    "MAX_SLOTS",
    "AbstractState",
    "build_abstract_state",
    "is_handle_slot",
    "handle_slot_index",
    "available_cooperative_subtasks",
    "subtask_terminated",
]
