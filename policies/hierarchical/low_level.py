"""
Low-level primitive-action executor for cooperative subtasks.

This is the leaf of the paper's task hierarchy: the code that, given a
*currently selected* cooperative subtask, emits a single primitive
env action each step. In the Ghavamzadeh et al. paper, the primitive
level is also learned Q-wise. We hand-code it here for tractability;
the explicit approximations are documented in the module README.

The motion primitives (greedy step-toward-target, single-tile rotations)
match those already used by the reactive heuristic baseline so that the
*only* thing that differs between heuristic and HRL-paper is the
*coordination / high-level* logic.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .subtasks import HighSubtask, handle_slot_index, is_handle_slot


# Mirror the RWARE Action enum so we don't have to import it (adapter
# avoids the dep too; see baselines/rware_heuristic.py).
_A_NOOP = 0
_A_FORWARD = 1
_A_LEFT = 2
_A_RIGHT = 3
_A_TOGGLE = 4

_D_UP = 0
_D_DOWN = 1
_D_LEFT = 2
_D_RIGHT = 3


class LowLevelExecutor:
    """Stateless-per-step primitive executor.

    Each call to :meth:`act` returns an int (the RWARE action index) for
    the specified agent, given its currently running cooperative
    subtask. The high level is responsible for deciding when a subtask
    terminates (see ``subtasks.subtask_terminated``).
    """

    def __init__(self, n_slots: int):
        self.n_slots = int(n_slots)

    # ------------------------------------------------------------------

    def act(
        self,
        base,
        agent_idx: int,
        subtask: HighSubtask,
        slot_shelf_id: Optional[int],
    ) -> int:
        """Pick a primitive action for this agent executing ``subtask``."""
        if subtask == HighSubtask.IDLE:
            return _A_NOOP

        ag = base.agents[agent_idx]

        # Resolve the shelf the option is currently committed to.
        shelf = self._lookup_shelf(base, slot_shelf_id)
        if shelf is None and is_handle_slot(subtask):
            # Try using the slot index to recover the shelf if the id is stale.
            reqs = list(base.request_queue)
            si = handle_slot_index(subtask)
            if 0 <= si < len(reqs):
                shelf = reqs[si]

        # Subtask = RECOVER: pick up *any* still-pending request. We treat
        # it operationally as "head to the closest requested shelf that is
        # not currently at a goal". This mirrors the paper's cooperative
        # takeover -- a teammate's old claim is up for grabs.
        if subtask == HighSubtask.RECOVER:
            shelf = self._closest_requested_shelf(base, agent_idx)
            if shelf is None:
                return _A_NOOP

        if shelf is None:
            return _A_NOOP

        carrying = getattr(ag, "carrying_shelf", None)
        requested_ids = self._requested_ids(base)

        if carrying is None:
            # Phase 1: navigate to the target shelf cell, then TOGGLE to pick up.
            return _step_toward(ag, (shelf.x, shelf.y))

        if carrying.id == shelf.id:
            # Phase 2: carry to nearest goal, then TOGGLE to deliver.
            goal = _nearest(base.goals, (ag.x, ag.y))
            if goal is None:
                return _A_NOOP
            return _step_toward(ag, goal)

        # We are carrying a shelf that is NOT the one this subtask is
        # about. Two plausible cases:
        #   - agent picked up the wrong shelf earlier -> return it.
        #   - queue shifted mid-option -> high level should preempt soon.
        # Either way, head to the nearest empty shelf slot and drop.
        if carrying.id in requested_ids:
            # Accidentally useful -- deliver it anyway.
            goal = _nearest(base.goals, (ag.x, ag.y))
            if goal is None:
                return _A_NOOP
            return _step_toward(ag, goal)
        drop = self._nearest_empty_shelf_cell(base, (ag.x, ag.y))
        if drop is None:
            return _A_NOOP
        return _step_toward(ag, drop)

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _requested_ids(base) -> set[int]:
        try:
            return {s.id for s in base.request_queue}
        except Exception:
            return set()

    @staticmethod
    def _lookup_shelf(base, shelf_id: Optional[int]):
        if shelf_id is None:
            return None
        try:
            shelves = base.shelfs  # rware's misspelling of shelves
        except Exception:
            return None
        # shelves is a flat list, ids are 1-indexed.
        for s in shelves:
            if s.id == shelf_id:
                return s
        return None

    def _closest_requested_shelf(self, base, agent_idx: int):
        ag = base.agents[agent_idx]
        best = None
        best_d = None
        for s in base.request_queue:
            d = abs(s.x - ag.x) + abs(s.y - ag.y)
            if best_d is None or d < best_d:
                best, best_d = s, d
        return best

    def _nearest_empty_shelf_cell(self, base, pos):
        try:
            shelf_layer = base.grid[1]
        except Exception:
            return None
        highways = getattr(base, "highways", None)
        H, W = shelf_layer.shape
        px, py = pos
        best = None
        best_d = None
        for yy in range(H):
            for xx in range(W):
                if shelf_layer[yy, xx] != 0:
                    continue
                if highways is not None:
                    try:
                        if highways[yy, xx] != 0:
                            continue
                    except Exception:
                        pass
                d = abs(xx - px) + abs(yy - py)
                if best_d is None or d < best_d:
                    best, best_d = (xx, yy), d
        return best


# ---------------------------------------------------------------------------
# Motion primitives (duplicated from the heuristic baseline so that the
# hierarchical baseline does not depend on it). Kept byte-compatible.
# ---------------------------------------------------------------------------


def _nearest(cells, pos):
    px, py = pos
    best = None
    best_d = None
    for c in cells:
        cx, cy = c
        d = abs(cx - px) + abs(cy - py)
        if best_d is None or d < best_d:
            best, best_d = (cx, cy), d
    return best


def _step_toward(agent, target) -> int:
    tx, ty = target
    dx = tx - agent.x
    dy = ty - agent.y
    if dx == 0 and dy == 0:
        return _A_TOGGLE
    if abs(dx) >= abs(dy):
        want_dir = _D_RIGHT if dx > 0 else _D_LEFT
    else:
        want_dir = _D_DOWN if dy > 0 else _D_UP
    cur_dir = _dir_to_int(agent.dir)
    if cur_dir == want_dir:
        return _A_FORWARD
    return _rotation_action(cur_dir, want_dir)


def _dir_to_int(d) -> int:
    if isinstance(d, int):
        return d
    v = getattr(d, "value", None)
    if v is not None:
        return int(v)
    name = str(d).split(".")[-1].upper()
    return {"UP": _D_UP, "DOWN": _D_DOWN, "LEFT": _D_LEFT, "RIGHT": _D_RIGHT}.get(
        name, 0
    )


def _rotation_action(cur: int, want: int) -> int:
    if cur == want:
        return _A_FORWARD
    cw = {
        _D_UP: _D_RIGHT,
        _D_RIGHT: _D_DOWN,
        _D_DOWN: _D_LEFT,
        _D_LEFT: _D_UP,
    }
    if cw[cur] == want:
        return _A_RIGHT
    return _A_LEFT


__all__ = ["LowLevelExecutor"]
