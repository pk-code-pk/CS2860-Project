"""
Paper-faithful hierarchical multi-agent RL baseline for RWARE.

Implements a two-level Cooperative-HRL controller in the spirit of

    Ghavamzadeh, Mahadevan, Makar.
    "Hierarchical Multi-Agent Reinforcement Learning."

See ``policies/hierarchical/README.md`` for the explicit mapping between the
paper's Coop-HRL / COM-Coop-HRL algorithmic components and this
implementation, plus the list of approximations and omissions.
"""

from .subtasks import (
    HighSubtask,
    AbstractState,
    build_abstract_state,
    subtask_terminated,
    available_cooperative_subtasks,
)
from .low_level import LowLevelExecutor
from .controller import HierarchicalController, ControllerConfig
from .learner import CooperativeSMDPQLearner, QLearnerConfig
from .runner import HRLRunner, HRLRolloutStats

__all__ = [
    "HighSubtask",
    "AbstractState",
    "build_abstract_state",
    "subtask_terminated",
    "available_cooperative_subtasks",
    "LowLevelExecutor",
    "HierarchicalController",
    "ControllerConfig",
    "CooperativeSMDPQLearner",
    "QLearnerConfig",
    "HRLRunner",
    "HRLRolloutStats",
]
