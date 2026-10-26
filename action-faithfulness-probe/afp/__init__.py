"""AFP -- Action-Faithfulness Probe.

A training-free, judge-free, horizon-resolved benchmark that measures whether an
action-conditioned video world model's predicted future actually responds to the
action it is given (action faithfulness) vs. merely repainting the task from the
instruction (instruction reliance).

Public API is assembled lazily so that importing a single submodule (e.g. for a
subagent building one piece) does not require the whole package to be present.
"""

from afp.rollout import Episode, Rollout, WorldModelAdapter, ACTION_DIM, MOTION_DIMS, GRIPPER_DIM
from afp.distance import get_distance, available_backends
from afp.data import make_synthetic_dataset, load_episodes, save_episodes, embed_instruction
from afp.adapters.mock import MockWorldModel, make_mock_suite
from afp.score import Scorecard, score_model, score_suite

__version__ = "0.1.0"

__all__ = [
    # contracts
    "Episode",
    "Rollout",
    "WorldModelAdapter",
    "ACTION_DIM",
    "MOTION_DIMS",
    "GRIPPER_DIM",
    # distance
    "get_distance",
    "available_backends",
    # data
    "make_synthetic_dataset",
    "load_episodes",
    "save_episodes",
    "embed_instruction",
    # models
    "MockWorldModel",
    "make_mock_suite",
    # scoring
    "Scorecard",
    "score_model",
    "score_suite",
]
