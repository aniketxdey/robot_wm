"""World-model adapters.

Each adapter wraps a checkpoint (or a synthetic model) behind the uniform
``afp.rollout.WorldModelAdapter`` interface so AFP can score it.

- ``mock``       : deterministic synthetic world models with *tunable, known*
                   action- and instruction-sensitivity. Used to validate the
                   metrics end-to-end on CPU (no GPU, no downloads).
- ``cosmos_sac`` : NVIDIA Cosmos-Predict2-2B-Sample-Action-Conditioned (PRIMARY).
- ``irasim``     : IRASim trajectory->video (spread model).
- ``ctrl_world`` : Ctrl-World (spread model, DROID domain).

Real-checkpoint adapters guard ``available()`` and raise a helpful message until
weights are downloaded and licenses accepted.
"""

from afp.adapters.mock import MockWorldModel, make_mock_suite  # noqa: F401

__all__ = ["MockWorldModel", "make_mock_suite"]
