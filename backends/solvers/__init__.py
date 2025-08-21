from .base import Solver
from .deep import DeepCubeASolver
from .dqn import DQNSolver
from .cube import CubeSolver
from .magic import MagicSolver
from .muzero import MuzeroSolver

__all__ = (
    "CubeSolver",
    "DeepCubeASolver",
    "DQNSolver",
    "MagicSolver",
    "Solver",
    "MuzeroSolver",
)
