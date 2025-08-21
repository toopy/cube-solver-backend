import os
from dataclasses import dataclass
from typing import Dict

from fastapi import FastAPI

from .solvers import (
    CubeSolver,
    DeepCubeASolver,
    DQNSolver,
    MagicSolver,
    MuzeroSolver,
    Solver,
)

BACKENDS = {
    "cube-solver": CubeSolver,
    "deepcube-solver": DeepCubeASolver,
    "dqn-solver": DQNSolver,
    "magic-solver": MagicSolver,
    "muzero-solver": MuzeroSolver,
}

DEFAULT_BACKEND = os.getenv("DEFAULT_BACKEND", "dqn-solver")


@dataclass
class BackendSpec:
    class_name: str
    type: str
    weights_path: str
    device: str


class Registry:
    def __init__(self):
        self.backends: Dict[str, Solver] = {}
        self.default_backend: str = DEFAULT_BACKEND

    def load_all(self):
        for name, cls in BACKENDS.items():
            predictor: Solver = cls()
            predictor.load()
            self.backends[name] = predictor

    def get(self, name: str) -> Solver:
        return self.backends[name]


def init_registry(app: FastAPI):
    app._registry = Registry()
    app._registry.load_all()


def get_registry(app: FastAPI) -> Registry:
    if app._registry is None:
        raise RuntimeError("Registry non initialisé. Appel manquant à init_registry().")
    return app._registry
