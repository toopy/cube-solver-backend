"""High-level solver orchestration using A* + learned heuristic.

This module wires the DeepCubeA-style heuristic network into a batched A*
search to solve a 3x3 Rubik's Cube from a given color layout.
"""

import sys
from importlib.resources import (
    as_file,
    files,
)
from pathlib import Path
from typing import (
    Callable,
    List,
)

import numpy as np
import torch

from ..base import Solver
from .astar import (
    AStar,
    get_path,
)
from .environment import (
    Cube3,
    Cube3State,
)
from .heuristic_fn import load_heuristic_fn
from .utils import (
    get_m_cube_from_colors,
    get_state_from_m_cube,
)

BATCH_SIZE: int = 10000

# Mapping from environment move codes to standard cube notation
MOVES_MAPPING: dict[str, str] = {
    "U-1": "U'",
    "U1": "U",
    "D-1": "D'",
    "D1": "D",
    "L-1": "L'",
    "L1": "L",
    "R-1": "R'",
    "R1": "R",
    "B-1": "B'",
    "B1": "B",
    "F-1": "F'",
    "F1": "F",
}


class DeepCubeASolver(Solver):
    """Solver that uses A* search guided by a neural heuristic.

    This class loads a pre-trained model, constructs a heuristic function,
    and performs a bounded A* search to produce a move sequence.
    """

    def load(self) -> None:
        """Load the heuristic model and prepare the solver."""
        self.model = DeepCubeASolver.from_pth_file()

    def solve(self, color_input: List[List[str]]) -> List[str]:
        """Solve a 3x3 cube from a color layout.

        Parameters
        ----------
        color_input : list of list of str
            Facelet colors laid out according to the UI format.

        Returns
        -------
        list of str
            Sequence of moves in standard Singmaster notation (e.g., "U", "R'").
        """
        m_cube = get_m_cube_from_colors(color_input)
        state = Cube3State(np.array(get_state_from_m_cube(m_cube)))

        env = Cube3()
        verbose = False
        weight = 0.6

        # Initialize A* with one instance (the provided state)
        astar = AStar(
            [state],
            env,
            self.model,
            [weight],
        )

        steps = 0

        # Bounded number of A* iterations
        while not min(astar.has_found_goal()) and steps < 100:
            astar.step(
                self.model,
                BATCH_SIZE,
                verbose=verbose,
            )
            steps += 1

        if steps < 100:
            goal_node = astar.get_goal_node_smallest_path_cost(0)
            _, soln, _ = get_path(goal_node)
            action_list: List[str] = [MOVES_MAPPING[env.moves[m]] for m in soln]
        else:
            action_list = []

        return action_list

    @classmethod
    def from_pth_file(
        cls,
        batch_size: int = BATCH_SIZE,
        device: str = "cpu",
        weights_path: str | Path | None = None,
    ) -> Callable[[List], np.ndarray]:
        """Construct a heuristic function from saved weights.

        Parameters
        ----------
        batch_size : int, default BATCH_SIZE
            Batch size to use for neural network inference.
        device : str, default "cpu"
            Torch device string (e.g., "cpu", "cuda").
        weights_path : str or pathlib.Path or None, default None
            Directory containing the saved weights. If ``None``, attempts
            to resolve packaged data, then falls back relative to the module.

        Returns
        -------
        callable
            Heuristic function compatible with the A* implementation.
        """
        device = torch.device(device)
        on_gpu = device != "cpu"

        # 1) Explicitly provided weights path
        if weights_path is not None:
            weights_path = Path(weights_path)

        else:
            # 2) Try importlib.resources (packaged data)
            try:
                # Package containing the class
                pkg_name = cls.__module__.rsplit(".", 1)[0]
                res = files(pkg_name).joinpath("data")
                # as_file handles compressed/packaged resources
                with as_file(res) as p:
                    weights_path = Path(p)

            except Exception:
                # 3) Fallback: path relative to the module file
                mod_file = Path(sys.modules[cls.__module__].__file__).resolve()
                weights_path = mod_file.parent / "data"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # Environment and model
        env = Cube3()

        heuristic_fn = load_heuristic_fn(
            weights_path,
            device,
            on_gpu,
            env.get_nnet_model(),
            env,
            clip_zero=True,
            batch_size=batch_size,
        )

        return heuristic_fn
