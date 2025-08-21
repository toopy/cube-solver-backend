"""DQN-based solver orchestration.

This module loads a pre-trained DQN and greedily selects actions from the
current state until the cube is solved or a step limit is reached.
"""

from typing import List

import numpy as np
import torch
from oc_rubik_s_cube_dqn.cube import (
    INDEX_TO_MOVE,
    RubiksCubeEnv,
)
from oc_rubik_s_cube_dqn.model import DQN

from .base import Solver


class DQNSolver(Solver):
    """Solver that uses a DQN policy to pick greedy actions."""

    def load(self) -> None:
        """Load the DQN model from packaged weights."""
        self.model = DQN.from_pth_file()

    def solve(self, color_input: List[List[str]]) -> List[str]:
        """Solve a cube instance by greedy DQN action selection.

        Parameters
        ----------
        color_input : list of list of str
            Facelet colors as provided by the UI.

        Returns
        -------
        list of str
            Sequence of moves taken. Returns an empty list on timeout.
        """
        cube_env = RubiksCubeEnv()
        cube_env.set_cube_from_colors(color_input)

        # TODO: make this configurable
        scramble_moves: int = 5

        action_list: List[str] = []
        done: bool = False
        # Encode the cube to a numpy array
        state: np.ndarray = cube_env._get_state()
        steps: int = 0

        while not done and steps < scramble_moves * 3:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.long,
                    device="cpu",
                ).unsqueeze(0)
                q_values: torch.Tensor = self.model(state_tensor)
                action: int = torch.argmax(q_values).item()

            action_str = INDEX_TO_MOVE[action]
            action_list.append(action_str)

            state, _, done = cube_env.step(action)
            steps += 1

        return action_list if done else []
