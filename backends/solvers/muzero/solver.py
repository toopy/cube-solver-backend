"""MuZero-based solver orchestration.

This module loads a pre-trained MuZero checkpoint and leverages its
self-play interface to produce a move sequence from a color layout.
"""

import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, List

import torch
from muzero_baseline.self_play import SelfPlayDirect
from oc_rubik_s_cube_dqn.cube import RubiksCubeEnv
from oc_rubik_s_cube_muzero.config import MuZeroConfig
from oc_rubik_s_cube_muzero.game import Game

from ..base import Solver


class MuzeroSolver(Solver):
    """Solver that uses a MuZero checkpoint to generate a solution."""

    def load(self) -> None:
        """Load the MuZero model from packaged or provided weights."""
        self.model = MuzeroSolver.from_pth_file()

    def solve(self, color_input: List[List[str]]) -> List[str]:
        """Solve a cube instance using MuZero self-play.

        Parameters
        ----------
        color_input : list of list of str
            Facelet colors as provided by the UI.

        Returns
        -------
        list of str
            Move sequence. Returns an empty list if the episode hits
            ``config.max_moves`` without solving.
        """
        config = MuZeroConfig(scramble_moves=10)

        # Create a self-play worker that drives the game using the model
        self_play_worker = SelfPlayDirect(
            self.model,
            Game,
            config,
            42,
            game_kwargs={
                "color_input": color_input,
            },
        )

        self_play_worker.play_game(
            0,
            0,
            False,
            config.opponent,
            config.muzero_player,
        )

        action_list: List[str] = self_play_worker.game.moves

        # ensure really solved !
        cube_env = RubiksCubeEnv()
        cube_env.set_cube_from_colors(color_input)
        done = cube_env.apply_actions(action_list)

        return action_list if done else []

    @classmethod
    def from_pth_file(
        cls,
        weights_path: str | Path | None = None,
    ) -> Any:
        """Instantiate and load a MuZero model from a checkpoint.

        Parameters
        ----------
        weights_path : str or pathlib.Path or None, default None
            Path to ``model.checkpoint``. If ``None``, attempts to resolve a
            packaged resource, then falls back to a path relative to the
            ``Game`` module.

        Returns
        -------
        MuZero
            Loaded MuZero model ready for inference.
        """
        # 1) Explicit weights path
        if weights_path is not None:
            weights_path = Path(weights_path)

        else:
            # 2) Try importlib.resources (packaged data)
            try:
                # Package containing the game implementation
                pkg_name = Game.__module__.rsplit(".", 1)[0]
                res = files(pkg_name).joinpath("data/model.checkpoint")
                # as_file handles packaged/compressed resources
                with as_file(res) as p:
                    weights_path = Path(p)

            except Exception:
                # 3) Fallback: path relative to the module file
                mod_file = Path(sys.modules[Game.__module__].__file__).resolve()
                weights_path = mod_file.parent / "data" / "model.checkpoint"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        model = torch.load(weights_path)

        return model
