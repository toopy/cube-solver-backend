"""Wrapper around the `magiccube` solver.

This module adapts the external `magiccube` solver to the common `Solver`
interface. It builds a cube from UI colors and runs the basic solver.
"""

from typing import List

from magiccube import BasicSolver
from oc_rubik_s_cube_dqn.cube import RubiksCubeEnv

from .base import Solver


class MagicSolver(Solver):
    """Solver that defers to `magiccube.BasicSolver` for a solution."""

    def load(self) -> None:
        """No-op; `magiccube` does not require preloaded weights."""
        pass

    def solve(
        self,
        color_input: List[List[str]],
    ) -> List[str]:
        """Solve using `magiccube` from a color layout.

        Parameters
        ----------
        color_input : list of list of str
            Facelet colors arranged as provided by the UI.

        Returns
        -------
        list of str
            Sequence of moves returned by `magiccube`. Returns an empty list
            if the sequence exceeds a conservative cutoff (50 moves).
        """
        cube_env = RubiksCubeEnv()
        cube_env.set_cube_from_colors(color_input)

        solver = BasicSolver(cube_env.cube)
        actions = [str(a) for a in solver.solve()]

        # Safety cutoff to avoid overly long solutions
        if len(actions) > 50:
            return []

        return actions
