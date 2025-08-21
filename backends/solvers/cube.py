"""Lightweight cube state utilities and a simple solver.

This module defines a compact cube representation, helpers to parse a
color-based input into that representation, and a bidirectional BFS solver
for the classic layer-by-layer-style subproblems.
"""

from typing import List, TypedDict

from .base import Solver

CubeState = TypedDict(
    "CubeState",
    {
        "corner_position": list[int],
        "corner_orientation": list[int],
        "edge_position": list[int],
        "edge_orientation": list[int],
        "parity": bool,
    },
)


def parse_color_input(color_input: List[List[str]]) -> List[str]:
    """Map UI color input to cubie face-label strings.

    Parameters
    ----------
    color_input : list of list of str
        Colors per cubie in UI order. Each inner list is ordered by axis and
        matched to faces via ``_face_to_axis``.

    Returns
    -------
    list of str
        For edges, a 2-letter face label (e.g., "UF"); for corners, a
        3-letter label (e.g., "UFR"), ordered to match ``_goal``.
    """
    output = []
    for i, (ref, cubie) in enumerate(zip(_goal, color_input)):
        if i < 12:
            face1 = ref[0]
            face2 = ref[1]

            color1 = cubie[_face_to_axis[face1]]
            color2 = cubie[_face_to_axis[face2]]

            final_face1 = _color_to_face[color1]
            final_face2 = _color_to_face[color2]

            output.append(final_face1 + final_face2)
        else:
            face1 = ref[0]
            face2 = ref[1]
            face3 = ref[2]

            color1 = cubie[_face_to_axis[face1]]
            color2 = cubie[_face_to_axis[face2]]
            color3 = cubie[_face_to_axis[face3]]

            final_face1 = _color_to_face[color1]
            final_face2 = _color_to_face[color2]
            final_face3 = _color_to_face[color3]

            output.append(final_face1 + final_face2 + final_face3)
    return output


def build_cube_state(scramble: List[str]) -> CubeState:
    """Build a cube state from edge/corner labels.

    Parameters
    ----------
    scramble : list of str
        Edge labels first (12), followed by corner labels (8). Labels are
        face-letter sequences like "UF" or "URB".

    Returns
    -------
    CubeState
        Structured cube state with positions, orientations, and parity.

    Raises
    ------
    Exception
        If a cubie label cannot be rotated to match any goal orientation.
    """
    position = [0] * 20
    orientation = [0] * 20

    for i, cubie in enumerate(scramble):
        while cubie not in _goal:
            cubie = cubie[1:] + cubie[0]
            orientation[i] += 1
            if orientation[i] == 3:
                raise Exception("Invalid scramble")
        position[i] = _goal.index(cubie) if i < 12 else _goal.index(cubie) - 12

    corner_parity = False
    for i in range(8):
        for j in range(i + 1, 8):
            corner_parity ^= position[12 + i] > position[12 + j]

    return CubeState(
        {
            "corner_position": position[12:],
            "corner_orientation": orientation[12:],
            "edge_position": position[:12],
            "edge_orientation": orientation[:12],
            "parity": corner_parity,
        }
    )


def inverse_move(move: str) -> str:
    """Return the inverse of a given move string."""
    return _inverse_move[move]


def legal_moves(phase: int) -> List[str]:
    """Legal moves by solving phase.

    Parameters
    ----------
    phase : int
        Phase index in {1, 2, 3, 4}. Earlier phases allow more moves; later
        phases restrict to half-turns to preserve progress.

    Returns
    -------
    list of str
        Allowed moves for the specified phase.
    """
    if phase == 2:
        return [
            "U",
            "U'",
            "U2",
            "D",
            "D'",
            "D2",
            "F2",
            "B2",
            "L",
            "L'",
            "L2",
            "R",
            "R'",
            "R2",
        ]
    if phase == 3:
        return ["U", "U'", "U2", "D", "D'", "D2", "F2", "B2", "L2", "R2"]
    if phase == 4:
        return ["U2", "D2", "F2", "B2", "L2", "R2"]
    return [
        "U",
        "U'",
        "U2",
        "D",
        "D'",
        "D2",
        "F",
        "F'",
        "F2",
        "B",
        "B'",
        "B2",
        "L",
        "L'",
        "L2",
        "R",
        "R'",
        "R2",
    ]


class Cube:
    """Minimal cube state with turn and phase-ID utilities."""
    solved_state = CubeState(
        {
            "corner_position": [0, 1, 2, 3, 4, 5, 6, 7],
            "corner_orientation": [0, 0, 0, 0, 0, 0, 0, 0],
            "edge_position": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "edge_orientation": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "parity": False,
        }
    )

    def __init__(self, init_state: CubeState = solved_state) -> None:
        """Initialize from a provided state or the solved state."""
        self.corner_position = init_state["corner_position"]
        self.corner_orientation = init_state["corner_orientation"]

        self.edge_position = init_state["edge_position"]
        self.edge_orientation = init_state["edge_orientation"]
        self.parity = init_state["parity"]

    def __str__(self) -> str:
        """Human-readable summary of the cube state."""
        return (
            "Corner Position: "
            + str(self.corner_position)
            + "\n"
            + "Corner Orientation: "
            + str(self.corner_orientation)
            + "\n"
            + "Edge Position: "
            + str(self.edge_position)
            + "\n"
            + "Edge Orientation: "
            + str(self.edge_orientation)
            + "\n"
            + "Parity: "
            + str(self.parity)
        )

    def get_cube_state(self) -> CubeState:
        """Return a shallow copy of the current state as a ``CubeState``."""
        return CubeState(
            {
                "corner_position": self.corner_position[:],
                "corner_orientation": self.corner_orientation[:],
                "edge_position": self.edge_position[:],
                "edge_orientation": self.edge_orientation[:],
                "parity": self.parity,
            }
        )

    # Each cube permutation can be assigned an id depending on the phase
    # The id is not unique for each permutation, but permutations with matching ids are equivalent
    # for the subproblem being solved at the given phase
    def phase_id(self, phase: int) -> str:
        """Return an equivalence-class ID for the given solving phase.

        The ID collapses states that are equivalent for the subproblem solved
        in the specified phase.
        """
        if phase == 1:
            return str(self.edge_orientation)
        elif phase == 2:
            id = str(self.corner_orientation)
            equitorial_slice_index = str(
                ["a" if edge > 7 else "b" for edge in self.edge_position]
            )
            return " ".join([str(id), equitorial_slice_index])
        elif phase == 3:
            middle_standing_slice_index = str(
                [
                    "a" if edge > 7 else "b" if edge % 2 == 0 else "c"
                    for edge in self.edge_position
                ]
            )
            corner_pairing_index = str([corner & 5 for corner in self.corner_position])
            parity = str(self.parity)

            return " ".join([middle_standing_slice_index, corner_pairing_index, parity])
        return str(
            [
                x
                for x in self.corner_position
                + self.corner_orientation
                + self.edge_position
                + self.edge_orientation
            ]
        )

    def turn(self, turn: str) -> "Cube":
        """Apply a move and return a new cube state.

        Parameters
        ----------
        turn : str
            Move string (e.g., "U", "R'", "F2").

        Returns
        -------
        Cube
            New cube after applying ``turn``.
        """
        next_cube = Cube(self.get_cube_state())
        face, turn = _str_to_turn_tuple[turn]

        # Moves the corners and edges of the cube into position
        def turnface(corners, edges, turn):
            for i in range(4):
                if turn == 0:
                    j = (i + 1) % 4
                if turn == 1:
                    j = i - 1
                if turn == 2:
                    j = (i + 2) % 4

                next_cube.corner_position[corners[i]] = self.corner_position[corners[j]]
                next_cube.corner_orientation[corners[i]] = self.corner_orientation[
                    corners[j]
                ]
                next_cube.edge_position[edges[i]] = self.edge_position[edges[j]]
                next_cube.edge_orientation[edges[i]] = self.edge_orientation[edges[j]]

        def orientedges(edges, turn):
            if turn == 0 or turn == 1:
                next_cube.edge_orientation[edges[0]] = (
                    next_cube.edge_orientation[edges[0]] + 1
                ) % 2
                next_cube.edge_orientation[edges[1]] = (
                    next_cube.edge_orientation[edges[1]] + 1
                ) % 2
                next_cube.edge_orientation[edges[2]] = (
                    next_cube.edge_orientation[edges[2]] + 1
                ) % 2
                next_cube.edge_orientation[edges[3]] = (
                    next_cube.edge_orientation[edges[3]] + 1
                ) % 2

        def orientcorners(corners, turn):
            if turn == 0 or turn == 1:
                next_cube.corner_orientation[corners[0]] = (
                    next_cube.corner_orientation[corners[0]] - 1
                ) % 3
                next_cube.corner_orientation[corners[1]] = (
                    next_cube.corner_orientation[corners[1]] + 1
                ) % 3
                next_cube.corner_orientation[corners[2]] = (
                    next_cube.corner_orientation[corners[2]] - 1
                ) % 3
                next_cube.corner_orientation[corners[3]] = (
                    next_cube.corner_orientation[corners[3]] + 1
                ) % 3

        edges, corners = _face_to_idx[face]
        turnface(corners, edges, turn)
        if face == 2 or face == 3:
            orientedges(edges, turn)
        if face == 2 or face == 3 or face == 4 or face == 5:
            orientcorners(corners, turn)

        if turn == 0 or turn == 1:
            next_cube.parity = not self.parity

        return next_cube


_face_to_idx: List[tuple[List[int], List[int]]] = [
    ([0, 1, 2, 3], [0, 1, 2, 3]),
    ([4, 7, 6, 5], [4, 5, 6, 7]),
    ([0, 9, 4, 8], [0, 3, 5, 4]),
    ([2, 10, 6, 11], [2, 1, 7, 6]),
    ([3, 11, 7, 9], [3, 2, 6, 5]),
    ([1, 8, 5, 10], [1, 0, 4, 7]),
]

_inverse_move: dict[str, str] = {
    "U": "U'",
    "U'": "U",
    "U2": "U2",
    "D": "D'",
    "D'": "D",
    "D2": "D2",
    "F": "F'",
    "F'": "F",
    "F2": "F2",
    "B": "B'",
    "B'": "B",
    "B2": "B2",
    "L": "L'",
    "L'": "L",
    "L2": "L2",
    "R": "R'",
    "R'": "R",
    "R2": "R2",
}

_str_to_turn_tuple: dict[str, tuple[int, int]] = {
    "U": (0, 0),
    "U'": (0, 1),
    "U2": (0, 2),
    "D": (1, 0),
    "D'": (1, 1),
    "D2": (1, 2),
    "F": (2, 0),
    "F'": (2, 1),
    "F2": (2, 2),
    "B": (3, 0),
    "B'": (3, 1),
    "B2": (3, 2),
    "L": (4, 0),
    "L'": (4, 1),
    "L2": (4, 2),
    "R": (5, 0),
    "R'": (5, 1),
    "R2": (5, 2),
}

_color_to_face: dict[str, str] = {
    "blue": "U",
    "green": "D",
    "yellow": "F",
    "white": "B",
    "red": "L",
    "orange": "R",
}

_face_to_axis: dict[str, int] = {"U": 1, "D": 1, "F": 2, "B": 2, "L": 0, "R": 0}

_goal: List[str] = [
    "UF",
    "UR",
    "UB",
    "UL",
    "DF",
    "DR",
    "DB",
    "DL",
    "FR",
    "FL",
    "BR",
    "BL",
    "UFR",
    "URB",
    "UBL",
    "ULF",
    "DRF",
    "DFL",
    "DLB",
    "DBR",
]


class CubeSolver(Solver):
    """Basic solver using bidirectional BFS over phase-reduced state IDs."""

    def load(self) -> None:
        """No-op for the basic solver (no model to load)."""

    # Bidirectional breadth-first search
    # Returns a string of moves that solves the cube
    # The cube remains unchanged
    def solve(
        self,
        color_input: List[List[str]],
    ) -> List[str]:
        """Solve from a color layout using phase-wise bidirectional BFS.

        Parameters
        ----------
        color_input : list of list of str
            Facelet colors per cubie in UI order.

        Returns
        -------
        list of str
            Sequence of moves that solves the cube.
        """
        color_data = parse_color_input(color_input)
        cube_state = build_cube_state(color_data)

        cube = Cube(cube_state)

        goal_cube = Cube()
        phase = 0

        output = []

        while (phase := phase + 1) < 5:
            start_id = cube.phase_id(phase)
            goalId = goal_cube.phase_id(phase)

            if start_id == goalId:
                continue

            queue = [cube, goal_cube]

            visited = {
                start_id: (1, None, None),
                goalId: (2, None, None),
            }  # Map of seen ids to (direction, predecessor, last move)
            found = False

            while not found:
                old_cube = queue.pop(0)
                old_id = old_cube.phase_id(phase)

                for move in legal_moves(phase):
                    new_cube = old_cube.turn(move)
                    new_id = new_cube.phase_id(phase)

                    if new_id not in visited:
                        visited[new_id] = (visited[old_id][0], old_id, move)
                        queue.append(new_cube)

                    elif visited[new_id][0] != visited[old_id][0]:
                        found = True

                        # Realign so the current node is always in the forward direction
                        if visited[old_id][0] == 2:
                            old_id, new_id = new_id, old_id
                            move = inverse_move(move)

                        # Reconstruct the algorithm from the middle out
                        algorithm = [move]

                        # Reconstruct the portion that precedes the current node
                        while old_id != start_id:
                            algorithm.insert(0, visited[old_id][2])
                            old_id = visited[old_id][1]

                        # Reconstruct the portion that follows the current node
                        while new_id != goalId:
                            algorithm.append(inverse_move(visited[new_id][2]))
                            new_id = visited[new_id][1]

                        for move in algorithm:
                            output.append(move)
                            cube = cube.turn(move)

                        break
        return output
