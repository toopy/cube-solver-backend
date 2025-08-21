from abc import (
    ABC,
    abstractmethod,
)
from random import randrange
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
import torch.nn as nn

from .model import ResnetModel


class State(ABC):
    """Abstract base class for environment states.

    Subclasses must implement hashing and equality to enable use in
    dictionaries/sets and CLOSED lists.
    """

    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash value for the state."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Return True if ``other`` represents the same state."""
        pass


class Cube3State(State):
    """State for a 3x3 Rubik's Cube environment.

    Parameters
    ----------
    colors : numpy.ndarray
        Flattened color indices representing the cube configuration.
    """

    __slots__ = ["colors", "hash"]

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.hash = None

    def __hash__(self) -> int:
        if self.hash is None:
            self.hash = hash(",".join([str(i) for i in self.colors]))

        return self.hash

    def __eq__(self, other) -> bool:  # type: ignore[override]
        return np.array_equal(self.colors, other.colors)


class Environment(ABC):
    """Abstract environment interface for search and learning.

    The environment defines the state space, legal moves, transition costs,
    solved checks, and conversions to neural network inputs.
    """

    def __init__(self) -> None:
        self.dtype = float
        self.fixed_actions: bool = True

    @abstractmethod
    def next_state(
        self, states: List[State], action: int
    ) -> Tuple[List[State], List[float]]:
        """Compute successor states and transition costs for an action.

        Parameters
        ----------
        states : list of State
            Current states to transition from.
        action : int
            Action index to apply to all states.

        Returns
        -------
        list of State
            Next states for each input state.
        list of float
            Transition costs for each input state.
        """
        pass

    @abstractmethod
    def prev_state(self, states: List[State], action: int) -> List[State]:
        """Compute predecessor states by applying the inverse of an action.

        Parameters
        ----------
        states : list of State
            Current states.
        action : int
            Action whose inverse is applied to each state.

        Returns
        -------
        list of State
            Predecessor states.
        """
        pass

    @abstractmethod
    def generate_goal_states(self, num_states: int) -> List[State]:
        """Generate goal states.

        Parameters
        ----------
        num_states : int
            Number of goal states to generate.

        Returns
        -------
        list of State
            Goal states.
        """
        pass

    @abstractmethod
    def is_solved(self, states: List[State]) -> np.ndarray:
        """Check whether each state is solved.

        Parameters
        ----------
        states : list of State
            States to evaluate.

        Returns
        -------
        numpy.ndarray
            Boolean vector where element ``i`` indicates if state ``i`` is solved.
        """
        pass

    @abstractmethod
    def state_to_nnet_input(self, states: List[State]) -> List[np.ndarray]:
        """Convert states to neural network input tensors (numpy arrays).

        Parameters
        ----------
        states : list of State
            States to convert.

        Returns
        -------
        list of numpy.ndarray
            One or more arrays; the first dimension indexes the state within the batch.
        """
        pass

    @abstractmethod
    def get_num_moves(self) -> int:
        """Number of available actions for fixed-action environments.

        Returns
        -------
        int
            Number of action indices.
        """
        pass

    @abstractmethod
    def get_nnet_model(self) -> nn.Module:
        """Return the neural network model used by this environment."""
        pass

    def generate_states(
        self, num_states: int, backwards_range: Tuple[int, int]
    ) -> Tuple[List[State], List[int]]:
        """Generate training states via reverse moves from the goal.

        Parameters
        ----------
        num_states : int
            Number of states to generate.
        backwards_range : tuple of int
            Inclusive range ``(min, max)`` of reverse moves to apply.

        Returns
        -------
        list of State
            Generated states.
        list of int
            Number of reverse moves applied to produce each state.
        """
        assert num_states > 0
        assert backwards_range[0] >= 0
        assert (
            self.fixed_actions
        ), "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states: List[State] = self.generate_goal_states(num_states)

        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        while np.max(num_back_moves < scramble_nums):
            idxs: np.ndarray = np.where((num_back_moves < scramble_nums))[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_to_move = [states[i] for i in idxs]
            states_moved = self.prev_state(states_to_move, move)

            for state_moved_idx, state_moved in enumerate(states_moved):
                states[idxs[state_moved_idx]] = state_moved

            num_back_moves[idxs] = num_back_moves[idxs] + 1

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        """Generate all children for each state.

        Parameters
        ----------
        states : list of State
            States to expand.

        Returns
        -------
        list of list of State
            For each input state, the list of successor states for all actions.
        list of numpy.ndarray
            For each input state, a 1D vector of transition costs per action.
        """
        assert (
            self.fixed_actions
        ), "Environments without fixed actions must implement their own method"

        # Initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = []
        for _ in range(len(states)):
            states_exp.append([])

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # For each move, get next states and transition costs
        move_idx: int
        # move: int

        for move_idx in range(num_env_moves):
            # next state
            states_next_move: List[State]
            tc_move: List[float]
            states_next_move, tc_move = self.next_state(states, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(states_next_move[idx])

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l


class Cube3(Environment):
    """Rubik's Cube 3x3 environment with fixed actions."""
    moves: List[str] = [
        "%s%i" % (f, n) for f in ["U", "D", "L", "R", "B", "F"] for n in [-1, 1]
    ]
    moves_rev: List[str] = [
        "%s%i" % (f, n) for f in ["U", "D", "L", "R", "B", "F"] for n in [1, -1]
    ]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # Solved state colors
        self.goal_colors: np.ndarray = np.arange(
            0, (self.cube_len**2) * 6, 1, dtype=self.dtype
        )

        # Indices updated by each move
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(
            self.cube_len, self.moves
        )

    def next_state(
        self, states: List[Cube3State], action: int
    ) -> Tuple[List[Cube3State], List[float]]:
        """Apply an action to a batch of states.

        Parameters
        ----------
        states : list of Cube3State
            Input states.
        action : int
            Index of the move to apply.

        Returns
        -------
        list of Cube3State
            Next states after applying the move.
        list of float
            Transition costs per state.
        """
        states_np = np.stack([x.colors for x in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, action)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[Cube3State], action: int) -> List[Cube3State]:
        """Apply the inverse move to compute predecessor states."""
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(
        self, num_states: int, np_format: bool = False
    ) -> Union[List[Cube3State], np.ndarray]:
        """Generate solved states.

        Parameters
        ----------
        num_states : int
            Number of solved states to produce.
        np_format : bool, default False
            If True, return a numpy array batch; otherwise, a list of states.

        Returns
        -------
        list of Cube3State or numpy.ndarray
            Generated solved states.
        """
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube3State] = [
                Cube3State(self.goal_colors.copy()) for _ in range(num_states)
            ]

        return solved_states

    def is_solved(self, states: List[Cube3State]) -> np.ndarray:
        """Check whether each state equals the goal configuration."""
        states_np = np.stack([state.colors for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_colors, 0))

        return np.all(is_equal, axis=1)

    def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
        """Convert states into normalized integer features for the network."""
        states_np = np.stack([state.colors for state in states], axis=0)

        representation_np: np.ndarray = states_np / (self.cube_len**2)
        representation_np: np.ndarray = representation_np.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def get_num_moves(self) -> int:
        """Number of legal moves (fixed across states)."""
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        """Build and return the neural network model used for heuristics."""
        state_dim: int = (self.cube_len**2) * 6
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)

        return nnet

    def generate_states(
        self, num_states: int, backwards_range: Tuple[int, int]
    ) -> Tuple[List[Cube3State], List[int]]:
        """Generate states by scrambling solved states with reverse moves.

        Parameters
        ----------
        num_states : int
            Number of states to generate.
        backwards_range : tuple of int
            Inclusive range ``(min, max)`` of reverse moves to apply.

        Returns
        -------
        list of Cube3State
            Generated states.
        list of int
            Number of reverse moves applied to each state.
        """
        assert num_states > 0
        assert backwards_range[0] >= 0
        assert (
            self.fixed_actions
        ), "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        """Generate all children for each state using numpy operations."""
        assert (
            self.fixed_actions
        ), "Environments without fixed actions must implement their own method"

        # Initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # Numpy representation of states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        # For each move, compute next states and transition costs
        move_idx: int
        # move: int

        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube3State(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, action: int) -> Tuple[np.ndarray, List[float]]:
        """Vectorized cube move.

        Parameters
        ----------
        states_np : numpy.ndarray
            Batch of flattened cube color arrays, shape ``(N, 6 * L * L)``.
        action : int
            Move index to apply.

        Returns
        -------
        numpy.ndarray
            Next states after applying the move.
        list of float
            Transition costs (all ones).
        """
        action_str: str = self.moves[action]

        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[
            :, self.rotate_idxs_old[action_str]
        ]

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _get_adj(self) -> None:
        """Initialize adjacency relationships between faces.

        Face indices: WHITE=0, YELLOW=1, BLUE=2, GREEN=3, ORANGE=4, RED=5.
        """
        self.adj_faces: Dict[int, np.ndarray] = {
            0: np.array([2, 5, 3, 4]),
            1: np.array([2, 4, 3, 5]),
            2: np.array([0, 4, 1, 5]),
            3: np.array([0, 5, 1, 4]),
            4: np.array([0, 3, 1, 2]),
            5: np.array([0, 2, 1, 3]),
        }

    def _compute_rotation_idxs(
        self, cube_len: int, moves: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Pre-compute index mappings for rotating faces and their adjacents."""
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {
                0: {
                    2: [range(0, cube_len), cube_len - 1],
                    3: [range(0, cube_len), cube_len - 1],
                    4: [range(0, cube_len), cube_len - 1],
                    5: [range(0, cube_len), cube_len - 1],
                },
                1: {
                    2: [range(0, cube_len), 0],
                    3: [range(0, cube_len), 0],
                    4: [range(0, cube_len), 0],
                    5: [range(0, cube_len), 0],
                },
                2: {
                    0: [0, range(0, cube_len)],
                    1: [0, range(0, cube_len)],
                    4: [cube_len - 1, range(cube_len - 1, -1, -1)],
                    5: [0, range(0, cube_len)],
                },
                3: {
                    0: [cube_len - 1, range(0, cube_len)],
                    1: [cube_len - 1, range(0, cube_len)],
                    4: [0, range(cube_len - 1, -1, -1)],
                    5: [cube_len - 1, range(0, cube_len)],
                },
                4: {
                    0: [range(0, cube_len), cube_len - 1],
                    1: [range(cube_len - 1, -1, -1), 0],
                    2: [0, range(0, cube_len)],
                    3: [cube_len - 1, range(cube_len - 1, -1, -1)],
                },
                5: {
                    0: [range(0, cube_len), 0],
                    1: [range(cube_len - 1, -1, -1), cube_len - 1],
                    2: [cube_len - 1, range(0, cube_len)],
                    3: [0, range(cube_len - 1, -1, -1)],
                },
            }
            face_dict = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[
                    (np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to)))
                    % len(faces_to)
                ]

            cubes_idxs = [
                [0, range(0, cube_len)],
                [range(0, cube_len), cube_len - 1],
                [cube_len - 1, range(cube_len - 1, -1, -1)],
                [range(cube_len - 1, -1, -1), 0],
            ]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[
                    (np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to)))
                    % len(cubes_to)
                ]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [
                    [idx1, idx2]
                    for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten()
                    for idx2 in np.array([cubes_idxs[cubes_to[i]][1]]).flatten()
                ]
                idxs_old = [
                    [idx1, idx2]
                    for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten()
                    for idx2 in np.array([cubes_idxs[cubes_from[i]][1]]).flatten()
                ]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index(
                        (face, idxNew[0], idxNew[1]), colors_new.shape
                    )
                    flat_idx_old = np.ravel_multi_index(
                        (face, idxOld[0], idxOld[1]), colors.shape
                    )
                    rotate_idxs_new[move] = np.concatenate(
                        (rotate_idxs_new[move], [flat_idx_new])
                    )
                    rotate_idxs_old[move] = np.concatenate(
                        (rotate_idxs_old[move], [flat_idx_old])
                    )

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [
                    [idx1, idx2]
                    for idx1 in np.array([face_idxs[face_to][0]]).flatten()
                    for idx2 in np.array([face_idxs[face_to][1]]).flatten()
                ]
                idxs_old = [
                    [idx1, idx2]
                    for idx1 in np.array([face_idxs[face_from][0]]).flatten()
                    for idx2 in np.array([face_idxs[face_from][1]]).flatten()
                ]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index(
                        (face_to, idxNew[0], idxNew[1]), colors_new.shape
                    )
                    flat_idx_old = np.ravel_multi_index(
                        (face_from, idxOld[0], idxOld[1]), colors.shape
                    )
                    rotate_idxs_new[move] = np.concatenate(
                        (rotate_idxs_new[move], [flat_idx_new])
                    )
                    rotate_idxs_old[move] = np.concatenate(
                        (rotate_idxs_old[move], [flat_idx_old])
                    )

        return rotate_idxs_new, rotate_idxs_old
