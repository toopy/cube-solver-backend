import time
from heapq import (
    heappop,
    heappush,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from .environment import (
    Environment,
    State,
)

T = TypeVar("T")


def flatten(data: List[List[T]]) -> Tuple[List[T], List[int]]:
    """Flatten a nested list and record split indices.

    Parameters
    ----------
    data : list of list of T
        Nested list to flatten.

    Returns
    -------
    list of T
        Flattened list containing elements of all sublists in order.
    list of int
        Split indices usable with ``unflatten`` to reconstruct the nesting.
        Each index is an exclusive end in the flattened list for each original
        sublist except the last one.
    """
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat: List[T] = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def unflatten(data: List[T], split_idxs: List[int]) -> List[List[T]]:
    """Reconstruct a nested list from a flattened list and split indices.

    Parameters
    ----------
    data : list of T
        Flattened list of elements.
    split_idxs : list of int
        Split indices as produced by ``flatten``.

    Returns
    -------
    list of list of T
        Nested list where each inner list corresponds to a segment in
        ``data`` delimited by ``split_idxs``.
    """
    data_split: List[List[T]] = []

    start_idx: int = 0
    end_idx: int
    for end_idx in split_idxs:
        data_split.append(data[start_idx:end_idx])
        start_idx = end_idx

    data_split.append(data[start_idx:])

    return data_split


class Node:
    """Search-tree node used by A*.

    Attributes
    ----------
    state : State
        Environment-specific state represented by this node.
    path_cost : float
        Accumulated path cost from the root to this node.
    heuristic : float or None
        Heuristic estimate of the remaining cost-to-go. Set by
        ``add_heuristic_and_cost``.
    cost : float or None
        f-score used for prioritization (typically ``g + h`` or weighted).
    is_solved : bool
        Whether this node corresponds to a solved/goal state.
    parent_move : int or None
        Index of the move taken from the parent to reach this node.
    parent : Node or None
        Parent node in the search tree.
    transition_costs : list of float
        Transition costs to each child (one per move from this node).
    children : list of Node
        Child nodes expanded from this node.
    bellman : float
        One-step backup value used for optional value estimations.
    """
    __slots__ = [
        "state",
        "path_cost",
        "heuristic",
        "cost",
        "is_solved",
        "parent_move",
        "parent",
        "transition_costs",
        "children",
        "bellman",
    ]

    def __init__(
        self,
        state: State,
        path_cost: float,
        is_solved: bool,
        parent_move: Optional[int],
        parent: Optional["Node"],
    ) -> None:
        self.state: State = state
        self.path_cost: float = path_cost
        self.heuristic: Optional[float] = None
        self.cost: Optional[float] = None
        self.is_solved: bool = is_solved
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent

        self.transition_costs: List[float] = []
        self.children: List[Node] = []

        self.bellman: float = np.inf

    def compute_bellman(self) -> None:
        """Compute one-step Bellman backup value for the node.

        Notes
        -----
        This method assumes that ``heuristic`` has been computed for this
        node and all its children. If no children are present, the node's
        heuristic is used as the backup value.
        """
        if self.is_solved:
            self.bellman = 0.0
        elif len(self.children) == 0:
            self.bellman = self.heuristic
        else:
            for node_c, tc in zip(self.children, self.transition_costs):
                self.bellman = min(self.bellman, tc + node_c.heuristic)


OpenSetElem = Tuple[float, int, Node]


class Instance:
    """Per-root search instance maintaining OPEN/CLOSED sets.

    Parameters
    ----------
    root_node : Node
        Root node for this instance.
    """

    def __init__(self, root_node: Node) -> None:
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.popped_nodes: List[Node] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = root_node

        self.push_to_open([self.root_node])

    def push_to_open(self, nodes: List[Node]) -> None:
        """Push nodes onto the OPEN set (priority queue)."""
        for node in nodes:
            heappush(self.open_set, (node.cost, self.heappush_count, node))
            self.heappush_count += 1

    def pop_from_open(self, num_nodes: int) -> List[Node]:
        """Pop up to ``num_nodes`` nodes from the OPEN set.

        Parameters
        ----------
        num_nodes : int
            Maximum number of nodes to pop.

        Returns
        -------
        list of Node
            Nodes popped in priority order.
        """
        num_to_pop: int = min(num_nodes, len(self.open_set))

        popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]
        self.goal_nodes.extend([node for node in popped_nodes if node.is_solved])
        self.popped_nodes.extend(popped_nodes)

        return popped_nodes

    def remove_in_closed(self, nodes: List[Node]) -> List[Node]:
        """Filter nodes against CLOSED, keeping better paths.

        For a given state, only keep a node if it is not present in CLOSED
        or if it has a strictly smaller path cost than the recorded one.

        Parameters
        ----------
        nodes : list of Node
            Nodes to filter.

        Returns
        -------
        list of Node
            Nodes not pruned by CLOSED.
        """
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if path_cost_prev is None:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost
            elif path_cost_prev > node.path_cost:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost

        return nodes_not_in_closed


def pop_from_open(instances: List[Instance], batch_size: int) -> List[List[Node]]:
    """Pop nodes from multiple instances' OPEN sets.

    Parameters
    ----------
    instances : list of Instance
        Search instances to pop from.
    batch_size : int
        Maximum number of nodes to pop per instance.

    Returns
    -------
    list of list of Node
        Popped nodes per instance.
    """
    popped_nodes_all: List[List[Node]] = [
        instance.pop_from_open(batch_size) for instance in instances
    ]

    return popped_nodes_all


def expand_nodes(
    instances: List[Instance],
    popped_nodes_all: List[List[Node]],
    env: Environment,
) -> List[List[Node]]:
    """Expand popped nodes for each instance using the environment.

    This performs a batched expansion across all instances for efficiency.

    Parameters
    ----------
    instances : list of Instance
        Search instances corresponding to ``popped_nodes_all``.
    popped_nodes_all : list of list of Node
        Nodes popped from each instance to be expanded.
    env : Environment
        Environment providing ``expand`` and ``is_solved`` operations.

    Returns
    -------
    list of list of Node
        All generated child nodes grouped per instance.
    """
    # Expand children of all nodes at once (for speed)
    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = flatten(popped_nodes_all)

    if len(popped_nodes_flat) == 0:
        return [[]]

    states: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[np.ndarray]

    states_c_by_node, tcs_np = env.expand(states)

    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]

    # Compute solved flags on all states at once (for speed)
    states_c: List[State]

    states_c, split_idxs_c = flatten(states_c_by_node)
    is_solved_c: List[bool] = list(env.is_solved(states_c))
    is_solved_c_by_node: List[List[bool]] = unflatten(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    parent_path_costs = np.expand_dims(
        np.array([node.path_cost for node in popped_nodes_flat]), 1
    )
    path_costs_c: List[float] = (
        (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()
    )

    path_costs_c_by_node: List[List[float]] = unflatten(path_costs_c, split_idxs_c)

    # Reshape lists
    tcs_by_inst_node: List[List[List[float]]] = unflatten(tcs_by_node, split_idxs)
    patch_costs_c_by_inst_node: List[List[List[float]]] = unflatten(
        path_costs_c_by_node, split_idxs
    )
    states_c_by_inst_node: List[List[List[State]]] = unflatten(
        states_c_by_node, split_idxs
    )
    is_solved_c_by_inst_node: List[List[List[bool]]] = unflatten(
        is_solved_c_by_node, split_idxs
    )

    # Build child nodes and attach to parents
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        tcs_by_node: List[List[float]] = tcs_by_inst_node[inst_idx]
        path_costs_c_by_node: List[List[float]] = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node: List[List[State]] = states_c_by_inst_node[inst_idx]

        is_solved_c_by_node: List[List[bool]] = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        states_c: List[State]
        # str_reps_c: List[str]

        for parent_node, tcs_node, path_costs_c, states_c, is_solved_c in zip(
            parent_nodes,
            tcs_by_node,
            path_costs_c_by_node,
            states_c_by_node,
            is_solved_c_by_node,
        ):
            state: State
            for move_idx, state in enumerate(states_c):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                node_c: Node = Node(state, path_cost, is_solved, move_idx, parent_node)

                nodes_c_by_inst[inst_idx].append(node_c)

                parent_node.children.append(node_c)

            parent_node.transition_costs.extend(tcs_node)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])

    return nodes_c_by_inst


def remove_in_closed(
    instances: List[Instance], nodes_c_all: List[List[Node]]
) -> List[List[Node]]:
    """Filter expanded nodes against CLOSED for each instance.

    Parameters
    ----------
    instances : list of Instance
        Instances associated with ``nodes_c_all``.
    nodes_c_all : list of list of Node
        Child nodes per instance.

    Returns
    -------
    list of list of Node
        Filtered child nodes per instance.
    """
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx])

    return nodes_c_all


def add_heuristic_and_cost(
    nodes: List[Node],
    heuristic_fn: Callable[[List[State]], np.ndarray],
    weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute and attach heuristic and f-cost to nodes.

    Parameters
    ----------
    nodes : list of Node
        Nodes to evaluate.
    heuristic_fn : callable
        Function mapping a list of states to a 1D numpy array of heuristic
        values with the same length.
    weights : sequence of float
        Per-node weight applied to the path cost when computing the f-cost.
        Must be broadcastable to the shape of ``path_costs``.

    Returns
    -------
    numpy.ndarray
        Vector of path costs ``g`` for the given nodes.
    numpy.ndarray
        Vector of heuristic values ``h`` for the given nodes.
    """
    if len(nodes) == 0:
        return np.zeros(0), np.zeros(0)

    # Heuristic values
    states: List[State] = [node.state for node in nodes]
    heuristics: np.ndarray = heuristic_fn(states)

    # Compute f-costs
    path_costs: np.ndarray = np.array([node.path_cost for node in nodes])
    is_solved: np.ndarray = np.array([node.is_solved for node in nodes])
    costs: np.ndarray = np.array(weights) * path_costs + heuristics * np.logical_not(
        is_solved
    )

    # Attach values to nodes
    for node, heuristic, cost in zip(nodes, heuristics, costs):
        node.heuristic = float(heuristic)
        node.cost = float(cost)

    return path_costs, heuristics


def add_to_open(instances: List[Instance], nodes: List[List[Node]]) -> None:
    """Push child nodes into each instance's OPEN set."""
    nodes_inst: List[Node]
    instance: Instance
    for instance, nodes_inst in zip(instances, nodes):
        instance.push_to_open(nodes_inst)


def get_path(node: Node) -> Tuple[List[State], List[int], float]:
    """Reconstruct the path from the root to the given node.

    Parameters
    ----------
    node : Node
        Target node.

    Returns
    -------
    list of State
        List of states from root to ``node`` (inclusive).
    list of int
        Moves taken to reach ``node`` (aligned with transitions between states).
    float
        Path cost of ``node``.
    """
    path: List[State] = []
    moves: List[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


class AStar:
    """Batch-capable A* search over environment-defined states.

    Parameters
    ----------
    states : list of State
        Root states to search from; one instance is created per root.
    env : Environment
        Environment providing expansion, solved checks, and move costs.
    heuristic_fn : callable
        Function mapping a list of states to a numpy vector of heuristic
        estimates. The callable may accept additional optional parameters
        but must work when called as ``heuristic_fn(states)``.
    weights : list of float
        Per-instance weights applied to path costs when computing f-costs.
    """

    def __init__(
        self,
        states: List[State],
        env: Environment,
        heuristic_fn: Callable[[List[State]], np.ndarray],
        weights: List[float],
    ) -> None:
        self.env: Environment = env
        self.weights: List[float] = weights
        self.step_num: int = 0

        self.timings: Dict[str, float] = {
            "pop": 0.0,
            "expand": 0.0,
            "check": 0.0,
            "heur": 0.0,
            "add": 0.0,
            "itr": 0.0,
        }

        # Compute starting costs for root nodes
        root_nodes: List[Node] = []
        is_solved_states: np.ndarray = self.env.is_solved(states)
        for state, is_solved in zip(states, is_solved_states):
            root_node: Node = Node(state, 0.0, is_solved, None, None)
            root_nodes.append(root_node)

        add_heuristic_and_cost(root_nodes, heuristic_fn, self.weights)

        # Initialize per-root instances
        self.instances: List[Instance] = []
        for root_node in root_nodes:
            self.instances.append(Instance(root_node))

    def step(
        self,
        heuristic_fn: Callable[[List[State]], np.ndarray],
        batch_size: int,
        include_solved: bool = False,
        verbose: bool = False,
    ) -> None:
        """Execute one A* iteration for all active instances.

        Parameters
        ----------
        heuristic_fn : callable
            Heuristic function invoked on all newly generated states.
        batch_size : int
            Maximum number of nodes to pop from OPEN per instance.
        include_solved : bool, default False
            If True, include already-solved instances in the iteration.
        verbose : bool, default False
            If True, print timing and heuristic statistics.
        """
        start_time_itr = time.time()
        instances: List[Instance]
        if include_solved:
            instances = self.instances
        else:
            instances = [
                instance for instance in self.instances if len(instance.goal_nodes) == 0
            ]

        # Pop from OPEN
        start_time = time.time()
        popped_nodes_all: List[List[Node]] = pop_from_open(instances, batch_size)
        pop_time = time.time() - start_time

        # Expand nodes
        start_time = time.time()
        nodes_c_all: List[List[Node]] = expand_nodes(
            instances, popped_nodes_all, self.env
        )
        expand_time = time.time() - start_time

        # Heuristic of children; do this before CLOSED check to allow backups
        start_time = time.time()
        nodes_c_all_flat, _ = flatten(nodes_c_all)
        weights, _ = flatten(
            [
                [weight] * len(nodes_c)
                for weight, nodes_c in zip(self.weights, nodes_c_all)
            ]
        )
        path_costs, heuristics = add_heuristic_and_cost(
            nodes_c_all_flat, heuristic_fn, weights
        )
        heur_time = time.time() - start_time

        # Check if children are in CLOSED
        start_time = time.time()
        nodes_c_all = remove_in_closed(instances, nodes_c_all)
        check_time = time.time() - start_time

        # Add remaining children to OPEN
        start_time = time.time()
        add_to_open(instances, nodes_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr

        # Optional logging
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = np.min(heuristics)
                min_heur_pc = path_costs[np.argmin(heuristics)]
                max_heur = np.max(heuristics)
                max_heur_pc = path_costs[np.argmax(heuristics)]

                print(
                    "Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                    "%.2f(%.2f)/%.2f(%.2f) "
                    % (self.step_num, min_heur, min_heur_pc, max_heur, max_heur_pc)
                )

            print(
                "Times - pop: %.2f, expand: %.2f, check: %.2f, heur: %.2f, "
                "add: %.2f, itr: %.2f"
                % (pop_time, expand_time, check_time, heur_time, add_time, itr_time)
            )

            print("")

        # Accumulate timings
        self.timings["pop"] += pop_time
        self.timings["expand"] += expand_time
        self.timings["check"] += check_time
        self.timings["heur"] += heur_time
        self.timings["add"] += add_time
        self.timings["itr"] += itr_time

        self.step_num += 1

    def has_found_goal(self) -> List[bool]:
        """Return per-instance flags indicating if a goal was found."""
        goal_found: List[bool] = [
            len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))
        ]

        return goal_found

    def get_goal_nodes(self, inst_idx: int) -> List[Node]:
        """Get all goal nodes found in a given instance."""
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx) -> Node:
        """Return the goal node with the smallest path cost for an instance."""
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]

        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        """Number of nodes generated so far for an instance."""
        return self.instances[inst_idx].num_nodes_generated

    def get_popped_nodes(self) -> List[List[Node]]:
        """All nodes popped from OPEN for each instance so far."""
        popped_nodes_all: List[List[Node]] = [
            instance.popped_nodes for instance in self.instances
        ]
        return popped_nodes_all
