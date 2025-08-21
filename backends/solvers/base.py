from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class Solver(ABC):
    """Abstract base class for cube solvers.

    Provides a common interface and lifecycle for loading models and solving
    inputs into action sequences.
    """

    def __init__(self, device: str = "cpu", **kwargs: Any) -> None:
        """Initialize a solver with a target device and optional metadata.

        Parameters
        ----------
        device : str, default "cpu"
            Torch device string (e.g., "cpu", "cuda").
        **kwargs : Any
            Arbitrary metadata stored on the solver instance.
        """
        self.device: torch.device = torch.device(device)
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = kwargs

    @abstractmethod
    def load(self) -> None:
        """Load any required model weights or resources."""

    @abstractmethod
    def solve(self, input: Any) -> Any:
        """Solve an instance and return a result.

        Parameters
        ----------
        input : Any
            Problem specification, e.g., a color layout.

        Returns
        -------
        Any
            Solver-specific result, e.g., list of moves.
        """
