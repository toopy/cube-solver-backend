import os
import re
from collections import OrderedDict
from typing import (
    Callable,
    List,
    Optional,
)

import numpy as np
import torch
from torch import (
    Tensor,
    nn,
)

from .environment import Environment


def load_nnet(
    model_file: str,
    nnet: nn.Module,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load model weights into a neural network module.

    Parameters
    ----------
    model_file : str
        Path to a ``state_dict`` file produced by PyTorch.
    nnet : torch.nn.Module
        Model instance to load into.
    device : torch.device or None, default None
        Device to map tensors to while loading. If ``None``, CPU is used.

    Returns
    -------
    torch.nn.Module
        The same model instance with loaded weights and set to eval mode.
    """
    # Get state dict
    if device is None:
        state_dict = torch.load(
            model_file,
            map_location=torch.device("cpu"),
        )
    else:
        state_dict = torch.load(
            model_file,
            map_location=device,
        )

    # Remove potential DataParallel "module." prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub("^module\.", "", k)
        new_state_dict[k] = v

    # Set state dict and switch to eval
    nnet.load_state_dict(new_state_dict)
    nnet.eval()

    return nnet


def states_nnet_to_pytorch_input(
    states_nnet: List[np.ndarray],
    device: Optional[torch.device],
) -> List[Tensor]:
    """Convert numpy inputs to PyTorch tensors on a target device.

    Parameters
    ----------
    states_nnet : list of numpy.ndarray
        Neural-net-ready inputs (already batched).
    device : torch.device or None
        Target device for tensors. If ``None``, tensors are created on CPU.

    Returns
    -------
    list of torch.Tensor
        Tensors mirroring the provided numpy arrays.
    """
    states_nnet_tensors: List[Tensor] = []
    for tensor_np in states_nnet:
        tensor = torch.tensor(tensor_np, device=device)
        states_nnet_tensors.append(tensor)

    return states_nnet_tensors


def get_heuristic_fn(
    nnet: nn.Module,
    device: torch.device,
    env: Environment,
    clip_zero: bool = False,
    batch_size: Optional[int] = None,
) -> Callable[[List, bool], np.ndarray]:
    """Wrap a model into a batched heuristic function.

    Parameters
    ----------
    nnet : torch.nn.Module
        Neural network that maps state representations to cost-to-go values.
    device : torch.device
        Device on which the network runs.
    env : Environment
        Environment used to convert states to neural-net inputs.
    clip_zero : bool, default False
        If True, negative predictions are clipped to zero.
    batch_size : int or None, default None
        Max batch size for forward passes. If ``None``, processes all at once.

    Returns
    -------
    callable
        Function ``heuristic_fn(states, is_nnet_format=False) -> np.ndarray``.
        ``states`` can be a list of environment states or a list of numpy
        arrays representing pre-formatted network inputs when
        ``is_nnet_format`` is True.
    """
    nnet.eval()

    def heuristic_fn(
        states: List,
        is_nnet_format: bool = False,
    ) -> np.ndarray:
        cost_to_go: np.ndarray = np.zeros(0)
        if not is_nnet_format:
            num_states: int = len(states)
        else:
            num_states = int(states[0].shape[0])

        batch_size_inst: int = num_states
        if batch_size is not None:
            batch_size_inst = batch_size

        start_idx: int = 0
        while start_idx < num_states:
            # Batch slice
            end_idx: int = min(start_idx + batch_size_inst, num_states)

            # Convert to network input
            if not is_nnet_format:
                states_batch: List = states[start_idx:end_idx]
                states_nnet_batch: List[np.ndarray] = env.state_to_nnet_input(
                    states_batch
                )
            else:
                states_nnet_batch = [x[start_idx:end_idx] for x in states]

            # Model forward
            states_nnet_batch_tensors = states_nnet_to_pytorch_input(
                states_nnet_batch, device
            )
            cost_to_go_batch: np.ndarray = (
                nnet(*states_nnet_batch_tensors).cpu().data.numpy()
            )

            cost_to_go = np.concatenate(
                (cost_to_go, cost_to_go_batch[:, 0]), axis=0
            )

            start_idx = end_idx

        assert cost_to_go.shape[0] == num_states

        if clip_zero:
            cost_to_go = np.maximum(cost_to_go, 0.0)

        return cost_to_go

    return heuristic_fn


def load_heuristic_fn(
    nnet_dir: str,
    device: torch.device,
    on_gpu: bool,
    nnet: nn.Module,
    env: Environment,
    clip_zero: bool = False,
    gpu_num: int = -1,
    batch_size: Optional[int] = None,
) -> Callable[[List], np.ndarray]:
    """Load a model from disk and return a heuristic function.

    Parameters
    ----------
    nnet_dir : str
        Directory containing ``model_state_dict.pt``.
    device : torch.device
        Device for model inference.
    on_gpu : bool
        If True, wraps the model with ``nn.DataParallel`` after moving to GPU.
    nnet : torch.nn.Module
        Model architecture instance.
    env : Environment
        Environment for state-to-network conversion.
    clip_zero : bool, default False
        Whether to clip negative predictions to zero.
    gpu_num : int, default -1
        CUDA device index to expose via ``CUDA_VISIBLE_DEVICES`` when ``on_gpu``.
    batch_size : int or None, default None
        Max batch size for the returned heuristic function.

    Returns
    -------
    callable
        Heuristic function as defined in ``get_heuristic_fn``.
    """
    if (gpu_num >= 0) and on_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    model_file = f"{nnet_dir}/model_state_dict.pt"

    nnet = load_nnet(
        model_file,
        nnet,
        device=device,
    )
    nnet.eval()
    nnet.to(device)
    if on_gpu:
        nnet = nn.DataParallel(nnet)

    heuristic_fn = get_heuristic_fn(
        nnet,
        device,
        env,
        clip_zero=clip_zero,
        batch_size=batch_size,
    )

    return heuristic_fn
