"""Neural network models used for heuristic estimation.

This module defines a lightweight fully-connected ResNet-style model that
operates on flattened state representations and predicts scalar costs.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResnetModel(nn.Module):
    """Simple MLP with residual blocks for tabular inputs.

    Parameters
    ----------
    state_dim : int
        Length of the flattened input state vector.
    one_hot_depth : int
        If > 0, applies one-hot encoding of depth ``one_hot_depth`` to the
        integer-valued input before the first linear layer. If 0, uses the
        raw float-cast input directly.
    h1_dim : int
        Hidden dimension of the first fully connected layer.
    resnet_dim : int
        Hidden dimension of the residual stack.
    num_resnet_blocks : int
        Number of residual blocks (2-layer MLP residual units).
    out_dim : int
        Output dimension (e.g., 1 for scalar cost-to-go).
    batch_norm : bool
        If True, uses BatchNorm1d after the first and second FC layers and
        inside residual blocks.
    """

    def __init__(
        self,
        state_dim: int,
        one_hot_depth: int,
        h1_dim: int,
        resnet_dim: int,
        num_resnet_blocks: int,
        out_dim: int,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        states_nnet : torch.Tensor
            Input tensor of shape ``(N, state_dim)`` containing integer class
            indices when ``one_hot_depth > 0`` or floats otherwise.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(N, out_dim)``.
        """
        x: Tensor = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x
