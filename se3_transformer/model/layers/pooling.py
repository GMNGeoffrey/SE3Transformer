# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from typing import Dict, Literal

import torch
import torch.nn as nn
from dgl.nn.pytorch import AvgPooling, MaxPooling
from torch import Tensor

from se3_transformer.model.graph import SE3Graph, DGLGraphWrapper


# When I tried using torch geometric's own ops I got errors with torch.compile
def _segment_reduce(feat: Tensor, batch_num_nodes: Tensor, reduce: str) -> Tensor:
    """
    Perform segment-based reduction (sum, mean, max) over nodes grouped by graph.
    Used for PyTorchGraph backend.

    Args:
        feat: Node features [total_nodes, ...]
        batch_num_nodes: Number of nodes per graph [batch_size]
        reduce: Reduction operation ('sum', 'mean', 'max')

    Returns:
        Reduced features [batch_size, ...]
    """
    batch_size = batch_num_nodes.shape[0]
    device = feat.device

    # Create segment indices: [0, 0, 0, 1, 1, 1, 1, 2, 2, ...] for nodes belonging to each graph
    segment_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        batch_num_nodes
    )

    if reduce == 'sum':
        # Use scatter_add for sum
        output_shape = (batch_size,) + feat.shape[1:]
        output = torch.zeros(output_shape, dtype=feat.dtype, device=device)
        # Expand segment_ids to match feat dimensions
        expanded_ids = segment_ids.view(-1, *([1] * (feat.dim() - 1))).expand_as(feat)
        output.scatter_add_(0, expanded_ids, feat)
        return output

    elif reduce == 'mean':
        # Sum then divide by count
        output_shape = (batch_size,) + feat.shape[1:]
        output = torch.zeros(output_shape, dtype=feat.dtype, device=device)
        expanded_ids = segment_ids.view(-1, *([1] * (feat.dim() - 1))).expand_as(feat)
        output.scatter_add_(0, expanded_ids, feat)
        # Divide by node counts (broadcast to match feature dimensions)
        counts = batch_num_nodes.float().view(-1, *([1] * (feat.dim() - 1)))
        return output / counts

    elif reduce == 'max':
        # Use scatter_reduce for max (PyTorch 1.12+)
        output_shape = (batch_size,) + feat.shape[1:]
        output = torch.full(output_shape, float('-inf'), dtype=feat.dtype, device=device)
        expanded_ids = segment_ids.view(-1, *([1] * (feat.dim() - 1))).expand_as(feat)
        output.scatter_reduce_(0, expanded_ids, feat, reduce='amax', include_self=False)
        return output

    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


class GPooling(nn.Module):
    """
    Graph max/average pooling on a given feature type.
    The average can be taken for any feature type, and equivariance will be maintained.
    The maximum can only be taken for invariant features (type 0).
    If you want max-pooling for type > 0 features, look into Vector Neurons.
    """

    def __init__(self, feat_type: int = 0, pool: Literal['max', 'avg'] = 'max'):
        """
        :param feat_type: Feature type to pool
        :param pool: Type of pooling: max or avg
        """
        super().__init__()
        assert pool in ['max', 'avg'], f'Unknown pooling: {pool}'
        assert feat_type == 0 or pool == 'avg', 'Max pooling on type > 0 features will break equivariance'
        self.feat_type = feat_type
        self.pool_type = pool
        # DGL pooling modules for DGLGraphWrapper
        self._dgl_pool = MaxPooling() if pool == 'max' else AvgPooling()

    def forward(self, features: Dict[str, Tensor], graph: SE3Graph, **kwargs) -> Tensor:
        feat = features[str(self.feat_type)]

        if isinstance(graph, DGLGraphWrapper):
            pooled = self._dgl_pool(graph._graph, feat)
        else:
            # PyTorch-native pooling for PyTorchGraph
            batch_num_nodes = graph.batch_num_nodes()
            reduce = 'max' if self.pool_type == 'max' else 'mean'
            pooled = _segment_reduce(feat, batch_num_nodes, reduce)

        return pooled.squeeze(dim=-1)
