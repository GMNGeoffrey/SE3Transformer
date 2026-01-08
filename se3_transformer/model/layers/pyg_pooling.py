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

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_max_pool, global_mean_pool

from se3_transformer.model.graph import SE3Graph

class PyGPooling(nn.Module):
    """
    Module wrapper for PyTorch Geometric global pooling functions to match DGL pooling interface.
    """
    # TODO: Pretty sure there's already nn module versions of these in PyG. Use those instead.

    def __init__(self, pool: Literal['max', 'avg'] = 'max'):
        super().__init__()
        self.pooler = global_max_pool if pool == 'max' else global_mean_pool

    def forward(self, feat: Tensor, graph: SE3Graph) -> Tensor:
            # PyG pooling expects [num_nodes, num_features], so flatten extra dims
            orig_shape = feat.shape
            feat_flat = feat.flatten(start_dim=1)

            # Create batch assignment tensor from batch_num_nodes
            batch_num_nodes = graph.batch_num_nodes()
            batch = torch.repeat_interleave(
                torch.arange(batch_num_nodes.shape[0], device=feat.device),
                batch_num_nodes
            )

            pooled = self.pooler(feat_flat, batch, batch_num_nodes.shape[0])

            return pooled.view(pooled.shape[0], *orig_shape[1:]).squeeze(dim=-1)
