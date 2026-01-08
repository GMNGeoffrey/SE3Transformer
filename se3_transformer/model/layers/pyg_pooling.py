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
from torch_geometric.nn.aggr import MaxAggregation, MeanAggregation
from torch_geometric.utils import cumsum


from se3_transformer.model.graph import SE3Graph

class PyGPooling(nn.Module):
    """
    Module wrapper for PyTorch Geometric global pooling functions to match DGL pooling interface.
    """

    def __init__(self, pool: Literal['max', 'avg'] = 'max'):
        super().__init__()
        self.pooler = MaxAggregation() if pool == 'max' else MeanAggregation()


    def forward(self, feat: Tensor, graph: SE3Graph) -> Tensor:
        batch_num_nodes = graph.batch_num_nodes().to(feat.device)
        # PyG cumsum includes the leading zero we need here.
        batch_ptr = cumsum(batch_num_nodes)

        pooled = self.pooler(
            feat,
            ptr=batch_ptr,
            dim_size=batch_num_nodes.shape[0],
            dim=0,
        )

        return pooled.squeeze(dim=-1)
