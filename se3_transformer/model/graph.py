"""
SE3Graph: A graph abstraction for SE3Transformer that supports both DGL and PyTorch Geometric backends.

This module provides a unified graph interface that can use either:
1. DGL graphs (default, for exact compatibility with original implementation)
2. Pure PyTorch tensors with PyTorch Geometric ops (better compatibility with torch.compile)

The abstract base class defines the interface, and two subclasses implement
the specific backends.
"""

import os
import logging

import torch
from torch import Tensor

from se3_transformer.model.se3_graph import SE3Graph

USE_PYG_ENV_VAR = "SE3_USE_PYG"
SE3_USE_PYG = os.environ.get(USE_PYG_ENV_VAR, "1") == "1"

if SE3_USE_PYG:
    logging.info("Using PyTorch Geometric backend for SE3Graph")
    from se3_transformer.model.pyg_graph import PyGGraph
else:
    logging.info("Using DGL Graph backend for SE3Graph")
    import dgl
    from se3_transformer.model.dgl_graph_wrapper import DGLGraphWrapper


def create_graph(
    src: Tensor,
    dst: Tensor,
    num_nodes: int,
    device: torch.device = None,
) -> SE3Graph:
    """
    Factory function to create an SE3Graph using either DGL or PyTorch backend.

    Args:
        src: Source node indices for each edge
        dst: Destination node indices for each edge
        num_nodes: Total number of nodes
        device: Device for the graph (only used for DGL backend)

    Returns:
        SE3Graph instance (either DGLGraphWrapper or PyGGraph)
    """
    if SE3_USE_PYG:
        return PyGGraph(src, dst, num_nodes)
    else:
        if device is None:
            device = src.device
        # Checking the node count is a big performance hit
        dgl_graph = dgl.graph((src, dst), num_nodes=num_nodes, node_count_check=False, device=device)
        return DGLGraphWrapper(dgl_graph)


def from_dgl_graph(dgl_graph) -> SE3Graph:
    """
    Construct an SE3Graph from an existing DGL graph.

    Depending on the value of SE3_USE_PYG, this will either convert the
    DGL graph to a PyGGraph or wrap it in a DGLGraphWrapper. This is useful
    for wrapping graphs returned by dgl.batch() or other DGL operations.

    Args:
        dgl_graph: A DGL graph (possibly batched)

    Returns:
        SE3Graph wrapping the graph (either DGLGraphWrapper or PyGGraph)
    """
    if SE3_USE_PYG:
        src, dst = dgl_graph.edges()
        g = PyGGraph(
            src=src,
            dst=dst,
            num_nodes=dgl_graph.num_nodes(),
            batch_num_nodes=dgl_graph.batch_num_nodes(),
        )
        g._edata = {k: v for k, v in dgl_graph.edata.items()}
        return g
    else:
        return DGLGraphWrapper(dgl_graph)
