import torch
from torch import Tensor
import torch_geometric

from se3_transformer.model.se3_graph import SE3Graph


class PyGGraph(SE3Graph):
    """
    SE3Graph implementation using pure PyTorch tensors with PyTorch Geometric operations.

    This backend offers better compatibility with torch.compile.
    """

    def __init__(
        self,
        src: Tensor,
        dst: Tensor,
        num_nodes: int,
        batch_num_nodes: Tensor = None,
    ):
        """
        Create a graph from edge indices and relative positions.

        Args:
            src: Source node indices for each edge [num_edges]
            dst: Destination node indices for each edge [num_edges]
            num_nodes: Total number of nodes in the graph
            batch_num_nodes: Node counts per graph in batch [batch_size] (optional)
        """
        self._src = src
        self._dst = dst
        self._num_nodes = num_nodes
        self._edata = {}
        # Default to single graph if not specified
        if batch_num_nodes is None:
            self._batch_num_nodes = torch.tensor([num_nodes], device=src.device)
        else:
            self._batch_num_nodes = batch_num_nodes

        # Used for PyG aggregation methods. Not part of the SE3Graph interface.
        self.batch_ptr = torch_geometric.utils.cumsum(self._batch_num_nodes)

    def edges(self) -> tuple[Tensor, Tensor]:
        """Return (src, dst) tensors for all edges."""
        return self._src, self._dst

    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        return self._num_nodes

    @property
    def edata(self):
        """Access edge data dictionary."""
        return self._edata

    def batch_num_nodes(self) -> Tensor:
        """Return tensor of node counts per graph in batch."""
        return self._batch_num_nodes

    def to(self, device, **kwargs) -> 'PyGGraph':
        """Move the graph to the specified device."""

        new_graph = PyGGraph(
            self._src.to(device, **kwargs),
            self._dst.to(device, **kwargs),
            self._num_nodes,
            self._batch_num_nodes.to(device, **kwargs),
        )
        new_graph._edata = {k: (v.to(device, **kwargs) if isinstance(v, Tensor) else v) for k, v in self.edata.items()}
        return new_graph

    def copy_e_sum(self, edge_feats: Tensor) -> Tensor:
        """Sum edge features to destination nodes using PyG scatter."""
        return torch_geometric.utils.scatter(edge_feats, self._dst, dim=0, dim_size=self._num_nodes, reduce='sum')

    def e_dot_v(self, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
        """Dot product of edge features with destination node features."""
        return (edge_feats * node_feats[self._dst]).sum(dim=-1, keepdim=True)

    def edge_softmax(self, edge_weights: Tensor) -> Tensor:
        """Softmax over edges grouped by destination node using PyG softmax."""
        return torch_geometric.utils.softmax(edge_weights, self._dst, num_nodes=self._num_nodes)
