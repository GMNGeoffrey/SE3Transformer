import dgl
from torch import Tensor

from se3_transformer.model.graph import SE3Graph


class DGLGraphWrapper(SE3Graph):
    """
    SE3Graph implementation that wraps a DGL graph.

    This is the default backend that provides exact compatibility with the
    original SE3Transformer implementation. It uses DGL's graph operations
    which are highly optimized but cause graph breaks in torch.compile.
    """

    def __init__(self, dgl_graph):
        """
        Wrap an existing DGL graph.
        """
        self._graph = dgl_graph

    def edges(self) -> tuple[Tensor, Tensor]:
        """Return (src, dst) tensors for all edges."""
        return self._graph.edges()

    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        return self._graph.num_nodes()

    def batch_num_nodes(self) -> Tensor:
        """Return tensor of node counts per graph in batch."""
        return self._graph.batch_num_nodes()

    def to(self, device, **kwargs) -> 'DGLGraphWrapper':
        """Move the graph to the specified device."""
        return DGLGraphWrapper(self._graph.to(device))

    @property
    def edata(self):
        """Access edge data from the underlying DGL graph."""
        return self._graph.edata

    def copy_e_sum(self, edge_feats: Tensor) -> Tensor:
        """Sum edge features to destination nodes using DGL ops."""
        return dgl.ops.copy_e_sum(self._graph, edge_feats)

    def e_dot_v(self, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
        """Dot product using DGL ops."""
        return dgl.ops.e_dot_v(self._graph, edge_feats, node_feats)

    def edge_softmax(self, edge_weights: Tensor) -> Tensor:
        """Edge softmax using DGL ops."""
        return dgl.ops.edge_softmax(self._graph, edge_weights)
