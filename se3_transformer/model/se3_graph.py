from abc import ABC, abstractmethod

from torch import Tensor

class SE3Graph(ABC):
    """
    Abstract base class for SE3Transformer graph representations.

    This class defines the interface that both DGL and PyTorch Geometric backends
    implement. It is the minimal set of operations that SE3Transformer uses.

    The interface includes:
    - edges() -> (src, dst) tensors
    - num_nodes() -> int
    - batch_num_nodes() -> Tensor of node counts per graph in batch
    - edata property for edge data access
    - copy_e_sum(edge_feats) -> aggregate edge features to destination nodes
    - e_dot_v(edge_feats, node_feats) -> dot product of edge and node features
    - edge_softmax(edge_weights) -> softmax over edges per destination node
    """

    @abstractmethod
    def edges(self) -> tuple[Tensor, Tensor]:
        """Return (src, dst) tensors for all edges."""
        pass

    @abstractmethod
    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        pass

    @abstractmethod
    def batch_num_nodes(self) -> Tensor:
        """Return tensor of node counts per graph in batch."""
        pass

    @abstractmethod
    def to(self, device, **kwargs) -> 'SE3Graph':
        """Move the graph to the specified device."""
        pass

    @property
    @abstractmethod
    def edata(self):
        """Access edge data (dict-like interface)."""
        pass

    @abstractmethod
    def copy_e_sum(self, edge_feats: Tensor) -> Tensor:
        """
        Sum edge features to destination nodes.

        Args:
            edge_feats: Edge features of shape [num_edges, ...]

        Returns:
            Node features of shape [num_nodes, ...] with edge features summed to destinations
        """
        pass

    @abstractmethod
    def e_dot_v(self, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
        """
        Dot product of edge features with destination node features.

        Args:
            edge_feats: Edge features of shape [num_edges, ...]
            node_feats: Node features of shape [num_nodes, ...]

        Returns:
            Dot product result of shape [num_edges, ..., 1]
        """
        pass

    @abstractmethod
    def edge_softmax(self, edge_weights: Tensor) -> Tensor:
        """
        Softmax over edges grouped by destination node.

        Args:
            edge_weights: Edge weights of shape [num_edges, ...]

        Returns:
            Normalized edge weights of shape [num_edges, ...]
        """
        pass
