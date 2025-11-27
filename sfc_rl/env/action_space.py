"""Action space definitions and masking."""

from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict
import numpy as np

from ..data.schemas import PhysicalNetwork, VNRequest, VNF


class ActionSpace(ABC):
    """Abstract base class for action spaces."""
    
    @abstractmethod
    def get_action_dim(self, pn: PhysicalNetwork) -> int:
        """Get action space dimension.
        
        Args:
            pn: Physical Network
            
        Returns:
            Action space size
        """
        pass
    
    @abstractmethod
    def get_valid_actions(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
        partial_embedding: Dict[int, int],
    ) -> List[int]:
        """Get list of valid actions.
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF to embed
            partial_embedding: Current partial embedding (vnf_id -> pn_node_id)
            
        Returns:
            List of valid action indices
        """
        pass
    
    @abstractmethod
    def action_to_embedding(
        self,
        action: int,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
    ) -> int:
        """Convert action index to PN node ID.
        
        Args:
            action: Action index
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF
            
        Returns:
            PN node ID
        """
        pass
    
    @abstractmethod
    def get_action_mask(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
        partial_embedding: Dict[int, int],
    ) -> np.ndarray:
        """Get action mask (1 for valid, 0 for invalid).
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF
            partial_embedding: Current partial embedding
            
        Returns:
            Binary mask array
        """
        pass








class NodeSelectionActionSpace(ActionSpace):
    """Action space for selecting PN nodes for VNFs."""
    
    def __init__(self, mask_illegal: bool = True):
        """Initialize node selection action space.
        
        Args:
            mask_illegal: Whether to mask illegal actions
        """
        self.mask_illegal = mask_illegal
    
    def get_action_dim(self, pn: PhysicalNetwork) -> int:
        """Get action space dimension (number of PN nodes)."""
        return len(pn.nodes)
    
    def get_valid_actions(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
        partial_embedding: Dict[int, int],
    ) -> List[int]:
        """Get valid PN node IDs for current VNF.
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF
            partial_embedding: Current partial embedding
            
        Returns:
            List of valid PN node IDs
        """
        if current_vnf_idx >= len(vn_request.vnfs):
            return []
        
        vnf = vn_request.get_vnf(current_vnf_idx)
        valid_nodes = []
        
        for node_id, node in pn.nodes.items():
            # Check CPU capacity
            if node.can_host(vnf.cpu_demand):
                # If not first VNF, check connectivity to previous VNF
                if current_vnf_idx == 0:
                    valid_nodes.append(node_id)
                else:
                    prev_vnf_idx = current_vnf_idx - 1
                    prev_node_id = partial_embedding.get(prev_vnf_idx)
                    if prev_node_id is not None:
                        # Check if path exists
                        if (pn.graph.has_edge(node_id, prev_node_id) or \
                           self._has_path(pn, prev_node_id, node_id)) and (not (prev_node_id == node_id)):
                            valid_nodes.append(node_id)
        
        return valid_nodes
    
    def _has_path(self, pn: PhysicalNetwork, src: int, dst: int) -> bool:
        """Check if path exists between two nodes.
        
        Args:
            pn: Physical Network
            src: Source node ID
            dst: Destination node ID
            
        Returns:
            True if path exists
        """
        try:
            import networkx as nx
            return nx.has_path(pn.graph, src, dst)
        except:
            return False
    
    def action_to_embedding(
        self,
        action: int,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
    ) -> int:
        """Convert action index to PN node ID.
        
        Args:
            action: Action index (0 to num_nodes-1)
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF
            
        Returns:
            PN node ID
        """
        node_ids = sorted(pn.nodes.keys())
        if action < 0 or action >= len(node_ids):
            raise ValueError(f"Invalid action: {action}, valid range: [0, {len(node_ids)-1}]")
        return node_ids[action]
    
    def get_action_mask(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
        partial_embedding: Dict[int, int],
    ) -> np.ndarray:
        """Get action mask.
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF
            partial_embedding: Current partial embedding
            
        Returns:
            Binary mask array (1 for valid, 0 for invalid)
        """
        if not self.mask_illegal:
            return np.ones(self.get_action_dim(pn), dtype=np.float32)
        
        valid_actions = self.get_valid_actions(pn, vn_request, current_vnf_idx, partial_embedding)
        mask = np.zeros(self.get_action_dim(pn), dtype=np.float32)
        
        node_ids = sorted(pn.nodes.keys())
        for node_id in valid_actions:
            if node_id in node_ids:
                idx = node_ids.index(node_id)
                mask[idx] = 1.0
        
        return mask

