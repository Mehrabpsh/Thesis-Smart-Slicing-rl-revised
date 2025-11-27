"""State encoders for the SFC environment."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from omegaconf import DictConfig

from ..data.schemas import PhysicalNetwork, VNRequest


class StateEncoder(ABC):
    """Abstract base class for state encoders."""
    
    @abstractmethod
    def encode(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        current_vnf_idx: int,
        partial_embedding: Dict[int, int],
        path_embeddings: Optional[Dict[tuple, list]] = None,
    ) -> np.ndarray:
        """Encode state to feature vector.
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF to embed
            partial_embedding: Current partial embedding (vnf_id -> pn_node_id)
            path_embeddings: Optional path embeddings (vnf_i, vnf_j) -> [node_ids]
            
        Returns:
            Feature vector
        """
        pass
    
    @abstractmethod
    def get_state_dim(self, pn: PhysicalNetwork) -> int:
        """Get state dimension.
        
        Args:
            pn: Physical Network
            
        Returns:
            State dimension
        """
        pass



class NormalizedStateEncoder():
    """Normalized state encoder with request-aware global normalization.
    
    State vector components:
    1. Normalized residual bandwidth of all links
    2. Normalized remaining latency budget
    3. Normalized remaining number of VNFs
    4. Normalized SFC bandwidth demand
    """
    
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize normalized state encoder.
        
        Args:
            config: Optional configuration
        """
        if config is None:
            from omegaconf import DictConfig as DC
            config = DC({})
        #self.max_latency_budget = float(config.get("max_latency_budget", 100.0) if config else 100.0)
    
    def encode(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        max_residual_bw:float,
        current_vnf_idx: int,
        partial_embedding: Dict[int, int],
        path_embeddings: Optional[Dict[tuple, list]] = None,
    ) -> np.ndarray:
        """Encode state as normalized feature vector.
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            current_vnf_idx: Index of current VNF to embed
            partial_embedding: Current partial embedding (vnf_id -> pn_node_id)
            path_embeddings: Optional path embeddings for latency computation
            
        Returns:
            Normalized feature vector
        """
        features = []
        #max_residual_bw: float = 0.0

        # Component 1: Normalized Residual Bandwidth of Links
        # Compute max residual bandwidth across all links (including SFC demand)
        residual_bws = [link.available_bandwidth for link in pn.links]
        sfc_bw_demand = vn_request.bandwidth_demand
        #max_residual_bw = max(max(residual_bws) if residual_bws else 0.0, sfc_bw_demand)
        
        # Normalize each link's residual bandwidth
        if max_residual_bw > 0:
            normalized_link_bws = [
                link.available_bandwidth / max_residual_bw 
                for link in pn.links
            ]
        else:
            # If no bandwidth available, set all to 0
            normalized_link_bws = [0.0] * len(pn.links)
        
        features.extend(normalized_link_bws)
        
        # Component 2: Normalized Remaining Latency Budget
        # Compute accumulated latency from path embeddings
        accumulated_latency = self._compute_accumulated_latency(
            pn, vn_request, partial_embedding, path_embeddings
        )
        remaining_latency_budget = max(0.0, vn_request.sla_latency_ms - accumulated_latency)
        #normalized_remaining_latency = remaining_latency_budget / self.max_latency_budget
        #instead of above commented one, normalize the remaining latency budget
        normalized_remaining_latency = remaining_latency_budget / vn_request.sla_latency_ms

        # Clamp to [0, 1] range
        #normalized_remaining_latency = min(1.0, max(0.0, normalized_remaining_latency))

        features.append(normalized_remaining_latency)

        # Component 3: Normalized Remaining Number of VNFs
        remaining_vnfs_count = len(vn_request.vnfs) - current_vnf_idx
        sfc_length = vn_request.sfc_length
        if sfc_length > 0:
            normalized_remaining_vnfs = remaining_vnfs_count / sfc_length
        else:
            normalized_remaining_vnfs = 0.0
        features.append(normalized_remaining_vnfs)
        
        # Component 4: Normalized SFC Bandwidth Demand
        if max_residual_bw > 0:
            normalized_sfc_bw_demand = sfc_bw_demand / max_residual_bw
        else:
            normalized_sfc_bw_demand = 0.0
        features.append(normalized_sfc_bw_demand)
        
        return np.array(features, dtype=np.float32)
    
    
    def _compute_accumulated_latency(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        partial_embedding: Dict[int, int],
        path_embeddings: Optional[Dict[tuple, list]],
    ) -> float:
        """Compute accumulated latency from embedded paths.
        
        Args:
            pn: Physical Network
            vn_request: Current VN request
            partial_embedding: Current partial embedding
            path_embeddings: Path embeddings (vnf_i, vnf_j) -> [node_ids]
            
        Returns:
            Accumulated latency in milliseconds
        """
        accumulated_latency = 0.0
        
        if path_embeddings is not None:
            # Compute latency from path embeddings
            for path_key, path in path_embeddings.items():
                for i in range(len(path) - 1):
                    link = pn.get_link(path[i], path[i + 1])
                    if link:
                        accumulated_latency += link.delay_ms
        else:
            # Fallback: compute latency from partial embedding by finding paths
            # This is less efficient but works if path_embeddings not available
            sorted_vnf_indices = sorted(partial_embedding.keys())
            for i in range(len(sorted_vnf_indices) - 1):
                vnf_i = sorted_vnf_indices[i]
                vnf_j = sorted_vnf_indices[i + 1]
                node_i = partial_embedding[vnf_i]
                node_j = partial_embedding[vnf_j]
                
                # Find shortest path (simplified - assumes direct or shortest path)
                try:
                    import networkx as nx
                    if pn.graph.has_edge(node_i, node_j):
                        link = pn.get_link(node_i, node_j)
                        if link:
                            accumulated_latency += link.delay_ms
                    else:
                        # Try to find shortest path
                        try:
                            path = nx.shortest_path(pn.graph, node_i, node_j)
                            for k in range(len(path) - 1):
                                link = pn.get_link(path[k], path[k + 1])
                                if link:
                                    accumulated_latency += link.delay_ms
                        except (nx.NetworkXNoPath, Exception):
                            # If no path found, skip this segment
                            pass
                except Exception:
                    # If graph operations fail, skip
                    pass
        
        return accumulated_latency
    
    def get_state_dim(self, pn: PhysicalNetwork) -> int:
        """Get state dimension.
        
        Args:
            pn: Physical Network
            
        Returns:
            State dimension: num_links + 3 (latency budget + remaining VNFs + SFC BW demand)
        """
        return len(pn.links) + 3


