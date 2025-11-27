"""Data schemas for Physical Network (PN) and Virtual Network (VN) requests."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import networkx as nx


@dataclass
class PNNode:
    """Physical Network Node.
    
    Attributes:
        node_id: Unique node identifier
        cpu_capacity: CPU capacity (units)
        available_cpu: Currently available CPU
        features: Optional additional node features
    """
    node_id: int
    cpu_capacity: float
    available_cpu: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.available_cpu is None:
            self.available_cpu = self.cpu_capacity
    
    def can_host(self, cpu_demand: float) -> bool:
        """Check if node can host a VNF with given CPU demand.
        
        Args:
            cpu_demand: Required CPU
            
        Returns:
            True if node has sufficient CPU
        """
        return self.available_cpu >= cpu_demand
    
    def allocate(self, cpu_demand: float) -> None:
        """Allocate CPU resources.
        
        Args:
            cpu_demand: CPU to allocate
        """
        if not self.can_host(cpu_demand):
            raise ValueError(f"Insufficient CPU: need {cpu_demand}, have {self.available_cpu}")
        self.available_cpu -= cpu_demand
    
    def deallocate(self, cpu_demand: float) -> None:
        """Deallocate CPU resources.
        
        Args:
            cpu_demand: CPU to deallocate
        """
        self.available_cpu += cpu_demand
        if self.available_cpu > self.cpu_capacity:
            self.available_cpu = self.cpu_capacity


@dataclass
class PNLink:
    """Physical Network Link.
    
    Attributes:
        src: Source node ID
        dst: Destination node ID
        bandwidth: Bandwidth capacity (units)
        available_bandwidth: Currently available bandwidth
        delay_ms: Link delay in milliseconds
        features: Optional additional link features
    """
    src: int
    dst: int
    bandwidth: float
    available_bandwidth: Optional[float] = None
    delay_ms: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.available_bandwidth is None:
            self.available_bandwidth = self.bandwidth
    
    def can_carry(self, bw_demand: float) -> bool:
        """Check if link can carry given bandwidth demand.
        
        Args:
            bw_demand: Required bandwidth
            
        Returns:
            True if link has sufficient bandwidth
        """
        return self.available_bandwidth >= bw_demand
    
    def allocate(self, bw_demand: float) -> None:
        """Allocate bandwidth resources.
        
        Args:
            bw_demand: Bandwidth to allocate
        """
        if not self.can_carry(bw_demand):
            raise ValueError(f"Insufficient bandwidth: need {bw_demand}, have {self.available_bandwidth}")
        self.available_bandwidth -= bw_demand
    
    def deallocate(self, bw_demand: float) -> None:
        """Deallocate bandwidth resources.
        
        Args:
            bw_demand: Bandwidth to deallocate
        """
        self.available_bandwidth += bw_demand
        if self.available_bandwidth > self.bandwidth:
            self.available_bandwidth = self.bandwidth


@dataclass
class PhysicalNetwork:
    """Physical Network topology and resources.
    
    Attributes:
        nodes: Dictionary of node_id -> PNNode
        links: List of PNLink objects
        graph: NetworkX graph representation
    """
    nodes: Dict[int, PNNode]
    links: List[PNLink]
    graph: nx.Graph = field(default=None, init=False)
    
    def __post_init__(self):
        """Build NetworkX graph from nodes and links."""
        self.graph = nx.Graph()
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, **node.features)
        for link in self.links:
            self.graph.add_edge(
                link.src,
                link.dst,
                bandwidth=link.bandwidth,
                available_bandwidth=link.available_bandwidth,
                delay_ms=link.delay_ms,
                **link.features
            )
    
    def get_node(self, node_id: int) -> PNNode:
        """Get node by ID.
        
        Args:
            node_id: Node identifier
            
        Returns:
            PNNode instance
        """
        return self.nodes[node_id]
    
    def get_link(self, src: int, dst: int) -> Optional[PNLink]:
        """Get link between two nodes.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            
        Returns:
            PNLink if exists, None otherwise
        """
        for link in self.links:
            if (link.src == src and link.dst == dst) or (link.src == dst and link.dst == src):
                return link
        return None
    
    def reset(self) -> None:
        """Reset all resources to initial capacity."""
        for node in self.nodes.values():
            node.available_cpu = node.cpu_capacity
        for link in self.links:
            link.available_bandwidth = link.bandwidth
    
    @classmethod
    def from_setting(cls, config: Dict, seed: Optional[int] = None) -> 'PhysicalNetwork':
        """Create PhysicalNetwork from setting (Virne-style).
        
        Args:
            config: Configuration dictionary with topology and attribute settings
            seed: Optional random seed
            
        Returns:
            PhysicalNetwork instance
        """
        from .generators import generate_erdos_renyi_pn
        from .loaders import load_pn_from_graphml
        from ..utils.seed import get_rng
        import os
        
        if seed is not None:
            rng = get_rng(seed)
        else:
            rng = get_rng()
        
        topology = config.get('topology', {})
        
        
        # Generate topology
        topo_type = topology.get('type', 'erdos_renyi')
        num_nodes = topology.get('num_nodes', 30)
        
        # Get attribute settings
        node_attrs_setting = config.get('node_attrs_setting', [])
        link_attrs_setting = config.get('link_attrs_setting', [])
        
        # Extract CPU range from node attributes
        cpu_attr: Dict = next((attr for attr in node_attrs_setting if attr.get('name') == 'cpu'), {})
        cpu_range = (cpu_attr.get('low', 8), cpu_attr.get('high', 16))
        
        # Extract bandwidth and delay from link attributes
        bw_attr: Dict = next((attr for attr in link_attrs_setting if attr.get('name') == 'bandwidth'), {})
        delay_attr: Dict = next((attr for attr in link_attrs_setting if attr.get('name') == 'delay'), {})
        bw_range = (bw_attr.get('low', 10), bw_attr.get('high', 40))
        delay_range = (delay_attr.get('low', 1), delay_attr.get('high', 5))
        
        # Generate based on topology type
        if topo_type == 'erdos_renyi':
            p = topology.get('p', 0.08)
            return generate_erdos_renyi_pn(
                num_nodes=num_nodes,
                p=p,
                node_cpu_range=cpu_range,
                link_bw_range=bw_range,
                link_delay_range=delay_range,
                rng=rng,
            )
        else:
            raise ValueError(f"Unsupported topology type: {topo_type}")
    
    def save_dataset(self, dataset_dir: str, file_name: str = 'p_net.gml') -> None:
        """Save physical network to GML file (Virne-style).
        
        Args:
            dataset_dir: Directory to save the dataset
            file_name: Name of the GML file
        """
        import os
        os.makedirs(dataset_dir, exist_ok=True)
        file_path = os.path.join(dataset_dir, file_name)
        
        # Convert to NetworkX graph and save as GML
        gml_graph = nx.Graph()
        gml_graph.graph.update({
            'num_nodes': len(self.nodes),
            'num_links': len(self.links),
        })
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            gml_graph.add_node(node_id, cpu_capacity=node.cpu_capacity, **node.features)
        
        # Add edges with attributes
        for link in self.links:
            gml_graph.add_edge(
                link.src,
                link.dst,
                bandwidth=link.bandwidth,
                delay_ms=link.delay_ms,
                **link.features
            )
        
        nx.write_gml(gml_graph, file_path)
    
    @classmethod
    def load_dataset(cls, dataset_dir: str, file_name: str = 'p_net.gml') -> 'PhysicalNetwork':
        """Load physical network from GML file (Virne-style).
        
        Args:
            dataset_dir: Directory containing the dataset
            file_name: Name of the GML file
            
        Returns:
            PhysicalNetwork instance
        """
        from .loaders import load_pn_from_graphml
        from pathlib import Path
        import os
        
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file '{file_name}' not found in directory: {dataset_dir}")
        
        return load_pn_from_graphml(Path(file_path))
    
    @classmethod
    def from_gml(cls, file_path: str) -> 'PhysicalNetwork':
        """Load physical network from GML file.
        
        Args:
            file_path: Path to GML file
            
        Returns:
            PhysicalNetwork instance
        """
        from .loaders import load_pn_from_graphml
        from pathlib import Path
        return load_pn_from_graphml(Path(file_path))


@dataclass
class VNF:
    """Virtual Network Function.
    
    Attributes:
        vnf_id: VNF identifier (within a VN request)
        vnf_type: Type of VNF (e.g., 'fw', 'nat', 'ids')
        cpu_demand: Required CPU
        features: Optional additional VNF features
    """
    vnf_id: int
    vnf_type: str
    cpu_demand: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class VNRequest:
    """Virtual Network Request (SFC).
    
    Attributes:
        request_id: Unique request identifier
        vnfs: List of VNFs in the chain (ordered)
        bandwidth_demand: Required bandwidth between consecutive VNFs
        sla_latency_ms: Maximum acceptable latency (milliseconds)
        arrival_time: Request arrival time
        features: Optional additional request features
    """
    request_id: int
    group_id: int
    vnfs: List[VNF]
    bandwidth_demand: float
    sla_latency_ms: float
    arrival_time: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    
    @property
    def sfc_length(self) -> int:
        """Get Service Function Chain length."""
        return len(self.vnfs)
    
    def get_vnf(self, vnf_id: int) -> VNF:
        """Get VNF by ID.
        
        Args:
            vnf_id: VNF identifier
            
        Returns:
            VNF instance
        """
        return self.vnfs[vnf_id]

