"""Synthetic generators for Physical Network and Virtual Network requests."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from .schemas import PhysicalNetwork, PNNode, PNLink, VNRequest, VNF
from ..utils.serialization import save_json

from pathlib import Path
import os


def generate_erdos_renyi_pn(
    num_nodes: int,
    p: float,
    node_cpu_range: Tuple[float, float],
    link_bw_range: Tuple[float, float],
    link_delay_range: Tuple[float, float],
    rng: Optional[np.random.Generator] = None,
) -> PhysicalNetwork:
    """Generate an Erdős–Rényi random graph Physical Network.
    
    Args:
        num_nodes: Number of nodes
        p: Edge probability
        node_cpu_range: (min, max) CPU capacity per node
        link_bw_range: (min, max) bandwidth per link
        link_delay_range: (min, max) delay in milliseconds per link
        rng: Optional random number generator
        
    Returns:
        PhysicalNetwork instance
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate graph
    #graph = nx.erdos_renyi_graph(num_nodes, p, seed=rng.integers(0, 2**31))
    graph = nx.erdos_renyi_graph(num_nodes, p, seed=rng)

    
    # Create nodes
    nodes = {}
    for node_id in range(num_nodes):
        cpu = rng.uniform(node_cpu_range[0], node_cpu_range[1])
        nodes[node_id] = PNNode(
            node_id=node_id,
            cpu_capacity=cpu,
        )
    
    # Create links
    links = []
    for src, dst in graph.edges():
        bw = rng.uniform(link_bw_range[0], link_bw_range[1])
        delay = rng.uniform(link_delay_range[0], link_delay_range[1])
        links.append(PNLink(
            src=src,
            dst=dst,
            bandwidth=bw,
            delay_ms=delay,
        ))
    
    return PhysicalNetwork(nodes=nodes, links=links)




def generate_poisson_vn_stream(
    num_requests: int,
    num_groups: int,
    rate: float,
    sfc_len_range: Tuple[int, int],
    vnf_types: List[str],
    cpu_demand_range: Tuple[float, float],
    bw_demand_range: Tuple[float, float],
    sla_latency_range: Tuple[float, float],
    cache_path: Path,
    rng: Optional[np.random.Generator] = None,
):# -> List[VNRequest]:
    """Generate a stream of VN requests following Poisson arrival process.
    
    Args:
        num_requests: Total number of requests to generate
        rate: Poisson arrival rate
        sfc_len_range: (min, max) SFC length
        vnf_types: List of available VNF types
        cpu_demand_range: (min, max) CPU demand per VNF
        bw_demand_range: (min, max) bandwidth demand
        sla_latency_range: (min, max) SLA latency in milliseconds
        rng: Optional random number generator
        
    Returns:
        List of VNRequest objects sorted by arrival time
    """
    if rng is None:
        rng = np.random.default_rng()
    
    cache_path.mkdir(parents=True, exist_ok=True)
    
    #requests = []
    current_time = 0.0
    
    for req_id in range(num_requests):
        # Generate arrival time (exponential inter-arrival)
        inter_arrival = rng.exponential(1.0 / rate) if rate > 0 else 0.0
        current_time += inter_arrival
        
        # Generate SFC length
        sfc_len = rng.integers(sfc_len_range[0], sfc_len_range[1] + 1)
        
        # Generate VNFs
        vnfs = []
        for vnf_id in range(sfc_len):
            vnf_type = rng.choice(vnf_types)
            cpu_demand = rng.uniform(cpu_demand_range[0], cpu_demand_range[1])
            vnfs.append(VNF(
                vnf_id=vnf_id,
                vnf_type=vnf_type,
                cpu_demand=cpu_demand,
            ))
        
        # Generate bandwidth and latency requirements
        bw_demand = rng.uniform(bw_demand_range[0], bw_demand_range[1])
        sla_latency = rng.uniform(sla_latency_range[0], sla_latency_range[1])
        # requests.append(VNRequest(
        #     request_id=req_id,
        #     group_id= req_id // num_groups,
        #     vnfs=vnfs,
        #     bandwidth_demand=bw_demand,
        #     sla_latency_ms=sla_latency,
        #     arrival_time=current_time,
        # ))
        v_net = VNRequest(
            request_id=req_id,
            group_id= req_id // num_groups,
            vnfs=vnfs,
            bandwidth_demand=bw_demand,
            sla_latency_ms=sla_latency,
            arrival_time=current_time,
            )

        v_net_data = {
            'request_id': v_net.request_id,
            'group_id' : v_net.group_id,
            'vnfs': [
                {
                    'vnf_id': vnf.vnf_id,
                    'vnf_type': vnf.vnf_type,
                    'cpu_demand': vnf.cpu_demand,
                    'features': vnf.features,
                }
                for vnf in v_net.vnfs
            ],
            'bandwidth_demand': v_net.bandwidth_demand,
            'sla_latency_ms': v_net.sla_latency_ms,
            'arrival_time': v_net.arrival_time,
            'features': v_net.features,
                    }

        save_json(v_net_data, Path(os.path.join(cache_path, f'v_net-{v_net.request_id:05d}.json')))
        
    
    #return requests


