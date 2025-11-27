"""File loaders for Physical Network and Virtual Network data."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx
from .schemas import PhysicalNetwork, PNNode, PNLink, VNRequest, VNF


def load_pn_from_json(path: Path) -> PhysicalNetwork:
    """Load Physical Network from JSON file.
    
    Expected JSON format:
    {
        "nodes": [
            {"node_id": 0, "cpu_capacity": 16.0, "features": {}}
        ],
        "links": [
            {"src": 0, "dst": 1, "bandwidth": 20.0, "delay_ms": 2.0, "features": {}}
        ]
    }
    
    Args:
        path: Path to JSON file
        
    Returns:
        PhysicalNetwork instance
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    nodes = {}
    for node_data in data["nodes"]:
        node_id = node_data["node_id"]
        nodes[node_id] = PNNode(
            node_id=node_id,
            cpu_capacity=node_data["cpu_capacity"],
            features=node_data.get("features", {}),
        )
    
    links = []
    for link_data in data["links"]:
        links.append(PNLink(
            src=link_data["src"],
            dst=link_data["dst"],
            bandwidth=link_data["bandwidth"],
            delay_ms=link_data.get("delay_ms", 0.0),
            features=link_data.get("features", {}),
        ))
    
    return PhysicalNetwork(nodes=nodes, links=links)



def load_pn_from_graphml(path: Path) -> PhysicalNetwork:
    """Load Physical Network from GML file (Virne-style).
    
    Args:
        path: Path to GML file
        
    Returns:
        PhysicalNetwork instance
    """
    try:
        # Try reading as GML with label='id'
        graph = nx.read_gml(path, label='id')
    except:
        # Fallback to regular GML
        graph = nx.read_gml(path)
    
    nodes = {}
    for node_id, node_data in graph.nodes(data=True):
        # Handle both int and string node IDs
        if isinstance(node_id, str):
            try:
                node_id = int(node_id)
            except ValueError:
                # Use hash if can't convert to int
                node_id = hash(node_id) % 10000
        
        cpu_capacity = float(node_data.get("cpu_capacity", node_data.get("cpu", 16.0)))
        nodes[node_id] = PNNode(
            node_id=node_id,
            cpu_capacity=cpu_capacity,
            features={k: float(v) for k, v in node_data.items() if k not in ["cpu_capacity", "cpu"]},
        )
    
    links = []
    for src, dst, link_data in graph.edges(data=True):
        # Handle both int and string node IDs
        if isinstance(src, str):
            try:
                src = int(src)
            except ValueError:
                src = hash(src) % 10000
        if isinstance(dst, str):
            try:
                dst = int(dst)
            except ValueError:
                dst = hash(dst) % 10000
        
        bandwidth = float(link_data.get("bandwidth", link_data.get("bw", 20.0)))
        delay_ms = float(link_data.get("delay_ms", link_data.get("delay", 0.0)))
        links.append(PNLink(
            src=src,
            dst=dst,
            bandwidth=bandwidth,
            delay_ms=delay_ms,
            features={k: float(v) for k, v in link_data.items() if k not in ["bandwidth", "bw", "delay_ms", "delay"]},
        ))
    
    return PhysicalNetwork(nodes=nodes, links=links)




def load_vn_requests_from_json(path: Path) -> List[VNRequest]:
    """Load VN requests from JSON file.
    
    Expected JSON format:
    {
        "requests": [
            {
                "request_id": 0,
                "vnfs": [
                    {"vnf_id": 0, "vnf_type": "fw", "cpu_demand": 2.0, "features": {}}
                ],
                "bandwidth_demand": 5.0,
                "sla_latency_ms": 20.0,
                "arrival_time": 0.0,
                "features": {}
            }
        ]
    }
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of VNRequest objects
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    requests = []
    for req_data in data["requests"]:
        vnfs = []
        for vnf_data in req_data["vnfs"]:
            vnfs.append(VNF(
                vnf_id=vnf_data["vnf_id"],
                vnf_type=vnf_data["vnf_type"],
                cpu_demand=vnf_data["cpu_demand"],
                features=vnf_data.get("features", {}),
            ))
        
        requests.append(VNRequest(
            request_id=req_data["request_id"],
            vnfs=vnfs,
            bandwidth_demand=req_data["bandwidth_demand"],
            sla_latency_ms=req_data["sla_latency_ms"],
            arrival_time=req_data.get("arrival_time", 0.0),
            features=req_data.get("features", {}),
        ))
    
    return requests
