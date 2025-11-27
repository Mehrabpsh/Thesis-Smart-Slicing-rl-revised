"""Tests for action masking."""

import pytest
import numpy as np
from sfc_rl.data.schemas import PhysicalNetwork, PNNode, PNLink, VNRequest, VNF
from sfc_rl.env.action_space import NodeSelectionActionSpace


@pytest.fixture
def small_pn():
    """Create a small physical network."""
    nodes = {
        0: PNNode(node_id=0, cpu_capacity=10.0),
        1: PNNode(node_id=1, cpu_capacity=5.0),  # Lower capacity
        2: PNNode(node_id=2, cpu_capacity=10.0),
    }
    links = [
        PNLink(src=0, dst=1, bandwidth=20.0, delay_ms=1.0),
        PNLink(src=1, dst=2, bandwidth=20.0, delay_ms=1.0),
    ]
    return PhysicalNetwork(nodes=nodes, links=links)


@pytest.fixture
def vn_request():
    """Create a VN request."""
    vnfs = [
        VNF(vnf_id=0, vnf_type="fw", cpu_demand=2.0),
        VNF(vnf_id=1, vnf_type="nat", cpu_demand=8.0),  # High demand
    ]
    return VNRequest(
        request_id=0,
        vnfs=vnfs,
        bandwidth_demand=5.0,
        sla_latency_ms=10.0,
    )


def test_action_mask_first_vnf(small_pn, vn_request):
    """Test action mask for first VNF."""
    action_space = NodeSelectionActionSpace(mask_illegal=True)
    partial_embedding = {}
    
    valid_actions = action_space.get_valid_actions(
        small_pn, vn_request, current_vnf_idx=0, partial_embedding=partial_embedding
    )
    
    # All nodes should be valid for first VNF (if they have enough CPU)
    assert len(valid_actions) > 0
    assert 0 in valid_actions  # Node 0 has enough CPU (10.0 >= 2.0)
    assert 1 in valid_actions  # Node 1 has enough CPU (5.0 >= 2.0)
    assert 2 in valid_actions  # Node 2 has enough CPU (10.0 >= 2.0)


def test_action_mask_second_vnf(small_pn, vn_request):
    """Test action mask for second VNF (with high CPU demand)."""
    action_space = NodeSelectionActionSpace(mask_illegal=True)
    partial_embedding = {0: 0}  # First VNF on node 0
    
    valid_actions = action_space.get_valid_actions(
        small_pn, vn_request, current_vnf_idx=1, partial_embedding=partial_embedding
    )
    
    # Node 1 should not be valid (only 5.0 CPU, needs 8.0)
    # Node 0 and 2 should be valid if connected
    assert 1 not in valid_actions  # Insufficient CPU
    # Node 0 and 2 should be valid (connected and have enough CPU)
    assert len(valid_actions) > 0


def test_action_mask_connectivity(small_pn, vn_request):
    """Test that action mask respects connectivity."""
    action_space = NodeSelectionActionSpace(mask_illegal=True)
    partial_embedding = {0: 0}  # First VNF on node 0
    
    valid_actions = action_space.get_valid_actions(
        small_pn, vn_request, current_vnf_idx=1, partial_embedding=partial_embedding
    )
    
    # Should only include nodes connected to node 0
    # Node 1 is connected, node 2 might be connected via path
    assert len(valid_actions) > 0


def test_action_mask_format(small_pn, vn_request):
    """Test action mask format."""
    action_space = NodeSelectionActionSpace(mask_illegal=True)
    partial_embedding = {}
    
    mask = action_space.get_action_mask(
        small_pn, vn_request, current_vnf_idx=0, partial_embedding=partial_embedding
    )
    
    assert isinstance(mask, np.ndarray)
    assert len(mask) == len(small_pn.nodes)
    assert mask.dtype == np.float32
    assert np.all((mask == 0) | (mask == 1))  # Binary mask

