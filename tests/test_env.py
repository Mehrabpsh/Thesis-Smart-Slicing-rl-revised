"""Tests for SFC environment."""

import pytest
import numpy as np
from sfc_rl.data.schemas import PhysicalNetwork, PNNode, PNLink, VNRequest, VNF
from sfc_rl.env.sfc_env import SFCEnv
from sfc_rl.env.state_encoders import FlattenedStateEncoder
from sfc_rl.env.action_space import NodeSelectionActionSpace
from sfc_rl.env.reward import TerminalQoEReward
from sfc_rl.env.qoe import SimpleLinearQoE


@pytest.fixture
def small_pn():
    """Create a small physical network."""
    nodes = {
        0: PNNode(node_id=0, cpu_capacity=10.0),
        1: PNNode(node_id=1, cpu_capacity=10.0),
        2: PNNode(node_id=2, cpu_capacity=10.0),
    }
    links = [
        PNLink(src=0, dst=1, bandwidth=20.0, delay_ms=1.0),
        PNLink(src=1, dst=2, bandwidth=20.0, delay_ms=1.0),
    ]
    return PhysicalNetwork(nodes=nodes, links=links)


@pytest.fixture
def simple_vn_request():
    """Create a simple VN request."""
    vnfs = [
        VNF(vnf_id=0, vnf_type="fw", cpu_demand=2.0),
        VNF(vnf_id=1, vnf_type="nat", cpu_demand=2.0),
    ]
    return VNRequest(
        request_id=0,
        vnfs=vnfs,
        bandwidth_demand=5.0,
        sla_latency_ms=10.0,
    )


@pytest.fixture
def env(small_pn, simple_vn_request):
    """Create a test environment."""
    state_encoder = FlattenedStateEncoder()
    action_space = NodeSelectionActionSpace()
    qoe_model = SimpleLinearQoE()
    reward_fn = TerminalQoEReward(qoe_model)
    
    return SFCEnv(
        pn=small_pn,
        vn_requests=[simple_vn_request],
        state_encoder=state_encoder,
        action_space=action_space,
        reward_fn=reward_fn,
        qoe_model=qoe_model,
        max_steps_per_request=100,
    )


def test_env_reset(env):
    """Test environment reset."""
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert env.current_request_idx == 0
    assert env.current_vnf_idx == 0
    assert len(env.partial_embedding) == 0


def test_env_step(env):
    """Test environment step."""
    obs, info = env.reset(seed=42)
    action_mask = env.action_mask()
    valid_actions = np.where(action_mask > 0)[0]
    
    if len(valid_actions) > 0:
        action = valid_actions[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs is not None
        assert isinstance(reward, (int, float))


def test_env_action_mask(env):
    """Test action masking."""
    obs, info = env.reset(seed=42)
    action_mask = env.action_mask()
    assert action_mask is not None
    assert isinstance(action_mask, np.ndarray)
    assert len(action_mask) == len(env.pn.nodes)


def test_pn_node_allocation(small_pn):
    """Test PN node resource allocation."""
    node = small_pn.nodes[0]
    initial_cpu = node.available_cpu
    
    node.allocate(2.0)
    assert node.available_cpu == initial_cpu - 2.0
    
    node.deallocate(2.0)
    assert node.available_cpu == initial_cpu


def test_pn_link_allocation(small_pn):
    """Test PN link resource allocation."""
    link = small_pn.links[0]
    initial_bw = link.available_bandwidth
    
    link.allocate(5.0)
    assert link.available_bandwidth == initial_bw - 5.0
    
    link.deallocate(5.0)
    assert link.available_bandwidth == initial_bw

