"""Tests for baseline policies."""

import pytest
import numpy as np
from sfc_rl.data.schemas import PhysicalNetwork, PNNode, PNLink, VNRequest, VNF
from sfc_rl.env.sfc_env import SFCEnv
from sfc_rl.env.state_encoders import FlattenedStateEncoder
from sfc_rl.env.action_space import NodeSelectionActionSpace
from sfc_rl.env.reward import TerminalQoEReward
from sfc_rl.env.qoe import SimpleLinearQoE
from sfc_rl.baselines.random_policy import RandomPolicy
from sfc_rl.baselines.exhaustive_solver import ExhaustiveSolver


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


def test_random_policy(env):
    """Test random policy."""
    policy = RandomPolicy(seed=42)
    env.reset(seed=42)
    
    action = policy.act(env)
    assert isinstance(action, int)
    assert 0 <= action < len(env.pn.nodes)


def test_random_policy_valid_action(env):
    """Test that random policy returns valid actions."""
    policy = RandomPolicy(seed=42)
    env.reset(seed=42)
    
    action_mask = env.action_mask()
    valid_actions = np.where(action_mask > 0)[0]
    
    if len(valid_actions) > 0:
        action = policy.act(env)
        assert action in valid_actions


def test_exhaustive_solver(env):
    """Test exhaustive solver."""
    solver = ExhaustiveSolver(timeout_seconds=1.0, max_embeddings=100, seed=42)
    env.reset(seed=42)
    
    action = solver.solve(env)
    assert isinstance(action, int)
    assert 0 <= action < len(env.pn.nodes)


def test_exhaustive_solver_timeout(env):
    """Test exhaustive solver timeout."""
    solver = ExhaustiveSolver(timeout_seconds=0.001, max_embeddings=100, seed=42)
    env.reset(seed=42)
    
    # Should complete quickly due to timeout
    action = solver.solve(env)
    assert isinstance(action, int)

