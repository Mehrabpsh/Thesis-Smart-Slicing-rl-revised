import time

import numpy as np

import sys

sys.path.append('/home/mehrab/Workspaces/Thesis-Smart-Slicing-rl-revised')

print(sys.path)


from sfc_rl.baselines.exhaustive_solver import ExhaustiveSolver
from sfc_rl.data.schemas import PhysicalNetwork, PNNode, PNLink, VNRequest, VNF


class DummyRewardFn:
    """Simple reward function for testing.

    Computes negative total delay over all mapped paths so that
    shorter paths are preferred but any feasible solution is acceptable.
    """

    def compute(self, pn, vn_request, embedding, path_embeddings, is_terminal: bool) -> float:
        total_delay = 0.0
        for path in path_embeddings.values():
            for i in range(len(path) - 1):
                link = pn.get_link(path[i], path[i + 1])
                if link is not None:
                    total_delay += link.delay_ms
        return -total_delay


class DummyEnv:
    """Minimal env that provides the interface required by ExhaustiveSolver."""

    def __init__(self, pn: PhysicalNetwork, vn_requests, reward_fn):
        self.pn = pn
        self.vn_requests = vn_requests
        self.current_request_idx = 0
        self.current_vnf_idx = 0
        self.partial_embedding = {}
        self.reward_fn = reward_fn

    def action_mask(self):
        """Allow all PN nodes as valid actions."""
        return np.ones(len(self.pn.nodes), dtype=np.int32)


def build_small_pn_and_request():
    """Construct a tiny PN and a simple VNRequest for testing."""
    # Three fully connected PN nodes with enough CPU and bandwidth
    nodes = {
        0: PNNode(node_id=0, cpu_capacity=10.0),
        1: PNNode(node_id=1, cpu_capacity=10.0),
        2: PNNode(node_id=2, cpu_capacity=2.0),
    }
    links = [
        PNLink(src=0, dst=1, bandwidth=10.0, delay_ms=1.0),
        PNLink(src=1, dst=2, bandwidth=10.0, delay_ms=1.0),
        PNLink(src=0, dst=2, bandwidth=10.0, delay_ms=1.0),
    ]
    pn = PhysicalNetwork(nodes=nodes, links=links)

    # VN request with two VNFs and moderate CPU/BW demands
    vnfs = [
        VNF(vnf_id=0, vnf_type="fw", cpu_demand=2.0),
        VNF(vnf_id=1, vnf_type="nat", cpu_demand=3.0),
    ]
    vn_request = VNRequest(
        request_id=0,
        group_id=0,
        vnfs=vnfs,
        bandwidth_demand=5.0,
        sla_latency_ms=100.0,
    )

    return pn, vn_request


def test_tree_search_finds_feasible_embeddings():
    """_tree_search_embeddings should find at least one feasible embedding."""
    solver = ExhaustiveSolver(timeout_seconds=5.0, max_embeddings=100)
    pn, vn_request = build_small_pn_and_request()

    start_time = time.time()
    solutions = solver._tree_search_embeddings(pn, vn_request, 0, {}, start_time)

    for id, tp in enumerate(solutions):
        print(f'the {id}th solution : {tp}....... this solution {'is' if solver._check_all_constraints(pn, vn_request, tp[0], tp[1]) else 'is not '} feasible')

    # Should find at least one complete embedding for this simple setup
    assert solutions, "Expected at least one feasible embedding"

    # All returned solutions should satisfy the final constraints
    for embedding, path_embeddings in solutions:
        assert solver._check_all_constraints(pn, vn_request, embedding, path_embeddings)


def test_solve_returns_valid_action_for_simple_env():
    """solve() should return a valid node-selection action for a simple env."""
    solver = ExhaustiveSolver(timeout_seconds=5.0, max_embeddings=100, seed=123)
    pn, vn_request = build_small_pn_and_request()
    env = DummyEnv(pn=pn, vn_requests=[vn_request], reward_fn=DummyRewardFn())

    action = solver.solve(env)

    # Action should be an integer index into the sorted PN node IDs
    assert isinstance(action, int)
    node_ids = sorted(pn.nodes.keys())
    assert 0 <= action < len(node_ids)

    chosen_node_id = node_ids[action]
    assert chosen_node_id in pn.nodes





test_tree_search_finds_feasible_embeddings()

