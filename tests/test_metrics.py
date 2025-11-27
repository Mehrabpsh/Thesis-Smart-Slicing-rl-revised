"""Tests for metrics computation."""

import pytest
from sfc_rl.train.metrics import Metrics, compute_metrics
from sfc_rl.data.schemas import VNRequest, VNF


@pytest.fixture
def vn_request():
    """Create a VN request."""
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


def test_metrics_update(vn_request):
    """Test metrics update."""
    metrics = Metrics()
    
    metrics.update(vn_request, accepted=True, response_time=0.1, qoe=5.0, reward=5.0)
    assert metrics.total_requests == 1
    assert metrics.accepted_requests == 1
    assert metrics.rejected_requests == 0


def test_metrics_compute(vn_request):
    """Test metrics computation."""
    metrics = Metrics()
    
    # Add some requests
    for i in range(10):
        accepted = i % 2 == 0
        metrics.update(
            vn_request,
            accepted=accepted,
            response_time=0.1,
            qoe=5.0 if accepted else None,
            reward=5.0 if accepted else -1.0,
        )
    
    results = metrics.compute()
    
    assert "acceptance_ratio" in results
    assert "response_time" in results
    assert "qoe" in results
    assert results["acceptance_ratio"] == 0.5  # 5 out of 10 accepted


def test_metrics_reset(vn_request):
    """Test metrics reset."""
    metrics = Metrics()
    
    metrics.update(vn_request, accepted=True, response_time=0.1, qoe=5.0, reward=5.0)
    assert metrics.total_requests == 1
    
    metrics.reset()
    assert metrics.total_requests == 0
    assert metrics.accepted_requests == 0


def test_compute_metrics_function(vn_request):
    """Test compute_metrics function."""
    metrics = Metrics()
    
    metrics.update(vn_request, accepted=True, response_time=0.1, qoe=5.0, reward=5.0)
    
    results = compute_metrics(metrics)
    assert isinstance(results, dict)
    assert "acceptance_ratio" in results

