"""Metrics computation for SFC VNE."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import time

from ..data.schemas import VNRequest


@dataclass
class Metrics:
    """Metrics container."""
    
    acceptance_ratio: float = 0.0
    response_time: float = 0.0
    qoe: float = 0.0
    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    request_metrics: List[Dict] = field(default_factory=list)
    
    def update(
        self,
        request: VNRequest,
        accepted: bool,
        response_time: float,
        reward: Optional[float] = None,
    ) -> None:
        """Update metrics with a request result.
        
        Args:
            request: VN request
            accepted: Whether request was accepted
            response_time: Response time in seconds
            qoe: Optional QoE value
            reward: Optional reward value
        """
        self.total_requests += 1
        if accepted:
            self.accepted_requests += 1
        else:
            self.rejected_requests += 1
        
        self.response_time += response_time
        if reward is not None:
            self.episode_rewards.append(reward)
        
        self.request_metrics.append({
            "request_id": request.request_id,
            "accepted": accepted,
            #"response_time": response_time,
            "reward": reward,
        })
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if self.total_requests == 0:
            return {
                "acceptance_ratio": 0.0,
                "response_time": 0.0,
                "qoe": 0.0,
                "mean_reward": 0.0,
                "mean_episode_length": 0.0,
            }
        
        acceptance_ratio = self.accepted_requests / self.total_requests
        avg_response_time = self.response_time / self.total_requests
        avg_qoe = self.qoe / self.accepted_requests if self.accepted_requests > 0 else 0.0
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        return {
            "acceptance_ratio": acceptance_ratio,
            "response_time": avg_response_time,
            "qoe": float(avg_qoe),
            "mean_reward": float(mean_reward),
        }
    
    def reset(self) -> None:
        """Reset metrics."""
        self.acceptance_ratio = 0.0
        self.response_time = 0.0
        #self.qoe = 0.0
        self.total_requests = 0
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.success_indices = {}

        #self.episode_rewards = []
        #self.episode_lengths = []
        self.request_metrics = []



