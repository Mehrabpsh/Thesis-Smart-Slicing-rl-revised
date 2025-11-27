"""Reward functions for the SFC environment."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from ..data.schemas import PhysicalNetwork, VNRequest
from .qoe import QoEModel
import numpy as np


class RewardFn(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Optional[Dict[int, int]],
        path_embeddings: Optional[Dict[tuple, list]],
        success: bool,
    ) -> float:
        """Compute reward.
        
        Args:
            pn: Physical Network
            vn_request: Virtual Network request
            embedding: VNF to PN node mapping (None if failed)
            path_embeddings: Path embeddings (None if failed)
            success: Whether embedding was successful
            
        Returns:
            Reward value
        """
        pass



class QoE_QoS_Reward(RewardFn):
    """Reward function based on QoE maximization with QoS constraints penalties."""

    def __init__(self, qoe_model: QoEModel, config: Optional[DictConfig] = None):
        """Initialize the reward function.

        Args:
            conf: Configuration object containing reward parameters.
                  Expected keys:
                  - penalty_failure (float): Penalty P for failed embeddings.
                  - penalty_exponent (bool): If True, apply exponential penalty
                                             Pen_rn^sfcic even on success.
                  - penalty_exp_factor (float): Factor for exponential penalty
                                                calculation (if applicable).
                  - penalty_weight (float): Weight for the exponential penalty
                                            term (if applicable).
                  - qoe_weight (float): Weight for the QoE term.
        """
        self.conf = config
        self.qoe_model = qoe_model
        # Use config values or defaults
        self.P = getattr(config, 'penalty_failure', 10.0) # Default P=10 from paper
        self.use_exponential_penalty = getattr(config, 'penalty_exponent', True)
        self.exp_factor = getattr(config, 'penalty_exp_factor', 1.0)
        self.penalty_weight = getattr(config, 'penalty_weight', 1.0)
        self.qoe_weight = getattr(config, 'qoe_weight', 1.0)

    def compute(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Optional[Dict[int, int]],
        path_embeddings: Optional[Dict[tuple, list]],
        success: bool,
    ) -> float:
        """Compute reward based on the paper's formula.

        R_QoE/QoS = A_rn * (QoE_sfcic - Pen_rn^sfcic) - (1 - A_rn) * P

        Args:
            pn: Physical Network
            vn_request: Virtual Network request
            embedding: VNF to PN node mapping (None if failed)
            path_embeddings: Path embeddings (None if failed)
            success: Whether embedding was successful (A_rn)

        Returns:
            Reward value
        """

        if success and embedding is not None and path_embeddings is not None:
            # Calculate QoE for the successful embedding
            qoe_sfcic = self.qoe_model.compute(pn, vn_request, embedding, path_embeddings)
            qoe_term = self.qoe_weight * qoe_sfcic

            # Calculate penalty for constraint violations even on success (Pen_rn^sfcic)
            penalty_term = 0.0
            if self.use_exponential_penalty:
                 penalty_term = self._calculate_exponential_penalty(pn, vn_request, embedding, path_embeddings)
                 penalty_term = self.penalty_weight * penalty_term

            # Successful embedding reward
            reward = qoe_term - penalty_term
        else:
            # Failed embedding penalty
            reward = - self.P # (1 - A_rn) * (-P) = 1 * (-P) = -P

        return reward

    def _calculate_exponential_penalty(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Dict[int, int],
        path_embeddings: Dict[tuple, list],
    ) -> float:
        """Calculate the exponential penalty Pen_rn^sfcic from Eq. (10).

        Pen_rn^sfcic = P * e^(-sqrt(sum((QoS_t^sfcic - QoS_t^rn)^2)))

        This example focuses on delay as per the paper's note. Bandwidth
        penalty can be added similarly.
        """

        achieved_delay = self._calculate_achieved_delay(pn, path_embeddings)
        requested_delay = vn_request.sla_latency_ms # Assuming this attribute exists

        # Calculate Euclidean distance for delay (and potentially bandwidth)
        # Example: sqrt((achieved_delay - requested_delay)^2 + (achieved_bw - requested_bw)^2)
        # For simplicity, assume only delay penalty for now.
        # Add bandwidth calculation similarly if needed.
        if achieved_delay is not None and requested_delay is not None:
             delay_diff_sq = (achieved_delay - requested_delay)**2
             # bandwidth_diff_sq = (achieved_bw - requested_bw)**2 # Example
             # total_diff_sq = delay_diff_sq + bandwidth_diff_sq # Example
             total_diff_sq = delay_diff_sq # Using only delay for now
             euclidean_dist = np.sqrt(total_diff_sq)
        else:
             # If calculation fails, assume maximum penalty
             euclidean_dist = float('inf')

        if euclidean_dist == float('inf'):
            penalty = self.P # Maximum penalty if calculation failed
        else:
            # Apply exponential penalty based on distance to requirement
            # Paper uses: P * e^(-distance). We add exp_factor for tuning.
            penalty = self.P * np.exp(- self.exp_factor * euclidean_dist)

        return penalty

    def _calculate_achieved_delay(
        self,
        pn: PhysicalNetwork,
        path_embeddings: Dict[tuple, list]
    ) -> Optional[float]:
       
        """ Calculate total end-to-end delay based on the embedding."""
        accumulated_latency = 0.0
        
        if path_embeddings is not None:
            # Compute latency from path embeddings
            for path_key, path in path_embeddings.items():
                for i in range(len(path) - 1):
                    link = pn.get_link(path[i], path[i + 1])
                    if link:
                        accumulated_latency += link.delay_ms

        return accumulated_latency