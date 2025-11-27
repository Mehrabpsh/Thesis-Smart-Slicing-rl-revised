"""QoE (Quality of Experience) models - black box implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from omegaconf import DictConfig

from ..data.schemas import PhysicalNetwork, VNRequest
import math


class QoEModel(ABC):
    """Abstract base class for QoE computation."""
    
    @abstractmethod
    def compute(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Dict[int, int],  # vnf_id -> pn_node_id
        path_embeddings: Dict[tuple, list],  # (vnf_i, vnf_j) -> [node_ids]
    ) -> float:
        """Compute QoE for a given embedding.
        
        Args:
            pn: Physical Network
            vn_request: Virtual Network request
            embedding: VNF to PN node mapping
            path_embeddings: Path embeddings between consecutive VNFs
            
        Returns:
            QoE value (higher is better)
        """
        pass



class QoE_QoS_PaperModel(QoEModel):
    """QoE computation based on the paper's QoE/QoS correlation model."""

    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize the QoE model.

        Args:
            conf: Configuration object containing model parameters.
                  Expected keys (with defaults):
                  - qoe_weight_delay (float): Weight for delay (negative metric).
                  - qoe_weight_bandwidth (float): Weight for bandwidth (positive metric).
                  - alpha_n (float): Parameter for IQX (delay).
                  - beta_n (float): Parameter for IQX (delay).
                  - gamma_n (float): Parameter for IQX (delay).
                  - theta_n (float): Parameter for IQX (delay).
                  - alpha_p (float): Parameter for WFL (bandwidth).
                  - beta_p (float): Parameter for WFL (bandwidth).
                  - gamma_p (float): Parameter for WFL (bandwidth).
                  - theta_p (float): Parameter for WFL (bandwidth).
        """
        self.conf = config
        # Use config values or defaults based on our derivation
        self.w_delay = getattr(config, 'qoe_weight_delay', 0.5) # Negative weight for delay (bad -> subtract)
        self.w_bw = getattr(config, 'qoe_weight_bandwidth', 0.5) # Positive weight for bandwidth (good -> add)
        
        # IQX (Delay) parameters (from our derivation)
        self.alpha_n = getattr(config, 'alpha_n', -math.log(2) / 100.0) # -0.006931
        self.beta_n = getattr(config, 'beta_n', 0.0)
        self.gamma_n = getattr(config, 'gamma_n', 4.0)
        self.theta_n = getattr(config, 'theta_n', 1.0)
        
        # WFL (Bandwidth) parameters (from our derivation)
        self.alpha_p = getattr(config, 'alpha_p', 0.02484)
        self.beta_p = getattr(config, 'beta_p', 1.185)
        self.gamma_p = getattr(config, 'gamma_p', 1.923)
        self.theta_p = getattr(config, 'theta_p', 1.0)

    def compute(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Dict[int, int],  # vnf_id -> pn_node_id
        path_embeddings: Dict[tuple, list],  # (vnf_i, vnf_j) -> [node_ids]
    ) -> float:
        """Compute QoE based on the paper's formula.

        QoE_sfcic = sum(w_t * QoE_t^sfcic for positive t) - sum(w_t * QoE_t^sfcic for negative t)
        Using:
        - QoE_delay = gamma_n * e^(alpha_n * delay + beta_n) + theta_n (negative metric)
        - QoE_bw = gamma_p * ln(alpha_p * bw + beta_p) + theta_p (positive metric)

        Args:
            pn: Physical Network
            vn_request: Virtual Network request
            embedding: VNF to PN node mapping
            path_embeddings: Path embeddings between consecutive VNFs

        Returns:
            QoE value (higher is better)
        """
        # Calculate achieved metrics based on embedding and path
        achieved_delay = self._calculate_achieved_delay(pn, path_embeddings)
        achieved_bandwidth = vn_request.bandwidth_demand

        # Calculate QoE components using calibrated models
        qoe_delay = 0.0
        if achieved_delay is not None:
            qoe_delay = (
                self.gamma_n * math.exp(self.alpha_n * achieved_delay + self.beta_n)
                + self.theta_n
            )

        qoe_bandwidth = 0.0
        if achieved_bandwidth is not None:
            # Ensure argument to log is positive
            log_arg = self.alpha_p * achieved_bandwidth + self.beta_p
            if log_arg > 0:
                 qoe_bandwidth = (
                     self.gamma_p * math.log(log_arg) + self.theta_p
                 )
            else:
                 # Handle invalid log argument gracefully, maybe return a very low QoE
                 print(f"Warning: Invalid log argument for bandwidth: {log_arg}. Setting QoE_bw to 0.")
                 qoe_bandwidth = 0.0

        # Apply weights as per Eq. (8)
        # Assuming delay is negative (subtract) and bandwidth is positive (add)
        total_qoe = (
            self.w_bw * qoe_bandwidth # Positive metric contribution
            + self.w_delay * qoe_delay # Negative metric contribution (w_delay is negative)
        )

        return total_qoe

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

