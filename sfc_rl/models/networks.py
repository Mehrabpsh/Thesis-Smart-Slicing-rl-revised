"""Policy networks for DQN."""

from abc import ABC, abstractmethod
from typing import Optional, Union
from networkx import union
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class PolicyNetwork(ABC):
    """Abstract base class for policy networks."""
    
    @abstractmethod
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action_mask: Optional action mask [batch_size, action_dim]
            
        Returns:
            Q-values [batch_size, action_dim]
        """
        pass
    
    @abstractmethod
    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """Get action from state.
        
        Args:
            state: State tensor [state_dim]
            action_mask: Optional action mask [action_dim]
            
        Returns:
            Action index
        """
        pass


class MLPPolicyNetwork(nn.Module, PolicyNetwork):
    """Multi-Layer Perceptron policy network with optional dueling architecture."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Union[int,list[int]] = [256, 256],
        activation: str = "relu",
        dueling: bool = False,
    ):
        """Initialize MLP policy network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            dueling: Whether to use dueling architecture
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        input_dim = state_dim
        print(isinstance(hidden_sizes, list))
        
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation())
            input_dim = hidden_size


        self.feature_layers = nn.Sequential(*layers)
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_head = nn.Linear(input_dim, 1)
            self.advantage_head = nn.Linear(input_dim, action_dim)
        else:
            # Standard Q-network
            self.q_head = nn.Linear(input_dim, action_dim)
    
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action_mask: Optional action mask [batch_size, action_dim]
            
        Returns:
            Q-values [batch_size, action_dim]
        """
        features = self.feature_layers(state)
        
        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            q_values = self.q_head(features)
        
        # Apply action mask (set invalid actions to very negative value)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
        
        return q_values
    
    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """Get action from state (greedy).
        
        Args:
            state: State tensor [state_dim]
            action_mask: Optional action mask [action_dim]
            
        Returns:
            Action index
        """
        self.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            
            q_values = self.forward(state, action_mask)
            action = q_values.argmax(dim=1).item()
        return action

