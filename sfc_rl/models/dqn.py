"""DQN agent implementation."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from .networks import PolicyNetwork, MLPPolicyNetwork
from .replay_buffer import ReplayBuffer


class DQNPolicy:
    """DQN policy with epsilon-greedy exploration."""
    
    _class_name = "DQN" 
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network: PolicyNetwork,
        config: DictConfig,  
        device: Optional[torch.device] = None,
    ):
        """Initialize DQN policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            network: Policy network
            config: DQN configuration
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = network.to(self.device)
        #if train_mode:
        self.target_network = self._create_target_network()
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.buffer_size = config.dqn.buffer_size

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)
        
        # Optimizer
        
        optimizer_name = config.optimizer.name.lower()
        lr = config.optimizer.lr
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.q_network.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Training parameters
        self.gamma = config.dqn.gamma
        self.batch_size = config.dqn.batch_size
        self.eps_start = config.dqn.eps_start
        self.eps_end = config.dqn.eps_end
        self.eps_decay_steps = config.dqn.eps_decay_steps
        self.target_update_freq = config.dqn.target_update
        self.double_dqn = config.dqn.get("double", False)
        self.n_step = config.dqn.get("n_step", 1)

        # Training state
        self.step_count = 0
        self.epsilon = self.eps_start
      
    
    
    def _create_target_network(self) -> PolicyNetwork:
        """Create target network (copy of Q-network)."""
        # if isinstance(self.q_network, MLPPolicyNetwork):
        #     return MLPPolicyNetwork(
        #         state_dim=self.state_dim,
        #         action_dim=self.action_dim,
        #         hidden_sizes=[self.q_network.feature_layers[2].out_features, self.q_network.feature_layers[2].out_features] if hasattr(self.q_network, 'feature_layers') else [256, 256],
        #         activation="relu",
        #         dueling=self.q_network.dueling,
        #     )
        # else:
        #     # Fallback: create a copy
        #     import copy
        #     return copy.deepcopy(self.q_network)
        import copy
        return copy.deepcopy(self.q_network)
    
    
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            action_mask: Optional action mask
            training: Whether in training mode
            
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            # Random action (from valid actions)
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    return int(np.random.choice(valid_actions))
                else:
                    return 0
            else:
                return np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if action_mask is not None:
                action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            else:
                action_mask_tensor = None
            
            return self.q_network.get_action(state_tensor, action_mask_tensor)
    
    def learn(self) -> Optional[float]:
        """Perform one learning step.
        
        Returns:
            Loss value if learning occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        if action_masks is not None:
            action_masks = action_masks.to(self.device)
        if next_action_masks is not None:
            next_action_masks = next_action_masks.to(self.device)
        
        # Current Q values
        q_values = self.q_network.forward(states, action_masks)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        

        # Next Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use Q-network to select action, target network to evaluate
                next_q_values = self.q_network.forward(next_states, next_action_masks)
                next_actions = next_q_values.argmax(dim=1)
                next_q_value = self.target_network.forward(next_states, next_action_masks).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network.forward(next_states, next_action_masks)
                next_q_value = next_q_values.max(dim=1)[0]
            
            target_q_value = rewards + (self.gamma ** self.n_step) * next_q_value #* (1- dones)
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.step_count += 1
        if self.step_count <= self.eps_decay_steps:
            self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * (self.step_count / self.eps_decay_steps)
        else:
            self.epsilon = self.eps_end
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

           
        
    def save(self, path: str) -> None:
        """Save policy to file.
        
        Args:
            path: Path to save file
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }, path)
    
    def load(self, path: str) -> None:
        """Load policy from file.
        
        Args:
            path: Path to load file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint.get("step_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.eps_end)


    @property
    def name(self) -> str:
        return self._class_name