"""Experience replay buffer for DQN."""

from typing import Optional, Tuple
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer : deque = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            action_mask: Optional action mask
            next_action_mask: Optional next action mask
        """
        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done,
            action_mask,
            next_action_mask,
        ))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, action_masks, next_action_masks)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.BoolTensor([e[4] for e in batch])
        
        action_masks = None
        if batch[0][5] is not None:
            action_masks = torch.FloatTensor(np.array([e[5] for e in batch]))
        
        next_action_masks = None
        if batch[0][6] is not None:
            next_action_masks = torch.FloatTensor(np.array([e[6] for e in batch]))
        
        return states, actions, rewards, next_states, dones, action_masks, next_action_masks
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)

