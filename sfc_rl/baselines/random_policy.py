"""Random baseline policy."""

from typing import Optional
import numpy as np
import time 

from ..env.sfc_env import SFCEnv, SFCEnvRevised


class RandomPolicy:
    """Random policy that samples uniformly from valid actions."""
    
    _class_name = "RandomPolicy"  # class attribute

    def __init__(self, seed: Optional[int] = None):
        """Initialize random policy.
        
        Args:
            seed: Optional random seed
        """
        self.rng = np.random.default_rng(seed)
        self.solutionTime : float = 0

    def act(self, env: SFCEnvRevised) -> int:
        """Select a random valid action.
        
        Args:
            env: SFC environment
            
        Returns:
            Action index
        """
        start_time = time.time()
        action_mask = env.action_mask()
        valid_actions = np.where(action_mask > 0)[0]
        
        if len(valid_actions) == 0:
            self.solutionTime = time.time() - start_time    
            return 0  # Fallback
        
        action = int(self.rng.choice(valid_actions))
        self.solutionTime = time.time() - start_time    
        return action




    @property
    def name(self) -> str:
        return self._class_name