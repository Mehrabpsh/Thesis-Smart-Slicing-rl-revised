"""SFC Environment - Gym-like API for Service Function Chaining VNE."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import networkx as nx
from omegaconf import DictConfig
from logging import Logger

from ..data.schemas import PhysicalNetwork, VNRequest
from ..data.lazy_loader import LazyVNRequestList
from .state_encoders import NormalizedStateEncoder, StateEncoder
from .action_space import ActionSpace
from .reward import RewardFn
from .qoe import QoEModel


class SFCEnv:
    """Service Function Chaining Virtual Network Embedding Environment.
    
    Gym-like API: reset(), step(action), action_mask(), render(), seed()
    """
    
    def __init__(
        self,
        pn: PhysicalNetwork,
        vn_requests: LazyVNRequestList, #List[VNRequest],
        state_encoder: StateEncoder,
        action_space: ActionSpace,
        reward_fn: RewardFn,
        qoe_model: QoEModel,
        max_steps_per_request: int = 100,
    ):
        """Initialize SFC environment.
        
        Args:
            pn: Physical Network
            vn_requests: List of VN requests
            state_encoder: State encoder
            action_space: Action space
            reward_fn: Reward function
            qoe_model: QoE model
            max_steps_per_request: Maximum steps per request
        """
        self.pn = pn
        self.vn_requests = vn_requests
        self.state_encoder = state_encoder
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.qoe_model = qoe_model
        self.max_steps_per_request = max_steps_per_request
        
        # State
        self.current_request_idx: int = 0
        self.current_vnf_idx: int = 0
        self.partial_embedding: Dict[int, int] = {}  # vnf_id -> pn_node_id
        self.path_embeddings: Dict[tuple, list] = {}  # (vnf_i, vnf_j) -> [node_ids]
        self.step_count: int = 0
        self.episode_step_count: int = 0
        self._rng: Optional[np.random.Generator] = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset PN resources
        self.pn.reset()
        
        # Reset request tracking
        self.current_request_idx = 0
        self.current_vnf_idx = 0
        self.partial_embedding = {}
        self.path_embeddings = {}
        self.step_count = 0
        self.episode_step_count = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """

        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._

        if self.current_request_idx >= len(self.vn_requests):
            # Episode done
            return self._get_observation(), 0.0, True, False, self._get_info()

        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._


        
        current_request = self.vn_requests[self.current_request_idx]
        self.step_count += 1
        self.episode_step_count += 1
        
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._

        # Check if we've exceeded max steps
        if self.episode_step_count >= self.max_steps_per_request:
            # Reject current request and move to next
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, False, True, info
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._




        # Convert action to PN node ID
        try:
            pn_node_id = self.action_space.action_to_embedding(
                action, self.pn, current_request, self.current_vnf_idx
            )
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._
        except ValueError:
            # Invalid action - reject request
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, False, False, info
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._




        # Check if action is valid
        valid_actions = self.action_space.get_valid_actions(
            self.pn, current_request, self.current_vnf_idx, self.partial_embedding
        )
        node_ids = sorted(self.pn.nodes.keys())

        # node resource constraint violation
        if pn_node_id not in valid_actions:
            # Invalid action - reject request
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, False, False, info
        




        # Embed current VNF
        vnf = current_request.get_vnf(self.current_vnf_idx)
        try:
            # Allocate CPU
            node = self.pn.get_node(pn_node_id)
            node.allocate(vnf.cpu_demand)
            self.partial_embedding[self.current_vnf_idx] = pn_node_id
            
            # If not first VNF, embed path
            if self.current_vnf_idx > 0:
                prev_vnf_idx = self.current_vnf_idx - 1
                prev_node_id = self.partial_embedding[prev_vnf_idx]
                
                # Find shortest path
                path = self._find_path(prev_node_id, pn_node_id, current_request.bandwidth_demand)
                if path is None:
                    # No path found - rollback and reject
                    node.deallocate(vnf.cpu_demand)
                    del self.partial_embedding[self.current_vnf_idx]
                    reward = self.reward_fn.compute(
                        self.pn, current_request, None, None, False
                    )
                    self._move_to_next_request()
                    obs = self._get_observation()
                    info = self._get_info()
                    return obs, reward, False, False, info
                


                # Allocate bandwidth on path
                try:
                    for i in range(len(path) - 1):
                        link = self.pn.get_link(path[i], path[i + 1])
                        if link:
                            link.allocate(current_request.bandwidth_demand)
                    self.path_embeddings[(prev_vnf_idx, self.current_vnf_idx)] = path

                #link/path resource constraint violation

                except ValueError:
                    # Insufficient bandwidth - rollback and reject
                    node.deallocate(vnf.cpu_demand)
                    del self.partial_embedding[self.current_vnf_idx]
                    reward = self.reward_fn.compute(
                        self.pn, current_request, None, None, False
                    )
                    self._move_to_next_request()
                    obs = self._get_observation()
                    info = self._get_info()
                    return obs, reward, False, False, info
            






            # Move to next VNF
            self.current_vnf_idx += 1
            
            # Check if request is complete
            if self.current_vnf_idx >= len(current_request.vnfs):
                # Request completed successfully
                reward = self.reward_fn.compute(
                    self.pn, current_request, self.partial_embedding.copy(), self.path_embeddings.copy(), True
                )
                self._move_to_next_request()
                obs = self._get_observation()
                info = self._get_info()
                terminated = self.current_request_idx >= len(self.vn_requests)
                return obs, reward, terminated, False, info


            else:
                # Continue embedding
                obs = self._get_observation()
                info = self._get_info()
                return obs, 0.0, False, False, info

        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._
        
        except ValueError as e:
            # Resource allocation failed - reject request
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, False, False, info
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._




    
    def action_mask(self) -> np.ndarray:
        """Get action mask for current state.
        
        Returns:
            Binary mask array (1 for valid, 0 for invalid)
        """
        if self.current_request_idx >= len(self.vn_requests):
            return np.zeros(self.action_space.get_action_dim(self.pn), dtype=np.float32)
        
        current_request = self.vn_requests[self.current_request_idx]
        return self.action_space.get_action_mask(
            self.pn, current_request, self.current_vnf_idx, self.partial_embedding
        )
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            Optional string representation
        """
        if mode == "human":
            print(f"Request: {self.current_request_idx}/{len(self.vn_requests)}")
            print(f"VNF: {self.current_vnf_idx}/{len(self.vn_requests[self.current_request_idx].vnfs) if self.current_request_idx < len(self.vn_requests) else 0}")
            print(f"Embedding: {self.partial_embedding}")
            return None
        elif mode == "ansi":
            return f"Request: {self.current_request_idx}, VNF: {self.current_vnf_idx}, Embedding: {self.partial_embedding}"
        else:
            raise ValueError(f"Unknown render mode: {mode}")
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed.
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation array
        """
        if self.current_request_idx >= len(self.vn_requests):
            # Return zero observation when done
            return np.zeros(self.state_encoder.get_state_dim(self.pn), dtype=np.float32)
        
        current_request = self.vn_requests[self.current_request_idx]
        return self.state_encoder.encode(
            self.pn, 
            current_request, 
            self.current_vnf_idx, 
            self.partial_embedding,
            self.path_embeddings  # Pass path embeddings for latency computation
        )
    
    def _get_info(self) -> Dict:
        """Get info dictionary.
        
        Returns:
            Info dict
        """
        return {
            "current_request_idx": self.current_request_idx,
            "current_vnf_idx": self.current_vnf_idx,
            "step_count": self.step_count,
            "episode_step_count": self.episode_step_count,
            "partial_embedding": self.partial_embedding.copy(),
        }
    
    def _move_to_next_request(self) -> None:
        """Move to next request (rollback current if incomplete)."""
        if self.current_request_idx < len(self.vn_requests):
            current_request = self.vn_requests[self.current_request_idx]
            # Rollback resources if request was not completed
            if len(self.partial_embedding) < len(current_request.vnfs):
                # Deallocate resources
                for vnf_id, pn_node_id in self.partial_embedding.items():
                    vnf = current_request.get_vnf(vnf_id)
                    node = self.pn.get_node(pn_node_id)
                    node.deallocate(vnf.cpu_demand)
                
                for path_key, path in self.path_embeddings.items():
                    for i in range(len(path) - 1):
                        link = self.pn.get_link(path[i], path[i + 1])
                        if link:
                            link.deallocate(current_request.bandwidth_demand)
        
        # Move to next request
        self.current_request_idx += 1
        self.current_vnf_idx = 0
        self.partial_embedding = {}
        self.path_embeddings = {}
        self.episode_step_count = 0
    
    def _find_path(
        self, src: int, dst: int, bw_demand: float
    ) -> Optional[List[int]]:
        """Find a path between two nodes with sufficient bandwidth.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            bw_demand: Required bandwidth
            
        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        try:
            # Use shortest path with bandwidth constraint
            paths = list(nx.all_simple_paths(self.pn.graph, src, dst, cutoff=10))
            for path in paths:
                # Check if all links have sufficient bandwidth
                valid = True
                for i in range(len(path) - 1):
                    link = self.pn.get_link(path[i], path[i + 1])
                    if link is None or not link.can_carry(bw_demand):
                        valid = False
                        break
                if valid:
                    return path
            
            # Fallback to shortest path (will fail later if no bandwidth)
            try:
                return nx.shortest_path(self.pn.graph, src, dst)
            except nx.NetworkXNoPath:
                return None
        except Exception:
            return None






class SFCEnvRevised:
    """Service Function Chaining Virtual Network Embedding Environment.
    
    Gym-like API: reset(), step(action), action_mask(), render(), seed()
    """
    
    def __init__(
        self,
        pn: PhysicalNetwork,
        vn_requests: LazyVNRequestList,
        num_groups: int,
        #state_encoder: StateEncoder,
        state_encoder: NormalizedStateEncoder,
        action_space: ActionSpace,
        reward_fn: RewardFn,
        qoe_model: QoEModel,
        max_steps_per_request: int = 100,
    ):
        """Initialize SFC environment.
        
        Args:
            pn: Physical Network
            vn_requests: List of VN requests
            state_encoder: State encoder
            action_space: Action space
            reward_fn: Reward function
            qoe_model: QoE model
            max_steps_per_request: Maximum steps per request
        """
        self.pn = pn
        self.vn_requests = vn_requests
        self.num_groups = num_groups
        self.state_encoder = state_encoder
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.qoe_model = qoe_model
        self.max_steps_per_request = max_steps_per_request

        self.max_residual_bw : float = 0
        
        # State
        self.current_request_idx: int = 0
        self.current_vnf_idx: int = 0
        self.group_id: int = 0
        self.current_vn_requests: List[VNRequest] = []
        self.partial_embedding: Dict[int, int] = {}  # vnf_id -> pn_node_id
        self.path_embeddings: Dict[tuple, list] = {}  # (vnf_i, vnf_j) -> [node_ids]
        self.step_count: int = 0
        self.episode_step_count: int = 0
        self._rng: Optional[np.random.Generator] = None
    
        self.embedding_state = None #Union['success','fail', 'None']

    def reset(self,group_id:int, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset PN resources
        self.pn.reset()
        
        # Reset request tracking
        self.current_request_idx = 0
        self.current_vnf_idx = 0
        self.partial_embedding = {}
        self.path_embeddings = {}
        self.step_count = 0
        self.episode_step_count = 0
        self.group_id = group_id
        # Get initial observation
        self.current_vn_requests = self.vn_requests[self.group_id*(len(self.vn_requests)//self.num_groups):(self.group_id+1)*(len(self.vn_requests)//self.num_groups)]
        print(f"vn_requests[{self.group_id*(len(self.vn_requests)//self.num_groups)}:{(self.group_id+1)*(len(self.vn_requests)//self.num_groups)}]")
        self.current_vn_requests = list(self.current_vn_requests)
        self.max_residual_bw = float(np.max([link.available_bandwidth for link in self.pn.links]))
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """

        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._

        if self.current_request_idx >= len(self.current_vn_requests):
            # Episode done
            print(f'request of group {self.group_id} Terminated due to current_request_idx > current_vn_requests \n')
            return self._get_observation(), 0.0, True, False, self._get_info() # Terminated = True

        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._


        
        current_request = self.current_vn_requests[self.current_request_idx]
        self.step_count += 1
        self.episode_step_count += 1
        
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._

        # Check if we've exceeded max steps
        if self.episode_step_count >= self.max_steps_per_request:
            # Reject current request and move to next
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            print(f'request of group {self.group_id} Truncated due to exceed of  max steps \n')
            return obs, reward, False, True, info   # Truncated = True
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._




        # Convert action to PN node ID
        try:
            pn_node_id = self.action_space.action_to_embedding(
                action, self.pn, current_request, self.current_vnf_idx
            )
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._
        except ValueError:
            # Invalid action - reject request
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self.embedding_state = 'fail'
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            Terminated = self.current_request_idx >= len(self.current_vn_requests)
            return obs, reward, Terminated, False, info
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._




        # Check if action is valid
        valid_actions = self.action_space.get_valid_actions(
            self.pn, current_request, self.current_vnf_idx, self.partial_embedding
        )
        node_ids = sorted(self.pn.nodes.keys())

        # node resource constraint violation
        if pn_node_id not in valid_actions:
            # Invalid action - reject request
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self.embedding_state = 'fail'
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            Terminated = self.current_request_idx >= len(self.current_vn_requests)
            return obs, reward, Terminated, False, info  
        




        # Embed current VNF
        vnf = current_request.get_vnf(self.current_vnf_idx)
        try:
            # Allocate CPU
            node = self.pn.get_node(pn_node_id)
            node.allocate(vnf.cpu_demand)
            self.partial_embedding[self.current_vnf_idx] = pn_node_id
            
            # If not first VNF, embed path
            if self.current_vnf_idx > 0:
                prev_vnf_idx = self.current_vnf_idx - 1
                prev_node_id = self.partial_embedding[prev_vnf_idx]
                
                # Find shortest path
                path = self._find_path(prev_node_id, pn_node_id, current_request.bandwidth_demand)
                if path is None:#or prev_node_id == node_:
                    # No path found - rollback and reject
                    node.deallocate(vnf.cpu_demand)
                    del self.partial_embedding[self.current_vnf_idx]
                    reward = self.reward_fn.compute(
                        self.pn, current_request, None, None, False
                    )
                    self.embedding_state = 'fail'
                    self._move_to_next_request()
                    obs = self._get_observation()
                    info = self._get_info()
                    Terminated = self.current_request_idx >= len(self.current_vn_requests)
                    return obs, reward, Terminated, False, info
                


                # Allocate bandwidth on path
                try:
                    for i in range(len(path) - 1):
                        link = self.pn.get_link(path[i], path[i + 1])
                        if link:
                            link.allocate(current_request.bandwidth_demand)
                    self.path_embeddings[(prev_vnf_idx, self.current_vnf_idx)] = path

                #link/path resource constraint violation

                except ValueError:
                    # Insufficient bandwidth - rollback and reject
                    node.deallocate(vnf.cpu_demand)
                    del self.partial_embedding[self.current_vnf_idx]
                    reward = self.reward_fn.compute(
                        self.pn, current_request, None, None, False
                    )
                    self.embedding_state = 'fail'
                    self._move_to_next_request()
                    obs = self._get_observation()
                    info = self._get_info()
                    Terminated = self.current_request_idx >= len(self.current_vn_requests)
                    return obs, reward, Terminated, False, info
            






            # Move to next VNF
            self.current_vnf_idx += 1
            
            # Check if request is complete
            if self.current_vnf_idx >= current_request.sfc_length:
                # Request completed successfully
                reward = self.reward_fn.compute(
                    self.pn, current_request, self.partial_embedding.copy(), self.path_embeddings.copy(), True
                )
                self.embedding_state = 'success'
                self._move_to_next_request()
                obs = self._get_observation()
                info = self._get_info()
                #terminated = self.current_request_idx >= len(self.vn_requests)
                #return obs, reward, terminated, False, info
                Terminated = self.current_request_idx >= len(self.current_vn_requests)
                return obs, reward, Terminated, False, info


            else:
                # Continue embedding
                self.embedding_state = None
                obs = self._get_observation()
                info = self._get_info()
                Terminated = self.current_request_idx >= len(self.current_vn_requests)
                return obs, 0.0, Terminated, False, info

        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._
        
        except ValueError as e:
            # Resource allocation failed - reject request
            reward = self.reward_fn.compute(
                self.pn, current_request, None, None, False
            )
            self.embedding_state = 'rejected'
            self._move_to_next_request()
            obs = self._get_observation()
            info = self._get_info()
            Terminated = self.current_request_idx >= len(self.current_vn_requests)
            return obs, reward, Terminated, False, info
        #._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._


    
    def action_mask(self) -> np.ndarray:
        """Get action mask for current state.
        
        Returns:
            Binary mask array (1 for valid, 0 for invalid)
        """
        if self.current_request_idx >= len(self.vn_requests):
            return np.zeros(self.action_space.get_action_dim(self.pn), dtype=np.float32)
        
        current_request = self.current_vn_requests[self.current_request_idx]
        return self.action_space.get_action_mask(
            self.pn, current_request, self.current_vnf_idx, self.partial_embedding
        )
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            Optional string representation
        """
        if mode == "human":
            print(f"Request: {self.current_request_idx}/{len(self.vn_requests)}")
            print(f"VNF: {self.current_vnf_idx}/{len(self.vn_requests[self.current_request_idx].vnfs) if self.current_request_idx < len(self.vn_requests) else 0}")
            print(f"Embedding: {self.partial_embedding}")
            return None
        elif mode == "ansi":
            return f"Request: {self.current_request_idx}, VNF: {self.current_vnf_idx}, Embedding: {self.partial_embedding}"
        else:
            raise ValueError(f"Unknown render mode: {mode}")
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed.
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation array
        """
        if self.current_request_idx >= len(self.current_vn_requests):
            # Return zero observation when done
            return np.zeros(self.state_encoder.get_state_dim(self.pn), dtype=np.float32)
        
        current_request = self.current_vn_requests[self.current_request_idx]

        return self.state_encoder.encode(
            self.pn, 
            current_request,
            self.max_residual_bw, 
            self.current_vnf_idx, 
            self.partial_embedding,
            self.path_embeddings  # Pass path embeddings for latency computation
        )
    
    def _get_info(self) -> Dict:
        """Get info dictionary.
        
        Returns:
            Info dict
        """
        return {
            "current_request_idx": self.current_request_idx,
            "current_vnf_idx": self.current_vnf_idx,
            "step_count": self.step_count,
            "episode_step_count": self.episode_step_count,
            "partial_embedding": self.partial_embedding.copy(),
            'embedding_state': self.embedding_state,
        }
    
    def _move_to_next_request(self) -> None:
        """Move to next request (rollback current if incomplete)."""
        #if self.current_request_idx < len(self.current_vn_requests):
        current_request = self.current_vn_requests[self.current_request_idx]
        # Rollback resources if request was not completed
        if len(self.partial_embedding) < len(current_request.vnfs):
            # Deallocate resources
            for vnf_id, pn_node_id in self.partial_embedding.items():
                vnf = current_request.get_vnf(vnf_id)
                node = self.pn.get_node(pn_node_id)
                node.deallocate(vnf.cpu_demand)
            
            for path_key, path in self.path_embeddings.items():
                for i in range(len(path) - 1):
                    link = self.pn.get_link(path[i], path[i + 1])
                    if link:
                        link.deallocate(current_request.bandwidth_demand)
        
        # Move to next request
        self.current_request_idx += 1
        self.current_vnf_idx = 0
        self.partial_embedding = {}
        self.path_embeddings = {}
        self.episode_step_count = 0
        self.max_residual_bw = float(np.max([link.available_bandwidth for link in self.pn.links]))

        """ elif self.current_request_idx >= len(self.current_vn_requests):

            self.current_request_idx = 0
            self.group_id += 1
            self.current_vnf_idx = 0
            self.partial_embedding = {}
            self.path_embeddings = {}
            self.episode_step_count = 0"""
            
    
    def _find_path(
        self, src: int, dst: int, bw_demand: float
    ) -> Optional[List[int]]:
        """Find a path between two nodes with sufficient bandwidth.
        
        Args:
            src: Source node ID
            dst: Destination node ID
            bw_demand: Required bandwidth
            
        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        try:
            # Use shortest path with bandwidth constraint
            paths = list(nx.all_simple_paths(self.pn.graph, src, dst, cutoff=10))
            for path in paths:
                # Check if all links have sufficient bandwidth
                valid = True
                for i in range(len(path) - 1):
                    link = self.pn.get_link(path[i], path[i + 1])
                    if link is None or not link.can_carry(bw_demand):
                        valid = False
                        break
                if valid:
                    return path
            
            # Fallback to shortest path (will fail later if no bandwidth)
            try:
                return nx.shortest_path(self.pn.graph, src, dst)
            except nx.NetworkXNoPath:
                return None
        except Exception:
            return None


