"""Exhaustive/Violent search baseline solver."""

from typing import Optional, Dict, List, Tuple
import time
import numpy as np
import networkx as nx

from ..env.sfc_env import SFCEnv, SFCEnvRevised
from ..data.schemas import PhysicalNetwork, VNRequest


class ExhaustiveSolver:
    """Exhaustive search solver that enumerates all feasible embeddings using tree search.
    
    Implements a tree search where:
    - Each level represents a decision for a VNF in the request sequence
    - At each level, branches to all feasible next steps (placing VNF on PSN node)
    - Continues until complete embedding paths are formed (all VNFs placed, all links mapped)
    - Evaluates each complete path against all constraints and objective function
    - Returns the embedding with highest objective function value
    """
    
    def __init__(self, max_embeddings: int = 10000, seed: Optional[int] = None):
        """Initialize exhaustive solver.
        
        Args:
            max_embeddings: Maximum number of embeddings to enumerate
            seed: Optional random seed
        """
        self.max_embeddings = max_embeddings
        self.rng = np.random.default_rng(seed)

        self._requestID: int = -1
        self._buffer : Optional[Tuple[Dict[int, int], Dict[tuple, list]]] = None
        self._groupID : int = -1
        self.solutionTime : float = 0

    def solve(self, env: SFCEnvRevised) -> int:
        """Solve current request using exhaustive search.
        
        Args:
            env: SFC environment (SFCEnv or SFCEnvRevised)
            
        Returns:
            Best action (first action of best embedding)
        """

        # Handle both SFCEnv and SFCEnvRevised
        if hasattr(env, 'current_vn_requests') and env.current_vn_requests:
            # SFCEnvRevised uses current_vn_requests for the current group
            vn_requests_list = env.current_vn_requests
        else:
            # SFCEnv uses vn_requests
            vn_requests_list = env.vn_requests
        
        if env.current_request_idx >= len(vn_requests_list):
            return 0
        
        current_request = vn_requests_list[env.current_request_idx]


        if self._buffer is not None and self._requestID == env.current_request_idx  and self._groupID== env.group_id and env.current_vnf_idx < len(current_request.vnfs):
            embedding, _ = self._buffer
            next_node_id = embedding.get(env.current_vnf_idx)
            if next_node_id is not None:
                # Convert node_id to action index
                node_ids = sorted(env.pn.nodes.keys())
                if next_node_id in node_ids:
                    return node_ids.index(next_node_id)

                
        
        start_time = time.time()
        best_solution = None
        best_objective = float('-inf')
        
        # Generate all possible complete embeddings using tree search
        complete_solutions = self._tree_search_embeddings(
            env.pn,
            current_request,
            env.current_vnf_idx,
            env.partial_embedding,
        )
        
        # Evaluate each complete solution
        for embedding, path_embeddings in complete_solutions:
            
            # Check all final constraints
            if not self._check_all_constraints(env.pn, current_request, embedding, path_embeddings):
                # Infeasible solution - assign very low value
                objective_value = float('-inf')
            else:
                # Feasible solution - compute objective function (QoE/QoS Reward)
                try:
                    objective_value = env.reward_fn.compute(
                        env.pn, current_request, embedding, path_embeddings, True
                    )
                except Exception:
                    objective_value = float('-inf')
            
            if objective_value > best_objective:
                best_objective = objective_value
                best_solution = (embedding, path_embeddings)
        
        self._buffer = best_solution
        self._requestID = env.current_request_idx
        self._groupID = env.group_id
        self.solutionTime = time.time() - start_time 

        # Return first action of best embedding
        if best_solution is not None and env.current_vnf_idx < len(current_request.vnfs):
            embedding, _ = best_solution
            next_node_id = embedding.get(env.current_vnf_idx)
            if next_node_id is not None:
                # Convert node_id to action index
                node_ids = sorted(env.pn.nodes.keys())
                if next_node_id in node_ids:
                    return node_ids.index(next_node_id)
        
        # Fallback: use random valid action
        action_mask = env.action_mask()
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) > 0:
            return int(self.rng.choice(valid_actions))
        return 0
    
    def _tree_search_embeddings(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        start_vnf_idx: int,
        partial_embedding: Dict[int, int],
    ) -> List[Tuple[Dict[int, int], Dict[tuple, list]]]:
        """Tree search to enumerate all feasible complete embeddings.
        
        Each level i represents the decision for the i-th VNF in the request sequence.
        At each level, branches to all feasible next steps (placing VNF on available PSN node).
        Continues until complete embedding paths are formed.
        
        Args:
            pn: Physical Network
            vn_request: VN request
            start_vnf_idx: Starting VNF index
            partial_embedding: Partial embedding so far
            start_time: Start time for timeout
            
        Returns:
            List of (embedding, path_embeddings) tuples for complete embeddings
        """
        complete_solutions: List[Tuple[Dict[int, int], Dict[tuple, list]]] = []
        
        def _recursive_tree_search(vnf_idx: int, current_embedding: Dict[int, int]):
            """Recursive tree search function.
            
            Args:
                vnf_idx: Current VNF index to place
                current_embedding: Current partial embedding (vnf_id -> pn_node_id)
            """
            
            # Base case: all VNFs placed, now find paths
            if vnf_idx >= len(vn_request.vnfs):
                # Complete embedding - find paths for all consecutive VNF pairs
                path_embeddings = self._find_all_paths(pn, vn_request, current_embedding)
                if path_embeddings is not None:
                    # Store complete solution
                    complete_solutions.append((current_embedding.copy(), path_embeddings))
                return
            
            # Get current VNF
            vnf = vn_request.get_vnf(vnf_idx)
            
            # Branch: try all feasible nodes for this VNF
            valid_nodes = self._get_feasible_nodes(
                pn, vn_request, vnf_idx, vnf, current_embedding
            )
            
            # Try each valid node (branching)
            for node_id in valid_nodes:
                # Place VNF on this node
                current_embedding[vnf_idx] = node_id
                
                # Recursively continue to next VNF
                _recursive_tree_search(vnf_idx + 1, current_embedding)
                
                # Backtrack: remove this placement
                del current_embedding[vnf_idx]
        
        # Start tree search from current state
        _recursive_tree_search(start_vnf_idx, partial_embedding.copy())
        return complete_solutions
    



    def _get_feasible_nodes(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        vnf_idx: int,
        vnf,
        current_embedding: Dict[int, int],
    ) -> List[int]:
        """Get all feasible nodes for placing the current VNF.
        
        Checks immediate constraints:
        - Node CPU capacity (accounting for VNFs already placed on the node)
        - Connectivity (if not first VNF, must be reachable from previous VNF)
        
        Args:
            pn: Physical Network
            vn_request: VN request
            vnf_idx: Current VNF index
            vnf: Current VNF object
            current_embedding: Current partial embedding
            
        Returns:
            List of feasible node IDs
        """
        valid_nodes = []
        
        for node_id, node in pn.nodes.items():
            # Calculate total CPU demand on this node (including already placed VNFs)
            total_cpu_demand = vnf.cpu_demand
            for placed_vnf_idx, placed_node_id in current_embedding.items():
                if placed_node_id == node_id:
                    placed_vnf = vn_request.get_vnf(placed_vnf_idx)
                    total_cpu_demand += placed_vnf.cpu_demand
            
            # Check CPU constraint
            if not node.can_host(total_cpu_demand):
                continue
            
            # For first VNF, any node with sufficient CPU is valid
            if vnf_idx == 0:
                valid_nodes.append(node_id)
            else:
                # For subsequent VNFs, check connectivity to previous VNF
                prev_vnf_idx = vnf_idx - 1
                prev_node_id = current_embedding.get(prev_vnf_idx)
                
                if prev_node_id is not None and (not(prev_node_id == node_id)):  # modified 
                    # Check if there's a path from previous node to current node
                    if pn.graph.has_edge(prev_node_id, node_id) or \
                       nx.has_path(pn.graph, prev_node_id, node_id):
                        valid_nodes.append(node_id)
        
        return valid_nodes
    
    def _find_all_paths(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Dict[int, int],
    ) -> Optional[Dict[tuple, list]]:
        """Find paths between all consecutive VNF pairs.
        
        For each consecutive VNF pair (i, i+1), finds a path from embedding[i] to embedding[i+1]
        that satisfies bandwidth constraints.
        
        Args:
            pn: Physical Network
            vn_request: VN request
            embedding: Complete VNF to node mapping
            
        Returns:
            Dictionary mapping (vnf_i, vnf_j) -> [node_ids] path, or None if infeasible
        """
        path_embeddings = {}
        
        for i in range(len(vn_request.vnfs) - 1):
            src = embedding.get(i)
            dst = embedding.get(i + 1)
            
            if src is None or dst is None:
                return None
            
            # Find a path with sufficient bandwidth
            path = self._find_feasible_path(pn, src, dst, vn_request.bandwidth_demand)
            if path is None:
                return None
            
            path_embeddings[(i, i + 1)] = path
        
        return path_embeddings
    
    def _find_feasible_path(
        self,
        pn: PhysicalNetwork,
        src: int,
        dst: int,
        bw_demand: float,
    ) -> Optional[List[int]]:
        """Find a path between two nodes with sufficient bandwidth.
        
        Args:
            pn: Physical Network
            src: Source node ID
            dst: Destination node ID
            bw_demand: Required bandwidth
            
        Returns:
            List of node IDs forming the path, or None if no feasible path exists
        """
        try:
            # Try to find all simple paths (up to reasonable length)
            # NetworkX returns generators that yield lists of node IDs
            paths_generator = nx.all_simple_paths(pn.graph, src, dst, cutoff=10)
            
            # Check each path for bandwidth availability
            for path in paths_generator:
                # Convert path to list of integers (node IDs)
                path_list = [int(node_id) for node_id in path]
                # Check if all links in path have sufficient bandwidth
                valid = True
                for j in range(len(path_list) - 1):
                    link = pn.get_link(path_list[j], path_list[j + 1])
                    if link is None or not link.can_carry(bw_demand):
                        valid = False
                        break
                
                if valid:
                    return path_list
            
            # No feasible path found
            return None
        except nx.NetworkXNoPath:
            return None
        except Exception:
            return None
    
    def _check_all_constraints(
        self,
        pn: PhysicalNetwork,
        vn_request: VNRequest,
        embedding: Dict[int, int],
        path_embeddings: Dict[tuple, list],
    ) -> bool:
        """Check all final constraints for a complete embedding.
        
        Checks:
        - Node CPU capacity (all VNFs can be placed, accounting for multiple VNFs per node)
        - Link bandwidth capacity (all paths have sufficient bandwidth, accounting for shared links)
        - End-to-end delay (total delay <= SLA latency)
        
        Args:
            pn: Physical Network
            vn_request: VN request
            embedding: Complete VNF to node mapping
            path_embeddings: Complete path embeddings
            
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        # Check 1: Node CPU constraints (aggregate CPU demand per node)
        node_cpu_demand: Dict[int, float] = {}
        for vnf_idx, node_id in embedding.items():
            vnf = vn_request.get_vnf(vnf_idx)
            if node_id not in node_cpu_demand:
                node_cpu_demand[node_id] = 0.0
            node_cpu_demand[node_id] += vnf.cpu_demand
        
        for node_id, total_cpu_demand in node_cpu_demand.items():
            node = pn.get_node(node_id)
            if not node.can_host(total_cpu_demand):
                return False
        
        # Check 2: Link bandwidth constraints (aggregate bandwidth demand per link)
        link_bw_demand: Dict[Tuple[int, int], float] = {}
        for path_key, path in path_embeddings.items():
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                # Use canonical ordering for undirected links
                link_key = (min(src, dst), max(src, dst))
                if link_key not in link_bw_demand:
                    link_bw_demand[link_key] = 0.0
                link_bw_demand[link_key] += vn_request.bandwidth_demand
        
        for (src, dst), total_bw_demand in link_bw_demand.items():
            link = pn.get_link(src, dst)
            if link is None or not link.can_carry(total_bw_demand):
                return False
        
        # Check 3: End-to-end delay constraint
        total_delay = 0.0
        for path_key, path in path_embeddings.items():
            for i in range(len(path) - 1):
                link = pn.get_link(path[i], path[i + 1])
                if link:
                    total_delay += link.delay_ms
        
        if total_delay > vn_request.sla_latency_ms:
            return False
        
        # All constraints satisfied
        return True








