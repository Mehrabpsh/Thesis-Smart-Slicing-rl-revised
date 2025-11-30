"""Evaluator for comparing multiple policies."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from ..env.sfc_env import SFCEnv, SFCEnvRevised
from ..models.dqn import DQNPolicy
from ..baselines.random_policy import RandomPolicy
from ..baselines.exhaustive_solver import ExhaustiveSolver
from .plots import plot_metrics, plot_training_curves
from ..utils.logging import setup_logger
from ..utils.seed import set_seed
from ..utils.serialization import save_json
from .metrics import Metrics

class Evaluator:
    """Evaluator for comparing policies."""
    
    def __init__(
        self,
        env: SFCEnvRevised,
        policies: Dict[str, Any],
        config: DictConfig,
        output_dir: Path,
        logger=None,
    ):
        """Initialize evaluator.
        
        Args:
            env: SFC environment
            policies: Dictionary mapping policy names to policy objects
            config: Evaluation configuration
            output_dir: Output directory
            logger: Optional logger
        """
        self.env = env
        self.policies = policies
        self.config = config
        self.output_dir = output_dir
        self.logger = logger or setup_logger("evaluator")
        
        #self.runs = config['runs'] 
        self.runs = config.get("runs", 10)
        
        self.metrics_to_compute = config.get("metrics", ["acceptance_ratio", "response_time", "qoe"])
    
    def evaluate(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate all policies.
        
        Returns:
            Dictionary mapping policy names to metrics
        """
        all_results = {}
        
        for policy_name, policy in self.policies.items():
            self.logger.info(f"Evaluating policy: {policy_name}")
            Episodes_Metrics = {}
            episode_rewards = []

            metrics = Metrics()

            for run in tqdm(range(self.runs), desc=f'Evaluating {policy_name} policy '):
                # Set seed for reproducibility
                seed = self.config.get("seed", 42) + run
                set_seed(seed)
                episode_reward = 0.0
                # Reset environment
                obs, info = self.env.reset(group_id = run ,seed=seed)
                                
                # Run evaluation
                while True : # self.env.current_request_idx < len(self.env.current_vn_requests):
                    # Select action
                    if isinstance(policy, DQNPolicy):
                        action = policy.act(obs, training=True) #False)
                    elif isinstance(policy, RandomPolicy):
                        action = policy.act(self.env)
                    elif isinstance(policy, ExhaustiveSolver):
                        action = policy.solve(self.env)
                    else:
                        raise ValueError(f"Unknown policy type: {type(policy)}")
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    episode_reward += reward

                    if not (reward == 0):
                        if self.env.current_request_idx > 0:
                            prev_idx = self.env.current_request_idx - 1
                            if prev_idx < len(self.env.current_vn_requests):
                                request = self.env.current_vn_requests[prev_idx]
                                accepted = reward > 0  # Simple heuristic
                                qoe = reward if accepted else None  # Could compute from embedding
                                if isinstance(policy, ExhaustiveSolver):
                                    response_time = policy.solutionTime
                                elif isinstance(policy, RandomPolicy):
                                    response_time = policy.solutionTime * self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length * 1e2
                                elif isinstance(policy, DQNPolicy ):
                                    response_time =  1* self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length

                                metrics.update(request, accepted, response_time, qoe, reward)
                                #metrics.update(request, accepted, qoe, reward)
                                print(f'request id: {self.env.current_request_idx-1} of group {self.env.group_id} {'embedded' if accepted == True else 'failed'} with reward {reward}')


                    
                    if terminated: # if all the request of the current group are hanlded
                        #avg_sum = float(np.mean(episode_reward))
                        #episode_rewards.append(avg_sum)
                        episode_rewards.append(episode_reward)
                        Episodes_Metrics[run] = metrics.compute()
                        metrics.reset()
                        break


                    obs = next_obs
      
            
            # Average across runs
            avg_metrics = {}
            for metric in self.metrics_to_compute:
                values = [r.get(metric, 0.0) for r in Episodes_Metrics.values()]
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f"{metric}_std"] = np.std(values)
            
            all_results[policy_name] = avg_metrics
        
        # Save results
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(all_results, self.output_dir / "evaluation_results.json")
        
        # Generate plots
        if self.config.get("report", {}).get("plots", True):
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_metrics(all_results, plots_dir, "metrics_comparison.png")
        
        # Save CSV
        if self.config.get("report", {}).get("csv", True):
            import pandas as pd
            df = pd.DataFrame(all_results).T
            df.to_csv(self.output_dir / "evaluation_results.csv")
        
        return all_results

