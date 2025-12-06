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
        all_results_raw = {}
        acceptance_ratios  = {}
        qoes = {}
        rewards = {}
        
        for policy_name, policy in self.policies.items():
            self.logger.info(f"Evaluating policy: {policy_name}")
            Episodes_Metrics = {}

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
                        action = policy.act(obs, training= False)
                    elif isinstance(policy, RandomPolicy):
                        action = policy.act(self.env)
                    elif isinstance(policy, ExhaustiveSolver):
                        action = policy.solve(self.env)
                    else:
                        raise ValueError(f"Unknown policy type: {type(policy)}")
                    
                    # Step environment
                    next_obs, reward, terminated, _, info = self.env.step(action)
                    
                    episode_reward += reward

                    if info['embedding_state'] == 'success' or info['embedding_state'] == 'fail': #not (reward == 0):
                        if self.env.current_request_idx > 0:
                            prev_idx = self.env.current_request_idx - 1
                            if prev_idx < len(self.env.current_vn_requests):
                                request = self.env.current_vn_requests[prev_idx]
                                accepted = True if info['embedding_state'] == 'success' else False
                                if isinstance(policy, ExhaustiveSolver):
                                    response_time = policy.solutionTime
                                elif isinstance(policy, RandomPolicy):
                                    response_time = policy.solutionTime * self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length * 1e2
                                elif isinstance(policy, DQNPolicy ):
                                    response_time =  1* self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length

                                metrics.update(request, accepted, response_time, reward)
                                print(f'request id: {self.env.current_request_idx-1} of group {self.env.group_id} {'embedded' if accepted == True else 'failed'} with reward {reward}')


                    
                    if terminated: # if all the request of the current group are hanlded
                        Episode_Metrics[run] = metrics.request_metrics
                        if episode not in acceptance_ratios:
                           acceptance_ratios[episdoe] = {}    
                        acceptance_ratios[episdoe][policy_name] = metrics.accepted_requests / metrics.total_requests}
                        #Episodes_Metrics[run] = metrics.compute()
                        metrics.reset()
                        break


                    obs = next_obs
      
            
            # # Average across runs
            # avg_metrics = {}
            # for metric in self.metrics_to_compute:
            #     values = [r.get(metric, 0.0) for r in Episodes_Metrics.values()]
            #     avg_metrics[metric] = np.mean(values)
            #     avg_metrics[f"{metric}_std"] = np.std(values)
            
            # all_results[policy_name] = avg_metrics

            all_results_raw[policy_name] =  Episode_Metrics

        for episode , (exh_dicts, rand_dicts) in enumerate(zip(all_results_raw['exhaust'].values(), all_results_raw['random'].values())):
            num = 0 
            qoe_random = 0
            qoe_exh = 0
            reward_random = 0
            reward_exh = 0
            for exh_dict, rand_dict in zip((exh_dicts, rand_dicts):
                reward_random =+ random_dict['reward']
                reward_exh =+ random_dict['reward']
                if exh_dict['accepted']:
                    qoe_random =+ random_dict['reward']
                    qoe_exh =+ random_dict['reward']
                    num +=1
            if num > 0:        
                qoes[episode] = {'random':qoe_random/num,'exhaustive':qoe_exh/num}
            rewards[episode] =  {'random':qoe_random/len(exh_dicts),'exhaustive':qoe_exh/len(exh_dicts)}
            
             #if random_dict['accepted']:
              #  qoe_random =+ random_dict['reward']
               # qoe_exh =+ random_dict['reward']


        #-------------------- plot qoe per episode ------------------

        plots_dir.mkdir(parents=True, exist_ok=True)
  
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    
        x_axis = np.linspace(0,len(qoes.keys()),len(qoes.keys()))

        axes[0].plot(x_axis, [qoes[i]['random'] for i in qoes.keys()], alpha=0.3, label="Raw", color="read")
        axes[0].plot(x_axis, [qoes[i]['exhaust'] for i in qoes.keys()], alpha=0.3, label="Raw", color="green")

        
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("qoes")
        axes[1].set_title("Qoes Progress")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "qoes_comparision_in_episdoes.png", dpi=150, bbox_inches='tight')
        
        plt.show() 
        plt.close()

        
        
        metrics_dict = {}

        
        #----------------------

        # # Save results
        # self.output_dir.mkdir(parents=True, exist_ok=True)
        # save_json(all_results, self.output_dir / "evaluation_results.json")
        
        # # Generate plots
        # if self.config.get("report", {}).get("plots", True):
        #     plots_dir = self.output_dir / "plots"
        #     plots_dir.mkdir(parents=True, exist_ok=True)
        #     plot_metrics(all_results, plots_dir, "metrics_comparison.png")
        
        #-------------------- Bar Plot -----------------------------
        # fig, axes = plt.subplots(1, len(self.metrics_to_compute), figsize=(5 * len(metrics), 5))
        # if len(metrics) == 1:
        #     axes = [axes]
        
        # for idx, metric in enumerate(metrics):
        #     values = [metrics_dict[policy].get(metric, 0.0) for policy in policies]
        #     axes[idx].bar(policies, values)
        #     axes[idx].set_title(metric.replace("_", " ").title())
        #     axes[idx].set_ylabel(metric.replace("_", " ").title())
        #     axes[idx].tick_params(axis='x', rotation=45)
        
        # plt.tight_layout()
        # plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        # plt.close()
        fig, axes = plt.subplots(1, 2, figsize=(5 * len(metrics), 5))
    
        policies = ['random', 'exhaust']
        
        values1 = [np.mean([ acceptance_ratios[i]['random'] for i in acceptance_ratios.keys()]),  np.mean([ acceptance_ratios[i]['exhaust'] for i in acceptance_ratios.keys()])]
        axes[0].bar(policies, values1)
        axes[0].set_title('episodes')
        axes[0].set_ylabel'qoe'
        axes[0].tick_params(axis='x', rotation=45)



        values2 = [np.mean([ rewards[i]['random'] for i in rewards.keys()]),  np.mean([ rewards[i]['exhaust'] for i in rewards.keys()])]
        axes[0].bar(policies, values2)
        axes[0].set_title('episodes')
        axes[0].set_ylabel'qoe'
        axes[0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


        # # Save CSV
        # if self.config.get("report", {}).get("csv", True):
        #     import pandas as pd
        #     df = pd.DataFrame(all_results).T
        #     df.to_csv(self.output_dir / "evaluation_results.csv")
        
        #return all_results

