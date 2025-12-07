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
        rewards = {}
        qoes = {}
        instant_rewards = {}
        #for debbuging purposes
        succeeded_embeddings_random = {}
        possible_embeddings_exhaustive = {}

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
                        if self.env.current_request_idx >= 0:
                            prev_idx = self.env.current_request_idx - 1
                            if prev_idx < len(self.env.current_vn_requests):
                                request = self.env.current_vn_requests[prev_idx]
                                accepted = True if info['embedding_state'] == 'success' else False
                                if isinstance(policy, ExhaustiveSolver):
                                    response_time = policy.solutionTime
                                    #print(f'policy.objective_values of exhasutive policy: {policy.objective_values}')
                                    #if accepted: # for debugging
                                    possible_embeddings_exhaustive[(self.env.group_id,self.env.current_request_idx-1)] = [choice[0] for choice in policy.complete_solutions]
                                elif isinstance(policy, RandomPolicy):
                                    if accepted: # for debugging
                                        succeeded_embeddings_random[(self.env.group_id,self.env.current_request_idx-1)] = info['partial_embedding']
                                    response_time = policy.solutionTime * self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length * 1e2
                                elif isinstance(policy, DQNPolicy ):
                                    response_time =  1* self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length

                                metrics.update(request, accepted, response_time, reward)
                                #available_cpu = {f'node {idx}':self.env.pn.nodes[idx].available_cpu for idx in self.env.pn.nodes.keys()}
                                #available_band = {f'link {idx}':pnlinks.available_bandwidth for idx, pnlinks in enumerate(self.env.pn.links)}

                                print(f'request id: {self.env.current_request_idx-1} of group {self.env.group_id} {'embedded' if accepted == True else 'failed'} with reward {reward}') #.\n available_cpus: {available_cpu} and available_bands{available_band} ')
                                if policy_name not in instant_rewards:
                                    instant_rewards[policy_name] = [] 
                                instant_rewards[policy_name].append(reward)
                             
                    
                    if terminated: # if all the request of the current group are hanlded
                        Episodes_Metrics[run] = metrics.request_metrics # to calculate qoe, i need it 
                        
                        if policy_name not in acceptance_ratios:
                           acceptance_ratios[policy_name] = {}    
                        acceptance_ratios[policy_name][run] = metrics.accepted_requests / metrics.total_requests

             
                        if policy_name not in rewards:
                           rewards[policy_name] = {}    
                        rewards[policy_name][run] = np.mean(metrics.episode_rewards)
                        
                        metrics.reset()
                        break


                    obs = next_obs

            all_results_raw[policy_name] =  Episodes_Metrics

        

        for  (grp,idx), solution in succeeded_embeddings_random.items():
            
            if solution in possible_embeddings_exhaustive[(grp,idx)]:
                if list(solution.keys())[-1] == (grp,idx):
                    self.logger.info('All matched')
                continue
            self.logger.critical(f'(grp,idx): {(grp,idx)} mismached ')

        # For each episode, iterate over all policies' results in parallel
        for episode, policy_results in enumerate(zip(*[list(all_results_raw[policy_name].values()) for policy_name in self.policies.keys()])):
            # policy_results is a tuple where each element corresponds to one policy's result dict for this episode
            policy_names = list(self.policies.keys())
            total_rewards = {name: 0.0 for name in policy_names}
            #accepted_counts = {name: 0 for name in policy_names}
            accepted_counts = 0
            
            policy_results = list(policy_results)
            refrence_index = policy_names.index('random')
            reference_list = policy_results.pop(refrence_index)
            policy_names.pop(refrence_index)

            for id,req in enumerate(reference_list):
                if req['accepted']:
                    total_rewards['random'] += req['reward']
                    #accepted_counts[policy_name] += 1
                    accepted_counts += 1
                        # Iterate over each policy's result dict for this episode
                    for policy_idx, result_dict in enumerate(policy_results):
                        policy_name = policy_names[policy_idx]
                        total_rewards[policy_name] += result_dict[id]['reward']
                        #accepted_counts[policy_name] += 1
        

            policy_names.append('random')
            policy_results.append(reference_list)

            # Compute averages only for policies with at least one accepted request
            for name in policy_names:
                if name not in qoes:
                    qoes[name] = {}
                if accepted_counts > 0:
                    qoes[name][episode] = total_rewards[name] / accepted_counts #accepted_counts[name]
                else:
                    qoes[name][episode]= -10 # or None, depending on how you want to handle no accepted cases
                    
    
        #print('reward:',rewards)
        print('qoe:',qoes)
        #print('accp:',acceptance_ratios)


        #all_results[policy]= {'acceptance_ratio': acceptance_ratios[policy],'qoe':qoe[policy],'reward':rewards[policy]} for policy in self.policies.keys()
        
        #-------------------- plot qoe per episode ------------------

        import matplotlib.pyplot as plt 
        
        plots_dir = self.output_dir   
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        policy_names = list(self.policies.keys())
        
        # ========== 1. QoE over episodes (time series) ==========
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        num_episodes = len(qoes[list(self.policies.keys())[0]])  

        x_axis = range(1,num_episodes+1)  
        for policy in policy_names:
            y_vals = list(qoes[policy].values())
            ax[0].plot(x_axis, y_vals, label=policy, alpha=0.7, linewidth=1.5)
        
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Average QoE (per accepted request)")
        ax[0].set_title("QoE Progress Over Episodes")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        

        for policy in policy_names:
            y_vals = instant_rewards[policy]
            ax[1].plot( y_vals, label=policy, alpha=0.7, linewidth=1.5)
        
        ax[1].set_xlabel("requests")
        ax[1].set_ylabel("Rewards ")
        ax[1].set_title("Rewards Over Episodes")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)



        instant_rewards

        plt.tight_layout()
        plt.savefig(plots_dir / "qoes_comparison_in_episodes.png", dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


        
        
   
        # ========== 2. Bar plots: mean acceptance ratio & mean reward ==========
        
        
        
        mean_acceptance = {
            policy: np.mean([acceptance_ratios[policy][ep] for ep in acceptance_ratios[policy].keys()])
            for policy in policy_names
        }
        mean_reward = {
            policy: np.mean([rewards[policy][ep] for ep in rewards[policy].keys()])
            for policy in policy_names
        }
        
        # Create bar plots
        fig, axes = plt.subplots(1, 2, figsize=(5 * 2, 5))  # 2 plots side by side
        
        # Acceptance ratio
        axes[0].bar(policy_names, [mean_acceptance[p] for p in policy_names])
        axes[0].set_title('Mean Acceptance Ratio')
        axes[0].set_ylabel('Acceptance Ratio')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Mean reward
        axes[1].bar(policy_names, [mean_reward[p] for p in policy_names])
        axes[1].set_title('Mean Reward')
        axes[1].set_ylabel('Reward')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


        # # Save CSV
        # if self.config.get("report", {}).get("csv", True):
        #     import pandas as pd
        #     df = pd.DataFrame({'model':[policy in policy_names],'acceptance_ratio':[mean_acceptance[policy] for policy in policy_names],'reward':[mean_reward[policy] for policy in  policy_names]}).T
        #     df.to_csv(self.output_dir / "evaluation_results.csv")
        
        #return all_results

        if self.config.get("report", {}).get("csv", True):
            import pandas as pd
            
            # Prepare data for DataFrame
            data = []
            for policy in policy_names:
                data.append({
                    'model': policy,
                    'acceptance_ratio': mean_acceptance.get(policy, 0.0),
                    'reward': mean_reward.get(policy, 0.0),
                    'qoe': qoes[policy]
                })
            
            df = pd.DataFrame(data)
            df.to_csv(self.output_dir / "evaluation_results.csv", index=False)
            self.logger.info(f"CSV report saved to {self.output_dir / 'evaluation_results.csv'}")
