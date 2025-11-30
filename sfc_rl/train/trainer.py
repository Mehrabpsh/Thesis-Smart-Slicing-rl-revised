# """Training loop for RL agents."""

# from typing import Optional, Dict, Any
# from pathlib import Path
# import numpy as np
# #from sympy import E
# #import torch
# from omegaconf import DictConfig
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# from ..env.sfc_env import SFCEnvRevised
# from ..models.dqn import DQNPolicy
# from ..baselines.random_policy import RandomPolicy
# from ..baselines.exhaustive_solver import ExhaustiveSolver
# from .metrics import Metrics
# from .plots import plot_training_curves
# from ..utils.logging import setup_logger
# from ..utils.seed import set_seed



# class TrainerRevised:
#     """Generic trainer for RL policies."""
    
#     def __init__(
#         self,
#         env: SFCEnvRevised,
#         policy: Any,  # DQNPolicy, RandomPolicy, or ExhaustiveSolver
#         config: DictConfig,
#         output_dir: Path,
#         logger=None,
#     ):
#         """Initialize trainer.
        
#         Args:
#             env: SFC environment
#             policy: Policy to train (DQNPolicy) or evaluate (RandomPolicy, ExhaustiveSolver)
#             config: Training configuration
#             output_dir: Output directory
#             logger: Optional logger
#         """
#         self.env = env
#         self.policy = policy
#         self.config = config
#         self.output_dir = output_dir
#         self.logger = logger or setup_logger("trainer")
        
#         self.episodes = config.episodes
#         self.max_steps_per_episode = config.max_steps_per_episode
#         self.log_interval = config.get("log_interval", 100)
#         self.eval_every = config.get("eval_every", 1)
#         self.save_every = config.get("save_every", 10)

#         # Initialize TensorBoard
#         self.tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
#         self.logger.info(f"TensorBoard logs directory: {output_dir / 'tensorboard'}")
    
    
#     def train(self) -> Dict[int, Dict]:
#         """Run training loop.
        
#         Returns:
#             Dictionary of training results
#         """
#         metrics = Metrics()
#         episode_rewards = []
#         episode_lengths = []
#         losses = []
#         Episodes_Metrics = {}

#         #obs, info = self.env.reset(seed=self.config.get("seed"))
#         #action_mask = self.env.action_mask()
#         # Each Episode means a run  which includes embedding a bunch of  requests in a run 
#         # episodes = num_groups

        
#         for episode in tqdm(range(self.episodes), desc=f"Training {self.policy.name}"):
            
#             obs, info = self.env.reset(group_id=episode, seed=self.config.get("seed"))
#             episode_reward = 0.0 # Sum_r
#             episode_length = 0
#             episode_start_time = None
            
#             # Here episode means a run 
#             # expected functionality -> in a run , embed all the request -> break if all the requests are embeded 
#             while True:
             
               
#                 if isinstance(self.policy, RandomPolicy):
#                     action = self.policy.act(self.env)
#                 elif isinstance(self.policy, ExhaustiveSolver):
#                     action = self.policy.solve(self.env)
#                 elif isinstance(self.policy, DQNPolicy):
#                     action = self.policy.act(obs,training = True)
#                 else:
#                     raise ValueError(f"Unknown policy type: {type(self.policy)}")
                    
#                 # Track episode start
#                 if episode_start_time is None:
#                     episode_start_time = self.env.current_vn_requests[self.env.current_request_idx].arrival_time if self.env.current_request_idx < len(self.env.current_vn_requests) else 0.0
                
#                 # Step environment
#                 next_obs, reward, terminated, truncated, info = self.env.step(action)
#                 #next_action_mask = self.env.action_mask()
                
#                 episode_reward += reward
#                 #episode_length += 1

                
#                 # Store experience (for DQN)
#                 if isinstance(self.policy, DQNPolicy):
#                     self.policy.replay_buffer.push(
#                         obs, action, reward, next_obs, terminated) #or truncated)
                      
#                     # # Learn if enough experiences is in replay buffer
#                     # if len(self.policy.replay_buffer) >= self.policy.replay_buffer.capacity:
#                     #     loss = self.policy.learn()
#                     #     if loss is not None:
#                     #         losses.append(loss)
                

#                 if not (reward == 0):

#                     # Learn if enough experiences is in replay buffer
#                     if len(self.policy.replay_buffer) >= self.policy.replay_buffer.capacity:
#                         loss = self.policy.learn()
#                         if loss is not None:
#                             losses.append(loss)
#                             self.tb_writer.add_scalar('Training/Loss', loss, len(losses))
                
#                     episode_length += 1
#                     if self.env.current_request_idx > 0:
#                         prev_idx = self.env.current_request_idx - 1
#                         if prev_idx < len(self.env.current_vn_requests):
#                             request = self.env.current_vn_requests[prev_idx]
#                             accepted = reward > 0  # Simple heuristic
#                             qoe = reward if accepted else None  # Could compute from embedding
#                             if isinstance(self.policy, ExhaustiveSolver):
#                                     response_time = self.policy.solutionTime
#                             elif isinstance(self.policy, RandomPolicy):
#                                     response_time = self.policy.solutionTime * self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length
#                             elif isinstance(self.policy, DQNPolicy ):
#                                     response_time =  1* self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length

#                             metrics.update(request, accepted, response_time, qoe, reward)
#                             #metrics.update(request, accepted, qoe, reward)
#                             print(f'request id: {self.env.current_request_idx-1} of group {self.env.group_id} {'embedded' if accepted == True else 'failed'} with reward {reward}')


                
#                 if terminated: # if all the request of the current group are hanlded
#                     #avg_sum = float(np.mean(episode_reward))
#                     #episode_rewards.append(avg_sum)
#                     episode_rewards.append(episode_reward)
#                     episode_lengths.append(episode_length)
#                     Episodes_Metrics[episode] = metrics.compute()
#                     metrics.reset()
#                     break

#                 obs = next_obs
#                 #action_mask = next_action_mask
                
          
#                 # Check max steps -> max number of steps per request embedding . Why conside it ?!
#                 #if episode_length >= self.max_steps_per_episode:
#                 #    break
       
            
#             # Logging
#             if (episode + 1) % self.log_interval == 0:
#                 avg_reward = np.mean(episode_rewards[-self.log_interval:])
#                 #avg_length = np.mean(episode_lengths[-self.log_interval:])
#                 avg_loss = np.mean(losses[-self.log_interval:]) if losses else 0.0
#                 self.logger.info(
#                     f"Episode {episode + 1}/{self.episodes} - "
#                     f"Avg Reward: {avg_reward:.4f}, "
#                     #f"Avg Length: {avg_length:.2f}, "
#                     f"Avg Loss: {avg_loss:.4f}"
#                 )
            
#             # Save checkpoint
#             if isinstance(self.policy, DQNPolicy) and (episode + 1) % self.save_every == 0:
#                 checkpoint_path = self.output_dir / f"checkpoint_ep{episode + 1}.pt"
#                 self.policy.save(str(checkpoint_path))
            
#         plot_training_curves(rewards=[Episodes_Metrics[i]['qoe'] for i in Episodes_Metrics.keys()], losses= losses,output_dir = self.output_dir / "training plots", filename="training_curve.png", window=5)
        
#         return Episodes_Metrics






















"""Training loop for RL agents."""

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # Add this import


from ..env.sfc_env import SFCEnvRevised
from ..models.dqn import DQNPolicy
from ..baselines.random_policy import RandomPolicy
from ..baselines.exhaustive_solver import ExhaustiveSolver
from .metrics import Metrics
from .plots import plot_training_curves
from ..utils.logging import setup_logger
from ..utils.seed import set_seed




class TrainerRevised:
    """Generic trainer for RL policies."""
    
    def __init__(
        self,
        env: SFCEnvRevised,
        policy: Any,  # DQNPolicy, RandomPolicy, or ExhaustiveSolver
        config: DictConfig,
        output_dir: Path,
        summarywriter:SummaryWriter,
        logger=None,
        tensorboard_use:bool = False,
    ):
        """Initialize trainer.
        
        Args:
            env: SFC environment
            policy: Policy to train (DQNPolicy) or evaluate (RandomPolicy, ExhaustiveSolver)
            config: Training configuration
            output_dir: Output directory
            logger: Optional logger
        """
        self.env = env
        self.policy = policy
        self.config = config
        self.output_dir = output_dir
        self.logger = logger or setup_logger("trainer")
        self.run_in_notebook = config.run_in_notebook

        self.episodes = config.episodes
        self.max_steps_per_episode = config.max_steps_per_episode
        self.log_interval = config.get("log_interval", 100)
        self.eval_every = config.get("eval_every", 1)
        self.save_every = config.get("save_every", 10)
        self.tensorboard_use = tensorboard_use
        
        if self.tensorboard_use:
        # Initialize TensorBoard
            self.tb_writer = summarywriter
            self.logger.info(f"TensorBoard logs directory: {output_dir / 'tensorboard'}")
        
    def train(self) -> Dict[int, Dict]:
        """Run training loop.
        
        Returns:
            Dictionary of training results
        """
        metrics = Metrics()
        episode_rewards = []
        episode_lengths = []
        losses = []
        Episodes_Metrics = {}

        for episode in tqdm(range(self.episodes), desc=f"Training {self.policy.name}"):
            
            obs, info = self.env.reset(group_id=episode, seed=self.config.get("seed"))
            episode_reward = 0.0
            episode_length = 0
            episode_start_time = None
            
            
            while True:
                if isinstance(self.policy, RandomPolicy):
                    action = self.policy.act(self.env)
                elif isinstance(self.policy, ExhaustiveSolver):
                    action = self.policy.solve(self.env)
                elif isinstance(self.policy, DQNPolicy):
                    action = self.policy.act(obs, training=True)
                else:
                    raise ValueError(f"Unknown policy type: {type(self.policy)}")
                    
                # Track episode start
                if episode_start_time is None:
                    episode_start_time = self.env.current_vn_requests[self.env.current_request_idx].arrival_time if self.env.current_request_idx < len(self.env.current_vn_requests) else 0.0
                
                # Step environment
                next_obs, reward, terminated, _ , info = self.env.step(action)
                
                episode_reward += reward

                # Store experience (for DQN)
                if isinstance(self.policy, DQNPolicy):
                    self.policy.replay_buffer.push(
                        obs, action, reward, next_obs, terminated)
            
                #learn each step instead of each embedding
                if len(self.policy.replay_buffer) >= self.policy.batch_size: #>= self.policy.replay_buffer.capacity:
                    loss = self.policy.learn()
                    if loss is not None:
                        losses.append(loss)
                        # Log loss to TensorBoard
                        if self.tensorboard_use:
                            self.tb_writer.add_scalar('Training/Loss', loss, len(losses))
                            self.tb_writer.flush()
                

                if info['embedding_state'] == 'success' or info['embedding_state'] == 'fail': #not (reward == 0):
                    # Learn if enough experiences is in replay buffer
                    # if len(self.policy.replay_buffer) >= self.policy.replay_buffer.capacity:
                    #     loss = self.policy.learn()
                    #     if loss is not None:
                    #         losses.append(loss)
                    #         # Log loss to TensorBoard
                    #         self.tb_writer.add_scalar('Training/Loss', loss, len(losses))
                    #         self.tb_writer.flush()
                
                    episode_length += 1
                    if self.env.current_request_idx > 0:
                        prev_idx = self.env.current_request_idx - 1
                        if prev_idx < len(self.env.current_vn_requests):
                            request = self.env.current_vn_requests[prev_idx]
                            accepted = True if info['embedding_state'] == 'success' else False
                            qoe = reward if accepted else None
                            if isinstance(self.policy, ExhaustiveSolver):
                                response_time = self.policy.solutionTime
                            elif isinstance(self.policy, RandomPolicy):
                                response_time = self.policy.solutionTime * self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length
                            elif isinstance(self.policy, DQNPolicy):
                                response_time = 1 * self.env.current_vn_requests[self.env.current_request_idx-1].sfc_length

                            metrics.update(request, accepted, response_time, qoe, reward)
                            
            
                            
                            print(f'request id: {self.env.current_request_idx-1} of group {self.env.group_id} {"embedded" if accepted else "failed"} with reward {reward}')

                if terminated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    Episodes_Metrics[episode] = metrics.compute()
                    
                    if self.tensorboard_use:
                        self.tb_writer.add_scalar('Episode/Reward', episode_reward, episode)
                        self.tb_writer.add_scalar('Metrics/Acceptance_Ratio',  Episodes_Metrics[episode]['acceptance_ratio'], episode)
                        self.tb_writer.flush()
                    
                    metrics.reset()
                    break

                obs = next_obs
            
            # Periodic logging
            if (episode + 1) % self.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-self.log_interval:])
                avg_loss = np.mean(losses[-self.log_interval:]) if losses else 0.0
               
                self.logger.info(
                    f"Episode {episode + 1}/{self.episodes} - "
                    f"Avg Reward: {avg_reward:.4f}, "
                    f"Avg Loss: {avg_loss:.4f}"
                )
                plot_training_curves(rewards=[Episodes_Metrics[i]['qoe'] for i in Episodes_Metrics.keys()], losses= losses,output_dir = self.output_dir / "training plots", filename=f"training_curve_till_episode{episode}.png", window=5,episode=episode,run_in_notebook=self.run_in_notebook)

            
            # Save checkpoint
            if isinstance(self.policy, DQNPolicy) and (episode + 1) % self.save_every == 0:
                checkpoint_path = self.output_dir / f"checkpoint_ep{episode + 1}.pt"
                self.policy.save(str(checkpoint_path))
        

        # Final plots and close TensorBoard
        #plot_training_curves(rewards=[Episodes_Metrics[i]['qoe'] for i in Episodes_Metrics.keys()], losses= losses,output_dir = self.output_dir / "training plots", filename="training_curve.png", window=5,episode=episode)
        if self.tensorboard_use:
            self.tb_writer.close()
        
        return Episodes_Metrics