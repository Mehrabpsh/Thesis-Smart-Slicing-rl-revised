"""CLI entry point for SFC RL framework."""

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter  

import time
import shutil
import re 

from .data.dataset_provider import DatasetProvider
from .data.dataset_generator import Generator
from .env.sfc_env import SFCEnvRevised
from .env.state_encoders import NormalizedStateEncoder
from .env.action_space import NodeSelectionActionSpace
from .env.reward import QoE_QoS_Reward
from .env.qoe import QoE_QoS_PaperModel
from .models.dqn import DQNPolicy
from .models.networks import MLPPolicyNetwork
from .models.replay_buffer import ReplayBuffer
from .baselines.random_policy import RandomPolicy
from .baselines.exhaustive_solver import ExhaustiveSolver
from .train.trainer import TrainerRevised
from .train.evaluator import Evaluator
from .utils.seed import set_seed
from .utils.logging import setup_logger
from .utils.tensorboard import launch_tensorboard
#--------------------------------------------------------------------------------------------

config_path = "./../config"
config_name="experiment"


@hydra.main(version_base=None, config_path= config_path, config_name=config_name)
def main(cfg: DictConfig) -> None:
    
    """Main entry point.
    Args:
        cfg: Hydra configuration
    """

    # Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    # Create output directory
    output_dir = Path(f"{cfg.get('output_dir', 'outputs')}/{cfg.get('project_name', 'Ciriaa')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Setup logger
    logger = setup_logger("sfc_rl_test_data", log_file=output_dir / "run.log")
    logger.info(f"Starting experiment with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # Load dataset 
    # Merge data config into main config for Generator
    # When using Hydra defaults, cfg.data contains the merged config from data/synthetic_small.yaml
    with open_dict(cfg):
        if 'p_net_setting' not in cfg:
            # Check if p_net_setting is in cfg.data (from Hydra defaults)
            if hasattr(cfg, 'data') and cfg.data is not None:
                if hasattr(cfg.data, 'p_net_setting') and cfg.data.p_net_setting is not None:
                    cfg.p_net_setting = cfg.data.p_net_setting
                elif hasattr(cfg.data, 'get') and cfg.data.get('p_net_setting') is not None:
                    cfg.p_net_setting = cfg.data.get('p_net_setting')
        
        if 'v_sim_setting' not in cfg:
            # Check if v_sim_setting is in cfg.data (from Hydra defaults)
            if hasattr(cfg, 'data') and cfg.data is not None:
                if hasattr(cfg.data, 'v_sim_setting') and cfg.data.v_sim_setting is not None:
                    cfg.v_sim_setting = cfg.data.v_sim_setting
                elif hasattr(cfg.data, 'get') and cfg.data.get('v_sim_setting') is not None:
                    cfg.v_sim_setting = cfg.data.get('v_sim_setting')
    



    # Create dataset provider
    dataset_provider = DatasetProvider(cfg, logger)
    cache_path = Path(cfg.data.v_sim_setting.get('cache_path', '.vnReqs_cache'))

    if cache_path.exists() and cache_path.is_dir(): 
        shutil.rmtree(cache_path)
        logger.info(f"Cleaned and removed cache: {cache_path}")

    pn = dataset_provider.get_physical_network()
    vn_requests = dataset_provider.get_vn_requests()
    flag_pn = cfg.data.p_net_setting.get('dataset_dir',None) is not None
    flag_vn = cfg.data.v_sim_setting.get('dataset_dir',None) is not None
    logger.info(f"PN with {len(pn.nodes)} nodes and {len(pn.links)} links {'Loaded from path ' if flag_pn else 'Generated' }")
    logger.info(f" {len(vn_requests)} VN requests {'Loaded from path ' if flag_vn else 'Generated' }")

    if cfg.get('End_phase','All') == 'data' :
        if cache_path.exists() and cache_path.is_dir(): 
            shutil.rmtree(cache_path)
            logger.info(f"Cleaned and removed cache: {cache_path}")
        else:
            logger.info(f"Cache directory not found: {cache_path}")

        logger.info(f"Finishied")
        return
    
    #------------------------------- Env------------------------------------

    num_groups = cfg.data.v_sim_setting.get('num_groups')

    # Create environment components
    env_cfg = cfg.env
    
    # State encoder
    state_encoder_type = env_cfg.state.get("encoder", "NormalizedStateEncoder")
    if state_encoder_type == "NormalizedStateEncoder":
        state_encoder = NormalizedStateEncoder(env_cfg.state.get("encoder_config", {}))
    else:
        raise ValueError(f"Unknown encoder type: {state_encoder_type}")
    state_dim = state_encoder.get_state_dim(pn)
    
    # Action space
    action_space = NodeSelectionActionSpace(
        mask_illegal=env_cfg.action.get("mask_illegal", True)
    )
    action_dim = action_space.get_action_dim(pn)
    
    # QoE model
    qoe_cfg = env_cfg.qoe_model
    if qoe_cfg.name == "qoe_qos_paper":
        qoe_model = QoE_QoS_PaperModel(qoe_cfg.get("config", {}))
    else:
        raise ValueError(f"Unknown QoE model: {qoe_cfg.name}")
    
    # Reward function
    #reward_fn = TerminalQoEReward(qoe_model, env_cfg.reward)
    reward_fn = QoE_QoS_Reward(qoe_model, env_cfg.reward)
    
    # Create environment
    env = SFCEnvRevised(
        pn=pn,
        vn_requests=vn_requests,
        num_groups=num_groups,
        state_encoder=state_encoder,
        action_space=action_space,
        reward_fn=reward_fn,
        qoe_model=qoe_model,
        max_steps_per_request=cfg.train.get("max_steps_per_episode", 2000),
    )
    



    #------------------------------------- Policy --------------------------------

    
    # Create policy
    model_cfg = cfg.model
    if model_cfg.type == "dqn":

        # Create network
        network_cfg = model_cfg.network
        network = MLPPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=network_cfg.hidden_sizes,
            activation=network_cfg.activation,
            dueling=model_cfg.dqn.get("dueling", False),
        )
        
        # Create DQN policy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            network=network,
            config=model_cfg,
            device=device,
        )
        logger.info(f"Created DQN policy on {device}")
    elif model_cfg.type == "random":
        policy = RandomPolicy(cfg.get("seed", 42))
    elif model_cfg.type == 'Violent':
        policy = ExhaustiveSolver( max_embeddings=1000000, seed = cfg.get("seed", 42))
    else:
        raise ValueError(f"Unknown model type: {model_cfg.type}")
    


    #------------------------------------- Train -----------------------------------------
    
    tensorboard_use = False

    if tensorboard_use:

        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        tb_process = launch_tensorboard(output_dir / "tensorboard")
        logger.info("launching Tensorboard...")

        time.sleep(5)  


    # Train
    train_cfg = cfg.train
    if train_cfg.get('train', True): #or getattr(train_cfg, 'train', True):

        trainer = TrainerRevised(
            env=env,
            policy=policy,
            config=train_cfg,
            output_dir=output_dir,
            summarywriter=tb_writer if tensorboard_use else None,
            logger=logger,
            tensorboard_use=tensorboard_use,
        )
        
        logger.info("Starting training...")
        results = trainer.train()

        for i in results.keys():  
            if 'mean_episode_length' in results[i]:
                del results[i]['mean_episode_length']        
        
        print(f'{'.\n' * 20}')
        logger.info(f"Training completed. Final metrics:")
        for episode, metric in results.items():
            print(f'Episode {episode}: {metric} \n' )
        
    else: 
        logger.info("training Skipped...")

    #------------------------------------- Load Best Model and Evaluate --------------------------------------


    eval_cfg = cfg.eval

    if eval_cfg.get('enabled', True):

        logger.info(f"Loading most recent model ")

        checkpoint_files = list(output_dir.glob("checkpoint_ep*.pt"))
        checkpoints = sorted(
            checkpoint_files,
            key=lambda x: int(re.search(r"checkpoint_ep(\d+)\.pt", x.name).group(1))
            )   
            
        

        if checkpoints:
            trained_policy = DQNPolicy(
                state_dim=policy.state_dim,
                action_dim=policy.action_dim,
                network=policy.q_network,
                config=policy.config,
                device=policy.device
            )
            trained_policy.load(str(checkpoints[-1]))
            trained_policy.q_network.eval()
            trained_policy.target_network.eval()
            logger.info(f"Successfully loaded the most recent model")
        else:
            logger.warning(f"No checkpoints found. ")
            # Fallback to latest checkpoint or current policy
            checkpoint_files = list(output_dir.glob("checkpoint_ep*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_ep')[-1]))
                trained_policy = DQNPolicy(
                    state_dim=policy.state_dim,
                    action_dim=policy.action_dim,
                    network=policy.q_network,
                    config=policy.config,
                    device=policy.device
                )
                trained_policy.load(str(latest_checkpoint))
                trained_policy.q_network.eval()
                trained_policy.target_network.eval()
            else:
                trained_policy = policy
                trained_policy.q_network.eval()
                trained_policy.target_network.eval()
    

        #-------------------- Evaluate ----------------------

        logger.info("Starting Evaluating...")
        evaluator = Evaluator(env,
        policies={'random':RandomPolicy(model_cfg.get("seed", 42)),'dqn':trained_policy,'exhaustive':ExhaustiveSolver( max_embeddings=1000000, seed = model_cfg.get("seed", 42))},
        config=eval_cfg, output_dir= output_dir / Path(eval_cfg.get('output_dir','output_eval')),logger = logger)

        evaluator.evaluate()

        logger.info(f"Experiment completed. Results saved to {output_dir / Path(eval_cfg.get('output_dir','output_eval')) }")
        if cache_path.exists() and cache_path.is_dir(): 
            shutil.rmtree(cache_path)
            logger.info(f"Cleaned and removed cache: {cache_path}")
        else:
            logger.info(f"Cache directory not found: {cache_path}")

    else :
        logger.info(" Evaluating skipped...")
        if cache_path.exists() and cache_path.is_dir(): 
            shutil.rmtree(cache_path)
            logger.info(f"Cleaned and removed cache: {cache_path}")
        else:
            logger.info(f"Cache directory not found: {cache_path}")

    if tensorboard_use:
        if tb_process:
            tb_process.terminate()
            print("TensorBoard stopped.")


main()



#python -m sfc_rl.cli data.v_sim_setting.num_groups=15 model.dqn.eps_decay_steps=1500 project_name=after_hashemi3


#Tasks to do:
#1- verify DQN , go watch some videos tutorials and ... of dqn implementation 1h
#2- use it in jupyter lab/ notebook (refactor its form if needed) 1-2h
#3- run for 500 episodes or sth 3-4 h
#4 - report to Hashemi 1h