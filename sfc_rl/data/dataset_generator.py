"""Dataset generator following Virne-style pattern."""

import os
import copy
from typing import Dict, Optional, Union, Tuple
from omegaconf import DictConfig, OmegaConf

from .schemas import PhysicalNetwork
from .generators import generate_erdos_renyi_pn, generate_poisson_vn_stream
from .loaders import load_pn_from_json, load_vn_requests_from_json, load_pn_from_graphml
from .vn_request_simulator import VNRequestSimulator
from ..utils.seed import set_seed


class Generator:
    """Generator class for creating datasets (Virne-style)."""
    
    @staticmethod
    def generate_dataset(
        config: Union[DictConfig, Dict],
        p_net: bool = True,
        v_nets: bool = True,
        save: bool = False,
    ) -> Tuple[Optional[PhysicalNetwork], Optional[VNRequestSimulator]]:
        """Generate a dataset consisting of a physical network and a virtual network request simulator.
        
        Args:
            config: Configuration object containing the settings
            p_net: Whether to generate a physical network dataset
            v_nets: Whether to generate a virtual network request simulator dataset
            save: Whether to save the generated datasets
            
        Returns:
            Tuple of (physical_network, v_net_simulator)
        """
        physical_network = Generator.generate_p_net_dataset_from_config(config, save=save) if p_net else None
        v_net_simulator = Generator.generate_v_nets_dataset_from_config(config, save=save) if v_nets else None
        return physical_network, v_net_simulator
    
    @staticmethod
    def generate_p_net_dataset_from_config(
        config: Union[DictConfig, Dict],
        save: bool = False,
    ) -> PhysicalNetwork:
        """Generate a physical network dataset based on the given configuration.
        
        Args:
            config: Configuration object containing p_net_setting
            save: Whether to save the generated dataset
            
        Returns:
            PhysicalNetwork instance
        """
        from omegaconf import OmegaConf
        
        # Convert DictConfig to dict if needed
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        
        assert isinstance(config_dict, dict), "config must be a DictConfig or dict"
        assert 'p_net_setting' in config_dict, "config must contain 'p_net_setting' key"
        
        seed = config_dict.get('seed', None)
        set_seed(seed)
        
        p_net_setting = config_dict.get('p_net_setting', {})
        # Convert to plain dict if it's a DictConfig
        if isinstance(p_net_setting, DictConfig):
            p_net_setting = OmegaConf.to_container(p_net_setting, resolve=True)
        
        p_net = PhysicalNetwork.from_setting(p_net_setting, seed=seed)
        
        if save:
            p_net_dataset_dir = Generator._get_p_net_dataset_dir_from_setting(p_net_setting, seed)
            p_net.save_dataset(p_net_dataset_dir)
        
        return p_net
    
    @staticmethod
    def generate_v_nets_dataset_from_config(
        config: Union[DictConfig, Dict],
        save: bool = False,
    ) -> VNRequestSimulator:
        """Generate a virtual network request simulator dataset based on the given configuration.
        
        Args:
            config: Configuration object containing v_sim_setting
            save: Whether to save the generated dataset
            
        Returns:
            VNRequestSimulator instance
        """
        from omegaconf import OmegaConf
        
        # Convert DictConfig to dict if needed
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
        
        assert isinstance(config_dict, dict), "config must be a DictConfig or dict"
        assert 'v_sim_setting' in config_dict, "config must contain 'v_sim_setting' key"
        
        seed = config_dict.get('seed', None)
        set_seed(seed)
        
        v_sim_setting = config_dict.get('v_sim_setting', {})
        # Convert to plain dict if it's a DictConfig
        if isinstance(v_sim_setting, DictConfig):
            v_sim_setting = OmegaConf.to_container(v_sim_setting, resolve=True)
        
        v_net_simulator = VNRequestSimulator.from_setting(v_sim_setting, seed=seed)
        v_net_simulator.renew()
        
        if save:
            v_nets_dataset_dir = Generator._get_v_nets_dataset_dir_from_setting(v_sim_setting, seed)
            v_net_simulator.save_dataset(v_nets_dataset_dir)
        
        return v_net_simulator
    
    @staticmethod
    def _get_p_net_dataset_dir_from_setting(p_net_setting: Dict, seed: Optional[int] = None) -> str:
        """Get the directory path for physical network dataset from setting.
        
        Args:
            p_net_setting: Physical network setting dictionary
            seed: Optional random seed
            
        Returns:
            Dataset directory path
        """
        output_dir = p_net_setting.get('output', {}).get('save_dir', 'datasets/p_net')
        
        # Build directory name from topology settings
        topology = p_net_setting.get('topology', {})

        topo_type = topology.get('type', 'erdos_renyi')
        num_nodes = topology.get('num_nodes', 0)
        p_net_name = f"{num_nodes}nodes-{topo_type}"
    
        # Add node and link attributes to name
        node_attrs = p_net_setting.get('node_attrs_setting', [])
        link_attrs = p_net_setting.get('link_attrs_setting', [])
        node_attrs_str = '-'.join([attr.get('name', 'cpu') for attr in node_attrs])
        link_attrs_str = '-'.join([attr.get('name', 'bandwidth') for attr in link_attrs])
        
        dataset_middir = f"{p_net_name}-{node_attrs_str}-{link_attrs_str}"
        if seed is not None:
            dataset_middir += f'-seed_{seed}'
        
        dataset_dir = os.path.join(output_dir, dataset_middir)
        return dataset_dir
    
    @staticmethod
    def _get_v_nets_dataset_dir_from_setting(v_sim_setting: Dict, seed: Optional[int] = None) -> str:
        """Get the directory path for virtual network dataset from setting.
        
        Args:
            v_sim_setting: Virtual network simulator setting dictionary
            seed: Optional random seed
            
        Returns:
            Dataset directory path
        """
        output_dir = v_sim_setting.get('output', {}).get('save_dir', 'datasets/v_nets')
        
        num_v_nets = v_sim_setting.get('num_v_nets', 0)
        groups = v_sim_setting.get('num_groups', 0)
        sfc_len = v_sim_setting.get('sfc_len', {})
        sfc_len_str = f"[{sfc_len.get('low', 3)}-{sfc_len.get('high', 5)}]"
        
        node_attrs = v_sim_setting.get('node_attrs_setting', [])
        qos_attrs = v_sim_setting.get('qos_attrs_setting', [])
        node_attrs_str = '-'.join([attr.get('name', 'cpu') for attr in node_attrs])
        link_attrs_str = '-'.join([attr.get('name', 'bandwidth') for attr in qos_attrs])
        
        arrival_rate = v_sim_setting.get('arrival_rate', {}).get('lam', 1.0)
        
        dataset_middir = f"{num_v_nets}sfcReqs-{groups}groups-{sfc_len_str}lengths-arrival_rate{arrival_rate}-{node_attrs_str}-{link_attrs_str}"
        if seed is not None:
            dataset_middir += f'-seed_{seed}'
        
        dataset_dir = os.path.join(output_dir, dataset_middir)
        return dataset_dir

