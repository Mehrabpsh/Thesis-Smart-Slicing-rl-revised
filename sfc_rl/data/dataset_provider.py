"""Dataset provider interface and implementations (Virne-style)."""

import sys

from typing import Optional, List
from omegaconf import DictConfig

from ..utils.logging import setup_logger

from .schemas import PhysicalNetwork, VNRequest
from .vn_request_simulator import VNRequestSimulator
from .dataset_generator import Generator


# For backward compatibility, keep DatasetProvider but use Generator internally
class DatasetProvider:
    """Dataset provider (Virne-style wrapper)."""
    
    def __init__(self, config: DictConfig, logger: setup_logger):
        """Initialize dataset provider.
        
        Args:
            config: Configuration with p_net_setting and v_sim_setting
        """
        self.config = config
        self.logger = logger
        self._pn: Optional[PhysicalNetwork] = None
        self._v_net_simulator: Optional[VNRequestSimulator] = None
    
    def get_physical_network(self) -> PhysicalNetwork:
        """Get the Physical Network.
        
        Returns:
            PhysicalNetwork instance
        """
        if self._pn is None:
            # Try to load from dataset directory first
            # Handle both DictConfig and dict
            if hasattr(self.config, 'get'):
                p_net_setting = self.config.get('p_net_setting', {})
            else:
                p_net_setting = getattr(self.config, 'p_net_setting', {})
            
            # Convert to dict if needed
            if hasattr(p_net_setting, '_content'):
                p_net_setting = dict(p_net_setting)
            
            p_net_dataset_dir = p_net_setting.get('dataset_dir') if isinstance(p_net_setting, dict) else getattr(p_net_setting, 'dataset_dir', None)
            if_save = p_net_setting.get('output', {}).get('if_save', False) if isinstance(p_net_setting, dict) else getattr(getattr(p_net_setting, 'output', None), 'if_save', False)
            if p_net_dataset_dir:
                try:
                    self._pn = PhysicalNetwork.load_dataset(p_net_dataset_dir)
                except FileNotFoundError:
                    # Generate if not found
                    self._pn = Generator.generate_p_net_dataset_from_config(self.config, save=if_save)
                    print(f'\ndrop a INFO/Warning here at line {sys._getframe(0).f_lineno} of dataset_provider\n')
                    self.logger.warning(f'File in path:{p_net_dataset_dir} not Found , Generating from Setting ')
            else:
                # Generate from setting
                self._pn = Generator.generate_p_net_dataset_from_config(self.config, save= if_save)

        
        return self._pn
    

    def get_vn_requests(self) -> list:
        """Get the list of VN requests.
        
        Returns:
            List of VNRequest objects
        """
        if self._v_net_simulator is None:
            # Try to load from dataset directory first
            # Handle both DictConfig and dict
            if hasattr(self.config, 'get'):
                v_sim_setting = self.config.get('v_sim_setting', {})
            else:
                v_sim_setting = getattr(self.config, 'v_sim_setting', {})

            # Convert to dict if needed
            if hasattr(v_sim_setting, '_content'):
                v_sim_setting = dict(v_sim_setting)
            
            v_nets_dataset_dir = v_sim_setting.get('dataset_dir') if isinstance(v_sim_setting, dict) else getattr(v_sim_setting, 'dataset_dir', None)
            if_save = v_sim_setting.get('output', {}).get('if_save', False) if isinstance(v_sim_setting, dict) else getattr(getattr(v_sim_setting, 'output', None), 'if_save', False)

            if v_nets_dataset_dir:
                try:
                    self._v_net_simulator = VNRequestSimulator.load_dataset(v_nets_dataset_dir)
                except FileNotFoundError:
                    # Generate if not found
                    self._v_net_simulator = Generator.generate_v_nets_dataset_from_config(self.config, save=if_save)
                    print(f'\ndrop a INFO/Warning here at line {sys._getframe(0).f_lineno} of dataset_provider\n')
                    self.logger.warning(f'File in path:{v_nets_dataset_dir} not Found , Generating from Setting ')


            else:
                # Generate from setting
                self._v_net_simulator = Generator.generate_v_nets_dataset_from_config(self.config, save=if_save)
        
        return self._v_net_simulator.v_nets
    
    def get_v_net_simulator(self) -> VNRequestSimulator:
        """Get the VN request simulator.
        
        Returns:
            VNRequestSimulator instance
        """
        if self._v_net_simulator is None:
            # Try to load from dataset directory first
            # Handle both DictConfig and dict
            if hasattr(self.config, 'get'):
                v_sim_setting = self.config.get('v_sim_setting', {})
            else:
                v_sim_setting = getattr(self.config, 'v_sim_setting', {})
            
            # Convert to dict if needed
            if hasattr(v_sim_setting, '_content'):
                v_sim_setting = dict(v_sim_setting)
            
            v_nets_dataset_dir = v_sim_setting.get('dataset_dir') if isinstance(v_sim_setting, dict) else getattr(v_sim_setting, 'dataset_dir', None)
            if_save = v_sim_setting.get('output', {}).get('if_save', False) if isinstance(v_sim_setting, dict) else getattr(getattr(v_sim_setting, 'output', None), 'if_save', False)

            if v_nets_dataset_dir:
                try:
                    self._v_net_simulator = VNRequestSimulator.load_dataset(v_nets_dataset_dir)
                except FileNotFoundError:
                    # Generate if not found
                    self._v_net_simulator = Generator.generate_v_nets_dataset_from_config(self.config, save=if_save)
                    self.logger.warning(f'File in path:{v_nets_dataset_dir} not Found , Generating from Setting ')

            else:
                # Generate from setting
                self._v_net_simulator = Generator.generate_v_nets_dataset_from_config(self.config, save=if_save)
        
        return self._v_net_simulator













