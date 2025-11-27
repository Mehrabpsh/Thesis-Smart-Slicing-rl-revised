"""Virtual Network Request Simulator (Virne-style)."""

import os
from pathlib import Path

import copy
from typing import Optional, Union, List, Sequence, Dict
from dataclasses import dataclass, field, asdict
from omegaconf import DictConfig, OmegaConf

from .schemas import VNRequest, VNF
from .generators import generate_poisson_vn_stream
from .loaders import load_vn_requests_from_json
from ..utils.seed import set_seed, get_rng
from ..utils.serialization import save_json, load_json
from .lazy_loader import LazyVNRequestList


@dataclass
class VNRequestEvent:
    """Event in the virtual network request simulator.
    
    Attributes:
        id: Event ID
        type: Event type (1 for arrival, 0 for leave)
        v_net_id: Virtual network request ID
        time: Event time
    """
    id: int
    type: int
    v_net_id: int
    time: float
    
    def __post_init__(self):
        if self.type not in [0, 1]:
            raise ValueError("Event type must be 0 (leave) or 1 (arrival)")
        if self.v_net_id < 0:
            raise ValueError("Virtual network ID must be non-negative")
        if self.time < 0:
            raise ValueError("Event time must be non-negative")


class VNRequestSimulator:
    """Simulator for sequentially arriving virtual network requests (Virne-style)."""
    
    _cached_vnets_loads: Dict[str, 'VNRequestSimulator'] = {}  # Cache for loaded datasets
    
    def __init__(
        self,
        v_nets: Union[LazyVNRequestList,List],
        events: Sequence[VNRequestEvent] = [],
        v_sim_setting: dict = {},
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize VN request simulator.
        
        Args:
            v_nets: List of virtual network requests
            events: List of events
            v_sim_setting: Simulator setting dictionary
            seed: Optional random seed
            **kwargs: Additional keyword arguments
        """
        #if isinstance(v_nets, list):
        #   self.v_nets = list(v_nets)
        #else:
        self.v_nets = v_nets
        self.events = list(events)
        self.v_sim_setting = copy.deepcopy(v_sim_setting)
        self.seed = seed
        self._construct_v2event_dict()
    
    @property
    def num_v_nets(self) -> int:
        """Get the number of virtual networks."""
        return len(self.v_nets)
    
    @property
    def num_events(self) -> int:
        """Get the number of events."""
        return len(self.events)
    
    def _construct_v2event_dict(self) -> None:
        """Construct a dictionary mapping virtual network ID to event ID."""
        self.v2event_dict = {}
        for event in self.events:
            if event.type == 1:  # Arrival event
                self.v2event_dict[event.v_net_id] = event.id
    
    @staticmethod
    def from_setting(setting: Union[dict, DictConfig], seed: Optional[int] = None) -> 'VNRequestSimulator':
        """Create VNRequestSimulator from setting (Virne-style).
        
        Args:
            setting: Configuration dictionary
            seed: Optional random seed
            
        Returns:
            VNRequestSimulator instance
        """
        if seed is not None:
            set_seed(seed)
        
        # Convert DictConfig to dict if needed
        if isinstance(setting, DictConfig):
            setting_converted = OmegaConf.to_container(setting, resolve=True)
            assert isinstance(setting_converted, dict), "Converted setting must be a dict."
            setting = setting_converted
        
        #v_nets = []

        return VNRequestSimulator(v_nets=[], events=[], v_sim_setting=setting, seed=seed)
    
    def renew(self, v_nets: bool = True, events: bool = True, seed: Optional[int] = None) -> tuple:
        """Renew virtual networks and events.
        
        Args:
            v_nets: Whether to renew virtual networks
            events: Whether to renew events
            seed: Optional random seed
            
        Returns:
            Tuple of (v_nets, events)
        """
        if seed is not None:
            set_seed(seed)
        
        if v_nets:
            self._renew_v_nets()
        if events:
            self._renew_events()
        
        return self.v_nets, self.events
    
    def _renew_v_nets(self) -> None:
        """Generate virtual network requests."""
        # Get settings
        num_groups = self.v_sim_setting.get('num_groups', 1000)
        cache_size = self.v_sim_setting.get('cache_size', 1000)
        num_v_nets = self.v_sim_setting.get('num_v_nets', 200) * num_groups
        sfc_len = self.v_sim_setting.get('sfc_len', {'low': 3, 'high': 5})
        vnf_types = self.v_sim_setting.get('vnf_types', ['fw', 'nat', 'ids', 'wanopt'])
        cache_path = Path(self.v_sim_setting.get('cache_path', '.vnReqs_cache'))
        # Get node attributes (CPU demand)
        node_attrs_setting = self.v_sim_setting.get('node_attrs_setting', [])
        cpu_attr: Dict = next((attr for attr in node_attrs_setting if attr.get('name') == 'cpu'), {})
        cpu_range = (cpu_attr.get('low', 1), cpu_attr.get('high', 4))
        
        # Get link attributes (bandwidth and latency)
        link_attrs_setting = self.v_sim_setting.get('qos_attrs_setting', [])
        bw_attr: Dict = next((attr for attr in link_attrs_setting if attr.get('name') == 'bandwidth'), {})
        latency_attr: Dict = next((attr for attr in link_attrs_setting if attr.get('name') == 'latency'), {})
        bw_range = (bw_attr.get('low', 1), bw_attr.get('high', 5))
        latency_range = (latency_attr.get('low', 5), latency_attr.get('high', 30))
        
        # Get arrival rate
        arrival_rate = self.v_sim_setting.get('arrival_rate', {}).get('lam', 1.0)
        
        # Generate VN requests
        rng = get_rng()
        generate_poisson_vn_stream(
            num_requests=num_v_nets,
            num_groups=num_groups,
            rate=arrival_rate,
            sfc_len_range=(sfc_len.get('low', 3), sfc_len.get('high', 5)),
            vnf_types=vnf_types,
            cpu_demand_range=cpu_range,
            bw_demand_range=bw_range,
            sla_latency_range=latency_range,
            cache_path = cache_path,
            rng=rng,
            )
        self.v_nets = LazyVNRequestList(directory = cache_path, cache_size = cache_size)


    def _renew_events(self) -> None:
        """Generate events from virtual network requests."""
        events = []
        event_id = 0
        
        for v_net in self.v_nets:
            # Arrival event
            events.append(VNRequestEvent(
                id=event_id,
                type=1,
                v_net_id=v_net.request_id,
                time=v_net.arrival_time,
            ))
            event_id += 1
            
            # Leave event (arrival_time + lifetime)
            lifetime = self.v_sim_setting.get('lifetime', {}).get('scale', 100.0)
            leave_time = v_net.arrival_time + lifetime
            events.append(VNRequestEvent(
                id=event_id,
                type=0,
                v_net_id=v_net.request_id,
                time=leave_time,
            ))
            event_id += 1
        
        # Sort events by time
        self.events = sorted(events, key=lambda e: e.time)
        self._construct_v2event_dict()
    
    def save_dataset(self, save_dir: str) -> None:
        """Save the dataset to a directory (Virne-style).
        
        Args:
            save_dir: Directory to save the dataset
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save VN requests
        v_nets_dir = os.path.join(save_dir, 'v_nets')
        os.makedirs(v_nets_dir, exist_ok=True)
        
        for v_net in self.v_nets:
            v_net_data = {
                'request_id': v_net.request_id,
                'group_id' : v_net.group_id,
                'vnfs': [
                    {
                        'vnf_id': vnf.vnf_id,
                        'vnf_type': vnf.vnf_type,
                        'cpu_demand': vnf.cpu_demand,
                        'features': vnf.features,
                    }
                    for vnf in v_net.vnfs
                ],
                'bandwidth_demand': v_net.bandwidth_demand,
                'sla_latency_ms': v_net.sla_latency_ms,
                'arrival_time': v_net.arrival_time,
                'features': v_net.features,
            }
            from pathlib import Path
            save_json(v_net_data, Path(os.path.join(v_nets_dir, f'v_net-{v_net.request_id:05d}.json')))
        
        # Save events
        events_data = [asdict(event) for event in self.events]
        events_file = self.v_sim_setting.get('output', {}).get('events_file_name', 'events.json')
        save_json(events_data, Path(os.path.join(save_dir, events_file)))
        
        # Save setting
        setting_file = self.v_sim_setting.get('output', {}).get('setting_file_name', 'v_sim_setting.json')
        save_json(self.v_sim_setting, Path(os.path.join(save_dir, setting_file)))
    
    @staticmethod
    def load_dataset(dataset_dir: str) -> 'VNRequestSimulator':
        """Load the dataset from a directory (Virne-style).
        
        Args:
            dataset_dir: Directory containing the dataset
            
        Returns:
            VNRequestSimulator instance
        """
        # Check cache
        cache = VNRequestSimulator._cached_vnets_loads
        if 'seed_' in dataset_dir and dataset_dir in cache:
            return copy.deepcopy(cache[dataset_dir])
        
        # Load setting
        from pathlib import Path
        setting_file = os.path.join(dataset_dir, 'v_sim_setting.json')
        if not os.path.exists(setting_file):
            raise FileNotFoundError(f"Setting file not found: {setting_file}")
        v_sim_setting = load_json(Path(setting_file))
        
        # Load events
        events_file = os.path.join(dataset_dir, 'events.json')
        if not os.path.exists(events_file):
            raise FileNotFoundError(f"Events file not found: {events_file}")
        events_data = load_json(Path(events_file))
        events = [VNRequestEvent(**event_data) for event_data in events_data]
        
        # Load VN requests lazily
        v_nets_dir = os.path.join(dataset_dir, 'v_nets')
        if not os.path.exists(v_nets_dir):
            raise FileNotFoundError(f"v_nets directory not found: {v_nets_dir}")
        
        v_nets = LazyVNRequestList(v_nets_dir)
        
        simulator = VNRequestSimulator(v_nets=v_nets, events=events, v_sim_setting=v_sim_setting)
        
        # Cache the result
        if 'seed_' in dataset_dir:
            cache[dataset_dir] = copy.deepcopy(simulator)
        
        return simulator

