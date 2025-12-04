import os
import json
from pathlib import Path
from typing import List, Union, Optional, Iterator, Dict
from collections import OrderedDict

from .schemas import VNRequest, VNF
from ..utils.serialization import load_json

class LazyVNRequestList:
    """A list-like object that lazily loads VN requests from disk.
    
    It scans the directory for JSON files and stores their paths.
    Requests are loaded only when accessed, with an LRU cache to manage memory.
    """
    
    def __init__(self, directory: Union[str, Path, List[str]], cache_size: int = 1000):
        """Initialize LazyVNRequestList.
        
        Args:
            directory: Directory containing VN request JSON files, OR a list of file paths.
            cache_size: Maximum number of requests to keep in memory.
        """
        self.cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()
        
        if isinstance(directory, list):
            self.file_paths = directory
        else:
            self.directory = Path(directory)
            if not self.directory.exists():
                raise FileNotFoundError(f"Directory not found: {self.directory}")
            
            # Scan for JSON files
            self.file_paths = sorted([
                str(self.directory / f) 
                for f in os.listdir(self.directory) 
                if f.endswith('.json')
            ])

    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[VNRequest, 'LazyVNRequestList']:
        if isinstance(index, slice):
            # Return a new LazyVNRequestList with the subset of file paths
            return LazyVNRequestList(self.file_paths[index], cache_size=self.cache_size)
        
        if index < 0 and index >= -len(self):
            index += len(self)
            
        if index >= len(self) or index < -len(self):
            raise IndexError("LazyVNRequestList index out of range")
            
        file_path = self.file_paths[index]
        
        # Check cache
        if file_path in self._cache:
            self._cache.move_to_end(file_path)
            return self._cache[file_path]
        
        # Load from disk
        vn_request = self._load_request(file_path)
        
        # Update cache
        self._cache[file_path] = vn_request
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
            
        return vn_request
    
    def _load_request(self, file_path: str) -> VNRequest:
        """Load a single VN request from a JSON file."""
        try:
            v_net_data = load_json(Path(file_path))
            
            vnfs = [
                VNF(
                    vnf_id=vnf_data['vnf_id'],
                    vnf_type=vnf_data['vnf_type'],
                    cpu_demand=vnf_data['cpu_demand'],
                    features=vnf_data.get('features', {}),
                )
                for vnf_data in v_net_data['vnfs']
            ]
            
            v_net = VNRequest(
                request_id=v_net_data['request_id'],
                group_id=v_net_data['group_id'],
                vnfs=vnfs,
                bandwidth_demand=v_net_data['bandwidth_demand'],
                sla_latency_ms=v_net_data['sla_latency_ms'],
                arrival_time=v_net_data.get('arrival_time', 0.0),
                features=v_net_data.get('features', {}),
            )
            return v_net
        except Exception as e:
            raise RuntimeError(f"Failed to load VN request from {file_path}: {e}")

    def __iter__(self) -> Iterator[VNRequest]:
        for i in range(len(self)):
            yield self[i]
            
    def __repr__(self) -> str:
        return f"LazyVNRequestList(len={len(self)}, cache_size={self.cache_size})"
