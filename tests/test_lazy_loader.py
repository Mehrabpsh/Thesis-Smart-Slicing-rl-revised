import unittest
import tempfile
import shutil
import os
import json
import sys
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['networkx'] = MagicMock()
sys.modules['omegaconf'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['colorlog'] = MagicMock()

from pathlib import Path
from sfc_rl.data.lazy_loader import LazyVNRequestList
from sfc_rl.data.schemas import VNRequest, VNF
from sfc_rl.data.vn_request_simulator import VNRequestSimulator

class TestLazyVNRequestList(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.v_nets_dir = os.path.join(self.test_dir, 'v_nets')
        os.makedirs(self.v_nets_dir)
        
        # Create dummy requests
        self.num_requests = 100
        for i in range(self.num_requests):
            data = {
                'request_id': i,
                'group_id': 0,
                'vnfs': [
                    {'vnf_id': 0, 'vnf_type': 'fw', 'cpu_demand': 1.0, 'features': {}},
                    {'vnf_id': 1, 'vnf_type': 'nat', 'cpu_demand': 2.0, 'features': {}}
                ],
                'bandwidth_demand': 5.0,
                'sla_latency_ms': 20.0,
                'arrival_time': float(i),
                'features': {}
            }
            with open(os.path.join(self.v_nets_dir, f'v_net-{i:05d}.json'), 'w') as f:
                json.dump(data, f)
                
        # Create dummy settings and events for simulator test
        with open(os.path.join(self.test_dir, 'v_sim_setting.json'), 'w') as f:
            json.dump({}, f)
        with open(os.path.join(self.test_dir, 'events.json'), 'w') as f:
            json.dump([], f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_len(self):
        lazy_list = LazyVNRequestList(self.v_nets_dir)
        self.assertEqual(len(lazy_list), self.num_requests)

    def test_getitem(self):
        lazy_list = LazyVNRequestList(self.v_nets_dir)
        req = lazy_list[0]
        self.assertIsInstance(req, VNRequest)
        self.assertEqual(req.request_id, 0)
        self.assertEqual(len(req.vnfs), 2)
        
        req5 = lazy_list[5]
        self.assertEqual(req5.request_id, 5)

    def test_caching(self):
        lazy_list = LazyVNRequestList(self.v_nets_dir, cache_size=2)
        
        # Access 0, 1, 2. Cache should have 1, 2 (0 evicted)
        _ = lazy_list[0]
        _ = lazy_list[1]
        _ = lazy_list[2]
        
        self.assertNotIn(lazy_list.file_paths[0], lazy_list._cache)
        self.assertIn(lazy_list.file_paths[1], lazy_list._cache)
        self.assertIn(lazy_list.file_paths[2], lazy_list._cache)
        
        # Access 1 again. Cache should have 2, 1 (LRU update)
        _ = lazy_list[1]
        # Access 3. Cache should have 1, 3 (2 evicted)
        _ = lazy_list[3]
        
        self.assertNotIn(lazy_list.file_paths[2], lazy_list._cache)
        self.assertIn(lazy_list.file_paths[1], lazy_list._cache)
        self.assertIn(lazy_list.file_paths[3], lazy_list._cache)

    def test_slicing(self):
        lazy_list = LazyVNRequestList(self.v_nets_dir)
        sliced = lazy_list[2:50]
        self.assertIsInstance(sliced, LazyVNRequestList)
        self.assertEqual(len(sliced), 48)
        self.assertEqual(sliced[0].request_id, 2)
        self.assertEqual(sliced[2].request_id, 4)
        print(len(sliced))
        print(len(list(sliced)))


    def test_simulator_integration(self):
        sim = VNRequestSimulator.load_dataset(self.test_dir)
        self.assertIsInstance(sim.v_nets, LazyVNRequestList)
        self.assertEqual(len(sim.v_nets), self.num_requests)
        self.assertEqual(sim.v_nets[0].request_id, 0)

if __name__ == '__main__':
    unittest.main()
