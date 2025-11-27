"""Tests for replay buffer."""

import pytest
import numpy as np
from sfc_rl.models.replay_buffer import ReplayBuffer


def test_replay_buffer_push():
    """Test adding experiences to replay buffer."""
    buffer = ReplayBuffer(capacity=100)
    
    state = np.array([1.0, 2.0, 3.0])
    action = 0
    reward = 1.0
    next_state = np.array([2.0, 3.0, 4.0])
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1


def test_replay_buffer_sample():
    """Test sampling from replay buffer."""
    buffer = ReplayBuffer(capacity=100)
    
    # Add multiple experiences
    for i in range(10):
        state = np.array([float(i), float(i+1), float(i+2)])
        action = i % 3
        reward = float(i)
        next_state = np.array([float(i+1), float(i+2), float(i+3)])
        done = i == 9
        buffer.push(state, action, reward, next_state, done)
    
    # Sample batch
    batch = buffer.sample(batch_size=5)
    states, actions, rewards, next_states, dones, action_masks, next_action_masks = batch
    
    assert states.shape[0] == 5
    assert actions.shape[0] == 5
    assert rewards.shape[0] == 5
    assert next_states.shape[0] == 5
    assert dones.shape[0] == 5


def test_replay_buffer_capacity():
    """Test replay buffer capacity limit."""
    buffer = ReplayBuffer(capacity=10)
    
    # Add more than capacity
    for i in range(20):
        state = np.array([float(i)])
        action = 0
        reward = 1.0
        next_state = np.array([float(i+1)])
        done = False
        buffer.push(state, action, reward, next_state, done)
    
    # Should only keep last 10
    assert len(buffer) == 10


def test_replay_buffer_with_masks():
    """Test replay buffer with action masks."""
    buffer = ReplayBuffer(capacity=100)
    
    state = np.array([1.0, 2.0])
    action = 0
    reward = 1.0
    next_state = np.array([2.0, 3.0])
    done = False
    action_mask = np.array([1.0, 0.0, 1.0])
    next_action_mask = np.array([1.0, 1.0, 0.0])
    
    buffer.push(state, action, reward, next_state, done, action_mask, next_action_mask)
    
    batch = buffer.sample(batch_size=1)
    states, actions, rewards, next_states, dones, action_masks, next_action_masks = batch
    
    assert action_masks is not None
    assert next_action_masks is not None
    assert action_masks.shape[1] == 3
    assert next_action_masks.shape[1] == 3

