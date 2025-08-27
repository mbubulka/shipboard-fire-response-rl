"""Test configuration and fixtures for Shipboard Fire Response RL System"""

import pytest
import tempfile
import os
import torch
import numpy as np

# Set random seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_state():
    """Sample state tensor for testing"""
    return torch.randn(20)


@pytest.fixture
def sample_experience():
    """Sample experience tuple for testing"""
    return {
        'state': torch.randn(20),
        'action': 2,
        'reward': 1.0,
        'next_state': torch.randn(20),
        'done': False,
        'source_id': 0
    }


@pytest.fixture
def sample_training_batch():
    """Sample training batch for testing"""
    batch_size = 32
    return {
        'states': torch.randn(batch_size, 20),
        'actions': torch.randint(0, 8, (batch_size,)),
        'rewards': torch.randn(batch_size),
        'next_states': torch.randn(batch_size, 20),
        'dones': torch.randint(0, 2, (batch_size,)).bool(),
        'source_ids': torch.randint(0, 3, (batch_size,))
    }


class TestDataHelper:
    """Helper class for test data generation"""
    
    @staticmethod
    def create_fire_scenario(scenario_type="galley_fire", severity="moderate"):
        """Create a test fire scenario"""
        return {
            "scenario_type": scenario_type,
            "location": "test_location",
            "fire_type": "test_fire",
            "severity": severity,
            "response_actions": ["test_action_1", "test_action_2"],
            "expected_outcome": "fire_suppressed"
        }
    
    @staticmethod
    def create_random_state(state_dim=20):
        """Create random state for testing"""
        return torch.randn(state_dim)
    
    @staticmethod
    def create_action_sequence(length=10, action_dim=8):
        """Create sequence of random actions"""
        return torch.randint(0, action_dim, (length,)).tolist()


@pytest.fixture
def test_data_helper():
    """Test data helper fixture"""
    return TestDataHelper()
