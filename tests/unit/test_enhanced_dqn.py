"""Unit tests for core DQN functionality"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from shipboard_ai.core.enhanced_dqn import EnhancedDQN, DQNAgent


class TestEnhancedDQN:
    """Test cases for Enhanced DQN model"""
    
    def test_model_initialization(self):
        """Test DQN model initialization"""
        state_size = 20
        action_size = 10
        hidden_size = 64
        
        model = EnhancedDQN(state_size, action_size, hidden_size)
        
        assert model.state_size == state_size
        assert model.action_size == action_size
        assert model.hidden_size == hidden_size
        assert isinstance(model.attention, torch.nn.Module)
        assert isinstance(model.q_network, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass through the network"""
        model = EnhancedDQN(state_size=20, action_size=10)
        
        # Create sample input
        batch_size = 32
        state = torch.randn(batch_size, 20)
        
        # Forward pass
        output = model(state)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_mechanism(self):
        """Test attention mechanism functionality"""
        model = EnhancedDQN(state_size=20, action_size=10)
        
        state = torch.randn(1, 20)
        attended_features = model.attention(state)
        
        assert attended_features.shape == (1, 20)
        assert not torch.isnan(attended_features).any()
    
    @pytest.mark.parametrize("state_size,action_size", [
        (10, 5),
        (50, 20),
        (100, 50)
    ])
    def test_different_network_sizes(self, state_size, action_size):
        """Test model with different input/output sizes"""
        model = EnhancedDQN(state_size, action_size)
        state = torch.randn(1, state_size)
        output = model(state)
        
        assert output.shape == (1, action_size)


class TestDQNAgent:
    """Test cases for DQN Agent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.state_size = 20
        self.action_size = 10
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.state_size == self.state_size
        assert self.agent.action_size == self.action_size
        assert self.agent.epsilon == 1.0  # Should start with high exploration
        assert len(self.agent.memory) == 0
    
    def test_remember_experience(self):
        """Test experience storage in memory"""
        state = np.random.random(self.state_size)
        action = 1
        reward = 10.0
        next_state = np.random.random(self.state_size)
        done = False
        
        self.agent.remember(state, action, reward, next_state, done)
        
        assert len(self.agent.memory) == 1
        stored_experience = self.agent.memory[0]
        np.testing.assert_array_equal(stored_experience[0], state)
        assert stored_experience[1] == action
        assert stored_experience[2] == reward
    
    def test_action_selection_exploration(self):
        """Test action selection during exploration"""
        self.agent.epsilon = 1.0  # Full exploration
        state = np.random.random(self.state_size)
        
        action = self.agent.act(state)
        
        assert 0 <= action < self.action_size
        assert isinstance(action, (int, np.integer))
    
    def test_action_selection_exploitation(self):
        """Test action selection during exploitation"""
        self.agent.epsilon = 0.0  # No exploration
        state = np.random.random(self.state_size)
        
        with patch.object(self.agent.q_network, 'forward') as mock_forward:
            mock_forward.return_value = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
            action = self.agent.act(state)
            
            assert action == 1  # Should select action with highest Q-value
    
    def test_epsilon_decay(self):
        """Test epsilon decay functionality"""
        initial_epsilon = self.agent.epsilon
        self.agent.epsilon_decay()
        
        assert self.agent.epsilon < initial_epsilon
        assert self.agent.epsilon >= self.agent.epsilon_min
    
    def test_replay_insufficient_memory(self):
        """Test replay when memory is insufficient"""
        # Add only a few experiences (less than batch size)
        for i in range(5):
            state = np.random.random(self.state_size)
            self.agent.remember(state, 0, 1.0, state, False)
        
        # Should not raise error and should return None or handle gracefully
        result = self.agent.replay()
        assert result is None or isinstance(result, dict)
    
    def test_replay_sufficient_memory(self):
        """Test replay with sufficient memory"""
        # Fill memory with enough experiences
        for i in range(100):
            state = np.random.random(self.state_size)
            action = i % self.action_size
            reward = np.random.random()
            next_state = np.random.random(self.state_size)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done)
        
        # Mock optimizer to avoid actual training
        with patch.object(self.agent.optimizer, 'step'):
            result = self.agent.replay()
            
            assert isinstance(result, dict)
            assert 'loss' in result
    
    def test_save_and_load_model(self, tmp_path):
        """Test model saving and loading"""
        filepath = tmp_path / "test_model.pth"
        
        # Save model
        self.agent.save_model(str(filepath))
        assert filepath.exists()
        
        # Create new agent and load model
        new_agent = DQNAgent(self.state_size, self.action_size)
        new_agent.load_model(str(filepath))
        
        # Compare model parameters
        original_params = list(self.agent.q_network.parameters())
        loaded_params = list(new_agent.q_network.parameters())
        
        for orig, loaded in zip(original_params, loaded_params):
            torch.testing.assert_close(orig, loaded)
    
    def test_get_training_stats(self):
        """Test training statistics collection"""
        # Add some training history
        for i in range(10):
            self.agent.training_history.append({
                'episode': i,
                'reward': i * 10,
                'loss': 1.0 / (i + 1),
                'epsilon': 1.0 - (i * 0.1)
            })
        
        stats = self.agent.get_training_stats()
        
        assert isinstance(stats, dict)
        assert 'total_episodes' in stats
        assert 'average_reward' in stats
        assert 'current_epsilon' in stats
        assert stats['total_episodes'] == 10
    
    @pytest.mark.parametrize("reward,expected_sign", [
        (10.0, "positive"),
        (-5.0, "negative"),
        (0.0, "zero")
    ])
    def test_reward_processing(self, reward, expected_sign):
        """Test different reward values"""
        state = np.random.random(self.state_size)
        action = 0
        next_state = np.random.random(self.state_size)
        
        self.agent.remember(state, action, reward, next_state, False)
        stored_reward = self.agent.memory[-1][2]
        
        if expected_sign == "positive":
            assert stored_reward > 0
        elif expected_sign == "negative":
            assert stored_reward < 0
        else:
            assert stored_reward == 0
