#!/usr/bin/env python3
"""
Enhanced DQN for Comprehensive Shipboard Fire Response Training
Integrates NFPA, USCG, and Maritime training standards
Uses multi-source scenario data for improved learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class EnhancedFireResponseDQN(nn.Module):
    """
    Enhanced Deep Q-Network with multi-source training integration
    Supports NFPA, USCG, and Maritime training standards
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 512, num_sources: int = 5):
        super(EnhancedFireResponseDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources  # NFPA, USCG, Maritime sources
        
        # Training source embeddings
        self.source_embedding = nn.Embedding(num_sources, 64)
        
        # Enhanced state processing with source awareness
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dim),  # +64 for source embedding
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-head attention for scenario complexity
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Standards-aware processing layers
        self.nfpa_layer = nn.Linear(hidden_dim, hidden_dim // 2)
        self.uscg_layer = nn.Linear(hidden_dim, hidden_dim // 2) 
        self.maritime_layer = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2) * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layers for different action types
        self.tactical_actions = nn.Linear(hidden_dim, action_dim // 3)  # Tactical decisions
        self.safety_actions = nn.Linear(hidden_dim, action_dim // 3)   # Safety decisions  
        self.coord_actions = nn.Linear(hidden_dim, action_dim // 3)    # Coordination decisions
        
        # Combined output
        self.final_layer = nn.Linear(action_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, state: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with source-aware processing
        
        Args:
            state: Environmental state tensor
            source_ids: Training source IDs (0=NFPA1500, 1=NFPA1521, 2=NFPA1670, 3=USCG, 4=Maritime)
        """
        batch_size = state.size(0)
        
        # Get source embeddings
        source_emb = self.source_embedding(source_ids)  # [batch_size, 64]
        
        # Combine state with source awareness
        enhanced_state = torch.cat([state, source_emb], dim=-1)  # [batch_size, state_dim + 64]
        
        # Encode enhanced state
        encoded = self.state_encoder(enhanced_state)  # [batch_size, hidden_dim]
        
        # Apply self-attention for scenario complexity
        attended, _ = self.attention(
            encoded.unsqueeze(1), 
            encoded.unsqueeze(1), 
            encoded.unsqueeze(1)
        )
        attended = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Standards-specific processing
        nfpa_features = F.relu(self.nfpa_layer(attended))
        uscg_features = F.relu(self.uscg_layer(attended))
        maritime_features = F.relu(self.maritime_layer(attended))
        
        # Integrate all features
        integrated = torch.cat([attended, nfpa_features, uscg_features, maritime_features], dim=-1)
        processed = self.integration_layer(integrated)
        
        # Generate different types of actions
        tactical = self.tactical_actions(processed)
        safety = self.safety_actions(processed)
        coordination = self.coord_actions(processed)
        
        # Combine action outputs
        combined_actions = torch.cat([tactical, safety, coordination], dim=-1)
        q_values = self.final_layer(combined_actions)
        
        return q_values

class EnhancedFireResponseEnvironment:
    """
    Enhanced environment using comprehensive training scenarios
    """
    
    def __init__(self, training_data_dir: str = "./data/training"):
        self.training_dir = Path(training_data_dir)
        self.load_training_data()
        self.setup_environment()
        
    def load_training_data(self):
        """Load comprehensive training scenarios"""
        try:
            # Load integrated scenarios
            with open(self.training_dir / "integrated_scenarios.json", 'r') as f:
                self.scenarios = json.load(f)
                
            # Load knowledge base
            with open(self.training_dir / "comprehensive_knowledge_base.json", 'r') as f:
                self.knowledge_base = json.load(f)
                
            print(f"✅ Loaded {len(self.scenarios)} enhanced training scenarios")
            
        except FileNotFoundError:
            print("⚠️  Enhanced training data not found, using basic scenarios")
            self.scenarios = self._create_basic_scenarios()
            self.knowledge_base = self._create_basic_knowledge()
    
    def _create_basic_scenarios(self):
        """Fallback basic scenarios if enhanced data not available"""
        return [
            {
                "id": "basic_001",
                "title": "Basic Engine Room Fire",
                "category": "fire_suppression",
                "sources": ["maritime_standards"],
                "description": "Basic fire response scenario"
            },
            {
                "id": "basic_002", 
                "title": "Galley Fire Response",
                "category": "safety_management",
                "sources": ["nfpa_1500"],
                "description": "Kitchen fire suppression scenario"
            },
            {
                "id": "basic_003",
                "title": "Electrical Fire Emergency", 
                "category": "technical_rescue",
                "sources": ["nfpa_1670"],
                "description": "Electrical system fire response"
            }
        ]
    
    def _create_basic_knowledge(self):
        """Fallback basic knowledge base"""
        return {
            "system_info": {"name": "Basic Fire Response System"},
            "training_standards": {
                "nfpa_1500": "Fire Department Occupational Safety",
                "nfpa_1521": "Fire Department Safety Officer",
                "nfpa_1670": "Technical Search and Rescue",
                "uscg_cg022": "Maritime Safety Standards",
                "maritime_rvss": "Risk and Vulnerability Standards"
            }
        }
    
    def setup_environment(self):
        """Setup enhanced environment parameters"""
        # Enhanced state space including multi-source factors
        self.state_dim = 64  # Increased for comprehensive scenarios
        
        # Enhanced action space for tactical/safety/coordination
        self.action_dim = 27  # 9 tactical + 9 safety + 9 coordination actions
        
        # Source mapping
        self.source_map = {
            "nfpa_1500": 0,
            "nfpa_1521": 1, 
            "nfpa_1670": 2,
            "uscg_cg022": 3,
            "maritime_rvss": 4
        }
        
        # Action categories
        self.action_categories = {
            "tactical": list(range(9)),      # Actions 0-8: Fire suppression tactics
            "safety": list(range(9, 18)),    # Actions 9-17: Safety procedures
            "coordination": list(range(18, 27))  # Actions 18-26: Inter-agency coordination
        }
        
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, int]:
        """Reset environment with random enhanced scenario"""
        # Select random scenario
        self.current_scenario = np.random.choice(self.scenarios)
        
        # Determine source ID
        sources = self.current_scenario.get("sources", ["maritime_standards"])
        primary_source = sources[0] if sources else "maritime_standards"
        self.current_source_id = self.source_map.get(primary_source, 4)
        
        # Generate enhanced state representation
        self.state = self._generate_state_from_scenario(self.current_scenario)
        
        return self.state, self.current_source_id
    
    def _generate_state_from_scenario(self, scenario: Dict) -> np.ndarray:
        """Generate state vector from scenario data"""
        state = np.zeros(self.state_dim)
        
        # Encode scenario category
        category_encoding = {
            "safety_management": [1, 0, 0, 0],
            "technical_rescue": [0, 1, 0, 0], 
            "marine_operations": [0, 0, 1, 0],
            "fire_suppression": [0, 0, 0, 1]
        }
        
        category = scenario.get("category", "fire_suppression")
        encoding = category_encoding.get(category, [0, 0, 0, 1])
        state[0:4] = encoding
        
        # Encode scenario complexity (based on number of sources)
        sources = scenario.get("sources", [])
        complexity = len(sources) / 5.0  # Normalize by max sources
        state[4] = complexity
        
        # Add random environmental factors
        state[5:] = np.random.normal(0, 0.1, self.state_dim - 5)
        
        return state
    
    def step(self, action: int, source_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return results"""
        # Calculate reward based on action appropriateness
        reward = self._calculate_enhanced_reward(action, source_id)
        
        # Update state
        self.state = self._update_state(action)
        
        # Check if scenario is complete
        done = np.random.random() < 0.1  # 10% chance to end
        
        # Additional info
        info = {
            "scenario": self.current_scenario["title"],
            "sources": self.current_scenario.get("sources", []),
            "action_category": self._get_action_category(action)
        }
        
        return self.state, reward, done, info
    
    def _calculate_enhanced_reward(self, action: int, source_id: int) -> float:
        """Calculate reward considering training standards"""
        base_reward = np.random.normal(1.0, 0.2)
        
        # Bonus for appropriate action category
        action_category = self._get_action_category(action)
        scenario_category = self.current_scenario.get("category", "fire_suppression")
        
        category_bonus = 0.0
        if scenario_category == "safety_management" and action_category == "safety":
            category_bonus = 0.5
        elif scenario_category == "technical_rescue" and action_category == "tactical":
            category_bonus = 0.5
        elif scenario_category == "marine_operations" and action_category == "coordination":
            category_bonus = 0.5
        
        # Source consistency bonus
        sources = self.current_scenario.get("sources", [])
        source_names = list(self.source_map.keys())
        current_source_name = source_names[source_id] if source_id < len(source_names) else "unknown"
        
        source_bonus = 0.2 if current_source_name in [s.lower().replace("-", "_") for s in sources] else 0.0
        
        total_reward = base_reward + category_bonus + source_bonus
        return np.clip(total_reward, -1.0, 2.0)
    
    def _get_action_category(self, action: int) -> str:
        """Get action category from action index"""
        if action in self.action_categories["tactical"]:
            return "tactical"
        elif action in self.action_categories["safety"]:
            return "safety"
        elif action in self.action_categories["coordination"]:
            return "coordination"
        else:
            return "unknown"
    
    def _update_state(self, action: int) -> np.ndarray:
        """Update state based on action"""
        # Simple state update with noise
        self.state += np.random.normal(0, 0.05, self.state_dim)
        return np.clip(self.state, -1.0, 1.0)

class EnhancedDQNAgent:
    """
    Enhanced DQN Agent with multi-source training capability
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        # Enhanced networks
        self.q_network = EnhancedFireResponseDQN(state_dim, action_dim).to(device)
        self.target_network = EnhancedFireResponseDQN(state_dim, action_dim).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64
        
        print(f"✅ Enhanced DQN Agent initialized on {device}")
        print(f"📊 State dim: {state_dim}, Action dim: {action_dim}")
    
    def select_action(self, state: np.ndarray, source_id: int) -> int:
        """Select action using epsilon-greedy with source awareness"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            source_tensor = torch.LongTensor([source_id]).to(self.device)
            q_values = self.q_network(state_tensor, source_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, source_id: int,
                        next_source_id: int):
        """Store experience in replay buffer"""
        experience = (state, action, reward, next_state, done, source_id, next_source_id)
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(experience)
    
    def train(self) -> Optional[float]:
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]
        
        # Unpack batch
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        source_ids = torch.LongTensor([e[5] for e in experiences]).to(self.device)
        next_source_ids = torch.LongTensor([e[6] for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states, source_ids).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states, next_source_ids).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"✅ Enhanced DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"✅ Enhanced DQN model loaded from {filepath}")

# Legacy aliases for backward compatibility
EnhancedDQN = EnhancedFireResponseDQN
DQNAgent = EnhancedDQNAgent

def main():
    """Main training function for enhanced DQN"""
    print("🔥 Enhanced Shipboard Fire Response DQN Training System")
    print("📚 Integrating NFPA, USCG, and Maritime training standards")
    
    # Initialize enhanced environment and agent
    env = EnhancedFireResponseEnvironment()
    agent = EnhancedDQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=1e-3,
        gamma=0.99
    )
    
    print(f"🎯 Training environment ready with {len(env.scenarios)} scenarios")
    return env, agent

if __name__ == "__main__":
    env, agent = main()
