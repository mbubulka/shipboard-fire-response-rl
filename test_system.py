#!/usr/bin/env python3
"""
Quick system test for Shipboard Fire Response RL
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("🧪 Testing Shipboard Fire Response RL System")
    print("=" * 50)
    
    # Test core imports
    print("📦 Testing imports...")
    from shipboard_rl.core.enhanced_dqn import EnhancedDQNAgent, EnhancedFireResponseEnvironment
    from shipboard_rl.scenario.generator import ShipboardScenarioGenerator
    from shipboard_rl.training.trainer import TrainingManager
    print("   ✅ All core modules imported successfully")
    
    # Test scenario generation
    print("\n🔥 Testing scenario generation...")
    generator = ShipboardScenarioGenerator(seed=42)
    scenario = generator.generate_scenario("easy")
    print(f"   ✅ Generated scenario: {scenario.compartment.value} - {scenario.fire_type.value}")
    
    # Test environment creation
    print("\n🌍 Testing environment...")
    env = EnhancedFireResponseEnvironment()
    print(f"   ✅ Environment created with state_dim={env.state_dim}, action_dim={env.action_dim}")
    
    # Test agent creation
    print("\n🤖 Testing DQN agent...")
    agent = EnhancedDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    print("   ✅ Enhanced DQN agent created successfully")
    
    # Test training manager
    print("\n🎯 Testing training manager...")
    trainer = TrainingManager()
    print("   ✅ Training manager initialized")
    
    print("\n🎉 All systems operational!")
    print("🚀 Ready for deployment to GitHub!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
