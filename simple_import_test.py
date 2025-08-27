"""Simple import test"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from shipboard_rl.core.enhanced_dqn import EnhancedDQNAgent, EnhancedFireResponseEnvironment
    print("✅ Imports successful")
    
    # Test basic initialization
    agent = EnhancedDQNAgent(state_dim=20, action_dim=8)
    print("✅ Agent created successfully")
    
    env = EnhancedFireResponseEnvironment()
    print("✅ Environment created successfully")
    
    print("🎉 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
