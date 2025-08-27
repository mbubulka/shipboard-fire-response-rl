#!/usr/bin/env python3
"""
Simple import test for Shipboard Fire Response RL
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("🧪 Simple Import Test")
print("=" * 30)

try:
    # Test basic structure
    import shipboard_rl
    print("✅ shipboard_rl package imported")
    
    # Test scenario generator (no torch dependency)
    from shipboard_rl.scenario.generator import ShipboardScenarioGenerator
    print("✅ scenario generator imported")
    
    generator = ShipboardScenarioGenerator(seed=42)
    scenario = generator.generate_scenario("easy")
    print(f"✅ Generated scenario: {scenario.compartment.value}")
    
    print("\n🎉 Basic functionality working!")
    print("💡 Note: PyTorch components may require additional setup")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
