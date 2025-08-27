"""
Shipboard Fire Response Reinforcement Learning System

An advanced RL system for shipboard fire response training and decision support,
integrating multiple maritime safety standards and providing real-time guidance
for emergency response scenarios.

Key Features:
- Enhanced Deep Q-Network (DQN) for decision-making
- Multi-source training data integration (NFPA, USCG, Maritime standards)
- Real-time feedback system for continuous learning
- Command-line interface for training and evaluation
- Comprehensive maritime safety compliance

Standards Integration:
- NFPA 1500: Fire Department Occupational Safety and Health Program
- NFPA 1521: Fire Department Safety Officer Professional Qualifications
- NFPA 1670: Operations and Training for Technical Search and Rescue
- USCG CG-022: Maritime Safety Standards
- International Maritime Safety Protocols

Author: Shipboard Fire Response RL Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Shipboard Fire Response RL Team"
__license__ = "MIT"

# Core imports for easy access
from .core.enhanced_dqn import EnhancedFireResponseDQN, EnhancedDQNAgent
from .training.trainer import TrainingManager
from .scenario.generator import ShipboardScenarioGenerator

# Legacy aliases for backward compatibility
EnhancedDQN = EnhancedFireResponseDQN
DQNAgent = EnhancedDQNAgent

__all__ = [
    "EnhancedFireResponseDQN",
    "EnhancedDQNAgent",
    "TrainingManager", 
    "ShipboardScenarioGenerator",
    "EnhancedDQN",  # Legacy alias
    "DQNAgent",     # Legacy alias
    "__version__",
    "__author__",
    "__license__"
]