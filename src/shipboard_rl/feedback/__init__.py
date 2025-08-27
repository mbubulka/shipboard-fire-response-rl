#!/usr/bin/env python3
"""
Shipboard Fire Response RL Feedback System
Comprehensive feedback collection and model improvement framework
"""

from .feedback_system import (
    RLFeedbackData,
    RLFeedbackDatabase, 
    RLFeedbackAnalyzer,
    RLFeedbackIntegration
)

from .trainer_integration import (
    FeedbackEnabledTrainer,
    run_feedback_enabled_training
)

# Conditional import for API (Flask may not be available in all environments)
try:
    from .feedback_api import app as feedback_app
    FLASK_AVAILABLE = True
except ImportError:
    feedback_app = None
    FLASK_AVAILABLE = False

__all__ = [
    'RLFeedbackData',
    'RLFeedbackDatabase',
    'RLFeedbackAnalyzer', 
    'RLFeedbackIntegration',
    'FeedbackEnabledTrainer',
    'run_feedback_enabled_training',
    'feedback_app',
    'FLASK_AVAILABLE'
]

__version__ = '1.0.0'
