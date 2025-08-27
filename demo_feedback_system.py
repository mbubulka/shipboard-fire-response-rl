#!/usr/bin/env python3
"""
Demo script showing the RL Feedback System in action
Demonstrates the complete feedback loop for model improvement
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shipboard_rl.feedback import (
    RLFeedbackDatabase,
    RLFeedbackAnalyzer,
    RLFeedbackData,
    FeedbackEnabledTrainer
)
from shipboard_rl.core.enhanced_dqn import EnhancedDQNAgent, EnhancedFireResponseEnvironment
import uuid
from datetime import datetime


def demo_feedback_collection():
    """Demonstrate basic feedback collection"""
    print("🔄 Demo: RL Feedback Collection")
    print("=" * 50)
    
    # Initialize feedback system
    db = RLFeedbackDatabase("demo_feedback.db")
    analyzer = RLFeedbackAnalyzer(db)
    
    # Create sample feedback data
    sample_feedback = RLFeedbackData(
        session_id=str(uuid.uuid4()),
        user_id="demo_user",
        scenario_id="demo_scenario_001",
        scenario_type="galley_fire",
        scenario_source="nfpa_1500",
        
        # RL episode data
        actions_taken=[0, 1, 2, 3, 4],
        q_values=[[0.8, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7] for _ in range(5)],
        episode_rewards=[1.0, 2.0, 3.0, 4.0, 5.0],
        episode_length=5,
        final_reward=15.0,
        success_rate=0.85,
        
        # User ratings
        difficulty_rating=3,
        ai_helpfulness=4,
        scenario_realism=4,
        confidence_level=3,
        
        # Qualitative feedback
        what_worked_well="AI made logical decisions",
        what_was_confusing="Some actions seemed suboptimal",
        suggested_improvements="Better handling of emergency scenarios",
        additional_comments="Overall good performance",
        
        # Context
        training_level="intermediate",
        previous_experience="2 years fire safety training"
    )
    
    # Store feedback
    feedback_id = db.store_feedback(sample_feedback)
    print(f"✅ Stored feedback with ID: {feedback_id}")
    
    # Add some action-level feedback
    for i in range(5):
        action_feedback_id = db.store_action_feedback(
            session_id=sample_feedback.session_id,
            step_number=i,
            state_vector=[0.1, 0.2, 0.3] * 7,  # 21-element state vector
            action_taken=i,
            q_values=[0.8, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
            reward_received=float(i + 1),
            user_agreement=4,  # High agreement
            feedback_text=f"Good action choice at step {i}"
        )
        print(f"   📊 Action feedback {i}: ID {action_feedback_id}")
    
    return db, analyzer


def demo_feedback_analysis(db, analyzer):
    """Demonstrate feedback analysis capabilities"""
    print("\n🔍 Demo: Feedback Analysis")
    print("=" * 50)
    
    # Get recent feedback
    recent_feedback = db.get_recent_feedback(days=30)
    print(f"📈 Recent feedback entries: {len(recent_feedback)}")
    
    if recent_feedback:
        feedback = recent_feedback[0]
        print(f"   Latest feedback:")
        print(f"   - Success rate: {feedback['success_rate']:.2f}")
        print(f"   - Final reward: {feedback['final_reward']:.1f}")
        print(f"   - AI helpfulness: {feedback['ai_helpfulness']}/5")
        print(f"   - User confidence: {feedback['confidence_level']}/5")
    
    # Analyze action patterns
    action_patterns = analyzer.analyze_action_patterns()
    print(f"\n🎯 Action patterns found: {len(action_patterns.get('action_patterns', []))}")
    
    # Identify improvement areas
    improvements = analyzer.identify_improvement_areas()
    print(f"🔧 Improvement areas identified:")
    print(f"   - Low performance scenarios: {len(improvements['low_performance_scenarios'])}")
    print(f"   - Problematic actions: {len(improvements['problematic_actions'])}")
    
    # Generate training data
    training_data = analyzer.generate_training_data(min_agreement=3)
    print(f"🎓 Training data generated:")
    print(f"   - Training examples: {training_data['total_examples']}")
    print(f"   - Expert corrections: {training_data['expert_examples']}")
    
    return training_data


def demo_integrated_training():
    """Demonstrate integrated training with feedback"""
    print("\n🤖 Demo: Integrated Training with Feedback")
    print("=" * 50)
    
    # Initialize RL components
    agent = EnhancedDQNAgent(state_dim=20, action_dim=8)
    environment = EnhancedFireResponseEnvironment()
    
    # Initialize feedback-enabled trainer
    trainer = FeedbackEnabledTrainer(agent, environment)
    
    # Start feedback session
    session_id = trainer.start_feedback_session(user_id="demo_user")
    print(f"🚀 Started training session: {session_id}")
    
    # Run a few training episodes
    for episode in range(3):
        print(f"\n📚 Episode {episode + 1}:")
        
        # Train episode with feedback collection
        episode_results = trainer.train_episode_with_feedback(scenario_source="nfpa_1500")
        
        print(f"   Reward: {episode_results['total_reward']:.2f}")
        print(f"   Length: {episode_results['episode_length']} steps")
        print(f"   Success: {episode_results['success_rate']:.2f}")
        
        # Simulate user feedback
        user_ratings = {
            'difficulty_rating': 3,
            'ai_helpfulness': 4,
            'scenario_realism': 4,
            'confidence_level': 3,
            'training_level': 'intermediate'
        }
        
        qualitative_feedback = {
            'what_worked_well': 'Good logical progression',
            'what_was_confusing': 'Some actions seemed random',
            'suggested_improvements': 'Better action explanations'
        }
        
        # Collect feedback
        feedback_id = trainer.collect_episode_feedback(
            user_id="demo_user",
            episode_results=episode_results,
            user_ratings=user_ratings,
            qualitative_feedback=qualitative_feedback
        )
        
        print(f"   📊 Feedback collected: {feedback_id}")
    
    # Analyze performance
    performance = trainer.analyze_recent_performance(days=1)
    print(f"\n📈 Performance Analysis:")
    if 'overall_performance' in performance:
        overall = performance['overall_performance']
        print(f"   Episodes: {overall['total_episodes']}")
        print(f"   Avg Success: {overall['avg_success_rate']:.3f}")
        print(f"   Avg Reward: {overall['avg_reward']:.2f}")
    
    # Check if retraining is needed
    should_retrain, metrics = trainer.should_retrain_model()
    print(f"\n🔄 Retraining Analysis:")
    print(f"   Should retrain: {should_retrain}")
    print(f"   Feedback count: {metrics.get('feedback_count', 0)}")
    
    return trainer


def demo_api_endpoints():
    """Demonstrate API endpoint functionality"""
    print("\n🌐 Demo: API Endpoints")
    print("=" * 50)
    
    try:
        from shipboard_rl.feedback import feedback_app, FLASK_AVAILABLE
        
        if FLASK_AVAILABLE and feedback_app:
            print("✅ Flask API available")
            print("🔗 Available endpoints:")
            print("   POST /api/rl-feedback/session/start")
            print("   POST /api/rl-feedback/session/action")
            print("   POST /api/rl-feedback/submit")
            print("   GET  /api/rl-feedback/analytics/summary")
            print("   GET  /api/rl-feedback/improvement-areas")
            print("   POST /api/rl-feedback/trigger-retraining")
            print("   GET  /feedback/form")
            print("   GET  /api/rl-feedback/status")
            print("\n💡 To start the API server, run:")
            print("   python -m shipboard_rl.feedback.feedback_api")
        else:
            print("⚠️  Flask not available - API endpoints disabled")
            print("   Install with: pip install flask flask-cors")
    
    except ImportError as e:
        print(f"⚠️  API import failed: {e}")
        print("   Install dependencies: pip install flask flask-cors")


def main():
    """Run the complete feedback system demo"""
    print("🎯 Shipboard Fire Response RL Feedback System Demo")
    print("=" * 60)
    print()
    
    try:
        # Demo 1: Basic feedback collection
        db, analyzer = demo_feedback_collection()
        
        # Demo 2: Feedback analysis
        training_data = demo_feedback_analysis(db, analyzer)
        
        # Demo 3: Integrated training
        trainer = demo_integrated_training()
        
        # Demo 4: API endpoints
        demo_api_endpoints()
        
        print("\n🎉 All demos completed successfully!")
        print("\n📊 Summary:")
        print("✅ Feedback collection system working")
        print("✅ Analytics and improvement detection working")
        print("✅ Integrated training with feedback working")
        print("✅ API endpoints configured")
        print("\n🚀 The feedback system is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
