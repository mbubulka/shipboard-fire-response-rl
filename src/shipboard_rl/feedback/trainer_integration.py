#!/usr/bin/env python3
"""
Integration module for Enhanced DQN with Feedback System
Connects the RL training with feedback collection and model improvement
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import uuid

from ..core.enhanced_dqn import EnhancedDQNAgent, EnhancedFireResponseEnvironment
from .feedback_system import RLFeedbackDatabase, RLFeedbackAnalyzer, RLFeedbackData


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackEnabledTrainer:
    """Enhanced DQN trainer with integrated feedback collection"""
    
    def __init__(self, 
                 agent: EnhancedDQNAgent,
                 environment: EnhancedFireResponseEnvironment,
                 feedback_db: Optional[RLFeedbackDatabase] = None):
        self.agent = agent
        self.environment = environment
        self.feedback_db = feedback_db or RLFeedbackDatabase()
        self.analyzer = RLFeedbackAnalyzer(self.feedback_db)
        
        # Training metrics
        self.current_session_id = None
        self.episode_actions = []
        self.episode_q_values = []
        self.episode_rewards = []
        self.episode_states = []
        
    def start_feedback_session(self, user_id: str = "system", 
                             scenario_type: str = "mixed") -> str:
        """Start a new feedback collection session"""
        self.current_session_id = str(uuid.uuid4())
        
        # Reset episode tracking
        self.episode_actions = []
        self.episode_q_values = []
        self.episode_rewards = []
        self.episode_states = []
        
        logger.info(f"✅ Started feedback session: {self.current_session_id}")
        return self.current_session_id
    
    def train_episode_with_feedback(self, scenario_source: str = "nfpa_1500") -> Dict:
        """Train one episode while collecting feedback data"""
        if not self.current_session_id:
            self.start_feedback_session()
        
        # Reset environment
        state, source_id = self.environment.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        # Clear episode data
        self.episode_actions = []
        self.episode_q_values = []
        self.episode_rewards = []
        self.episode_states = []
        
        while not done and step_count < 100:  # Max episode length
            # Store current state
            self.episode_states.append(state.tolist())
            
            # Get Q-values and select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                source_tensor = torch.LongTensor([source_id]).to(self.agent.device)
                q_values = self.agent.q_network(state_tensor, source_tensor).squeeze(0)
            
            action = self.agent.select_action(state, source_id)
            
            # Store action and Q-values
            self.episode_actions.append(action)
            self.episode_q_values.append(q_values.tolist())
            
            # Take action in environment
            next_state, reward, done, info = self.environment.step(action, source_id)
            
            # Store reward
            self.episode_rewards.append(reward)
            total_reward += reward
            
            # Store experience for training
            self.agent.store_experience(state, action, reward, next_state, done, 
                                      source_id, source_id)
            
            # Train the agent
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.train()
            else:
                loss = 0.0
            
            state = next_state
            step_count += 1
        
        # Calculate success rate based on final reward
        success_rate = min(1.0, max(0.0, (total_reward + 50) / 100))  # Normalize to 0-1
        
        episode_results = {
            'session_id': self.current_session_id,
            'total_reward': total_reward,
            'episode_length': step_count,
            'success_rate': success_rate,
            'scenario_source': scenario_source,
            'scenario_type': self.environment.current_scenario.get('scenario_type', 'unknown'),
            'final_state': state.tolist() if not done else None
        }
        
        logger.info(f"Episode completed: reward={total_reward:.2f}, "
                   f"length={step_count}, success={success_rate:.2f}")
        
        return episode_results
    
    def collect_episode_feedback(self, 
                               user_id: str,
                               episode_results: Dict,
                               user_ratings: Dict,
                                 qualitative_feedback: Optional[Dict] = None) -> int:
        """Collect comprehensive feedback for the completed episode"""
        
        if not self.current_session_id:
            raise ValueError("No active session. Start a session first.")
        
        # Create feedback data
        feedback = RLFeedbackData(
            session_id=self.current_session_id,
            user_id=user_id,
            scenario_id=f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scenario_type=episode_results.get('scenario_type', 'unknown'),
            scenario_source=episode_results.get('scenario_source', 'nfpa_1500'),
            
            # RL episode data
            actions_taken=self.episode_actions,
            q_values=self.episode_q_values,
            episode_rewards=self.episode_rewards,
            episode_length=episode_results['episode_length'],
            final_reward=episode_results['total_reward'],
            success_rate=episode_results['success_rate'],
            
            # User ratings (required)
            difficulty_rating=user_ratings['difficulty_rating'],
            ai_helpfulness=user_ratings['ai_helpfulness'],
            scenario_realism=user_ratings['scenario_realism'],
            confidence_level=user_ratings['confidence_level'],
            
            # Optional qualitative feedback
            what_worked_well=qualitative_feedback.get('what_worked_well', '') if qualitative_feedback else '',
            what_was_confusing=qualitative_feedback.get('what_was_confusing', '') if qualitative_feedback else '',
            suggested_improvements=qualitative_feedback.get('suggested_improvements', '') if qualitative_feedback else '',
            additional_comments=qualitative_feedback.get('additional_comments', '') if qualitative_feedback else '',
            
            # Context
            training_level=user_ratings.get('training_level', 'intermediate'),
            previous_experience=user_ratings.get('previous_experience', '')
        )
        
        # Store in database
        feedback_id = self.feedback_db.store_feedback(feedback)
        
        logger.info(f"✅ Collected feedback: {feedback_id}")
        return feedback_id
    
    def analyze_recent_performance(self, days: int = 7) -> Dict:
        """Analyze recent training performance using feedback data"""
        recent_feedback = self.feedback_db.get_recent_feedback(days=days)
        
        if not recent_feedback:
            return {'message': 'No recent feedback available'}
        
        # Calculate performance metrics
        total_episodes = len(recent_feedback)
        avg_success_rate = sum(f['success_rate'] for f in recent_feedback) / total_episodes
        avg_reward = sum(f['final_reward'] for f in recent_feedback) / total_episodes
        avg_episode_length = sum(f['episode_length'] for f in recent_feedback) / total_episodes
        
        # Analyze by scenario type
        scenario_performance = {}
        for feedback in recent_feedback:
            scenario_type = feedback['scenario_type']
            if scenario_type not in scenario_performance:
                scenario_performance[scenario_type] = {
                    'episodes': 0, 'total_success': 0.0, 'total_reward': 0.0
                }
            
            scenario_performance[scenario_type]['episodes'] += 1
            scenario_performance[scenario_type]['total_success'] += feedback['success_rate']
            scenario_performance[scenario_type]['total_reward'] += feedback['final_reward']
        
        # Calculate averages for each scenario type
        for scenario_type in scenario_performance:
            data = scenario_performance[scenario_type]
            episodes = data['episodes']
            data['avg_success_rate'] = data['total_success'] / episodes
            data['avg_reward'] = data['total_reward'] / episodes
            del data['total_success']
            del data['total_reward']
        
        return {
            'overall_performance': {
                'total_episodes': total_episodes,
                'avg_success_rate': round(avg_success_rate, 3),
                'avg_reward': round(avg_reward, 2),
                'avg_episode_length': round(avg_episode_length, 1)
            },
            'scenario_performance': scenario_performance,
            'period_days': days
        }
    
    def should_retrain_model(self) -> Tuple[bool, Dict]:
        """Check if model should be retrained based on feedback"""
        recent_feedback = self.feedback_db.get_recent_feedback(days=14)
        
        if len(recent_feedback) < 20:
            return False, {'reason': 'Insufficient feedback data', 'count': len(recent_feedback)}
        
        # Check performance trends
        avg_success = sum(f['success_rate'] for f in recent_feedback) / len(recent_feedback)
        avg_user_satisfaction = sum(
            (f['ai_helpfulness'] + f['confidence_level']) / 2 
            for f in recent_feedback
        ) / len(recent_feedback)
        
        # Retrain if performance is below thresholds
        should_retrain = avg_success < 0.7 or avg_user_satisfaction < 3.0
        
        return should_retrain, {
            'avg_success_rate': avg_success,
            'avg_user_satisfaction': avg_user_satisfaction,
            'feedback_count': len(recent_feedback),
            'success_threshold': 0.7,
            'satisfaction_threshold': 3.0
        }
    
    def retrain_from_feedback(self, 
                            min_agreement: int = 4,
                            training_epochs: int = 10) -> Dict:
        """Retrain the model using high-quality feedback data"""
        
        # Get training data from feedback
        training_data = self.analyzer.generate_training_data(min_agreement)
        
        if not training_data['training_examples']:
            raise ValueError("No high-quality training examples available")
        
        logger.info(f"Starting retraining with {len(training_data['training_examples'])} examples")
        
        # Prepare tensors
        states = []
        actions = []
        target_q_values = []
        
        for example in training_data['training_examples']:
            states.append(example['state'])
            actions.append(example['action'])
            # Use reward as target Q-value (simplified)
            target_q_values.append(example['reward'])
        
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        targets_tensor = torch.tensor(target_q_values, dtype=torch.float32)
        
        # Fine-tune the model
        initial_loss = 0.0
        final_loss = 0.0
        
        for epoch in range(training_epochs):
            # Create batch data
            batch_size = min(32, len(states))
            indices = torch.randperm(len(states))[:batch_size]
            
            batch_states = states_tensor[indices]
            batch_actions = actions_tensor[indices]
            batch_targets = targets_tensor[indices]
            
            # Forward pass with source awareness
            source_ids = torch.zeros(batch_size, dtype=torch.long)
            q_values = self.agent.q_network(batch_states, source_ids)
            
            # Get Q-values for taken actions
            action_q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(action_q_values, batch_targets)
            
            if epoch == 0:
                initial_loss = loss.item()
            
            # Backward pass
            self.agent.optimizer.zero_grad()
            loss.backward()
            self.agent.optimizer.step()
            
            final_loss = loss.item()
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        logger.info(f"✅ Retraining completed: Loss {initial_loss:.4f} → {final_loss:.4f}")
        
        return {
            'success': True,
            'training_examples': len(training_data['training_examples']),
            'expert_examples': len(training_data['expert_corrections']),
            'epochs': training_epochs,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': initial_loss - final_loss
        }
    
    def get_improvement_recommendations(self) -> Dict:
        """Get specific recommendations for model improvement"""
        return self.analyzer.identify_improvement_areas()


def run_feedback_enabled_training(episodes: int = 100, 
                                user_id: str = "demo_user") -> Dict:
    """Run a training session with feedback collection"""
    
    # Initialize components
    agent = EnhancedDQNAgent(state_dim=20, action_dim=8)
    environment = EnhancedFireResponseEnvironment()
    trainer = FeedbackEnabledTrainer(agent, environment)
    
    print(f"🚀 Starting feedback-enabled training: {episodes} episodes")
    
    # Start session
    session_id = trainer.start_feedback_session(user_id=user_id)
    
    results = []
    
    for episode in range(episodes):
        # Train episode
        episode_results = trainer.train_episode_with_feedback()
        
        # Simulate user feedback (in practice, this would come from UI)
        user_ratings = {
            'difficulty_rating': 3,  # 1-5 scale
            'ai_helpfulness': 4,
            'scenario_realism': 4,
            'confidence_level': 3,
            'training_level': 'intermediate'
        }
        
        # Collect feedback every 10 episodes
        if episode % 10 == 0:
            feedback_id = trainer.collect_episode_feedback(
                user_id=user_id,
                episode_results=episode_results,
                user_ratings=user_ratings,
                qualitative_feedback={
                    'what_worked_well': 'AI responses were logical',
                    'what_was_confusing': 'Some actions seemed suboptimal',
                    'suggested_improvements': 'Better handling of complex scenarios'
                }
            )
            print(f"📊 Episode {episode}: Feedback collected (ID: {feedback_id})")
        
        results.append(episode_results)
        
        # Check for retraining every 20 episodes
        if episode > 0 and episode % 20 == 0:
            should_retrain, metrics = trainer.should_retrain_model()
            if should_retrain:
                print(f"🔄 Triggering retraining at episode {episode}")
                try:
                    retrain_results = trainer.retrain_from_feedback()
                    print(f"✅ Retraining completed: {retrain_results}")
                except ValueError as e:
                    print(f"⚠️  Retraining skipped: {e}")
        
        if episode % 25 == 0:
            performance = trainer.analyze_recent_performance()
            print(f"📈 Episode {episode} performance: {performance}")
    
    # Final analysis
    final_performance = trainer.analyze_recent_performance(days=30)
    improvement_areas = trainer.get_improvement_recommendations()
    
    return {
        'session_id': session_id,
        'episodes_completed': episodes,
        'final_performance': final_performance,
        'improvement_recommendations': improvement_areas,
        'total_results': len(results)
    }


if __name__ == "__main__":
    # Run a demo training session
    results = run_feedback_enabled_training(episodes=50)
    print("\n🎯 Training Session Complete!")
    print(f"Session ID: {results['session_id']}")
    print(f"Episodes: {results['episodes_completed']}")
    print("Final Performance:", results['final_performance'])
