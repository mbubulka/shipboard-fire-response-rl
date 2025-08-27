#!/usr/bin/env python3
"""
Shipboard Fire Response RL Feedback Collection System
Comprehensive feedback mechanism for continuous learning and model improvement
"""

import json
import sqlite3
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLFeedbackData:
    """Structure for collecting user feedback on RL training assessments"""
    session_id: str
    user_id: str
    scenario_id: str
    scenario_type: str  # galley_fire, engine_room_fire, etc.
    scenario_source: str  # nfpa_1500, uscg_guidelines, maritime_standards
    
    # RL-specific metrics
    actions_taken: List[int]  # Sequence of actions selected
    q_values: List[List[float]]  # Q-values for each action at each step
    episode_rewards: List[float]  # Reward at each step
    episode_length: int
    final_reward: float
    success_rate: float  # 0.0-1.0
    
    # User feedback ratings (1-5 scale)
    difficulty_rating: int
    ai_helpfulness: int
    scenario_realism: int
    confidence_level: int
    
    # Qualitative feedback
    what_worked_well: str = ""
    what_was_confusing: str = ""
    suggested_improvements: str = ""
    additional_comments: str = ""
    
    # Expert review (optional)
    expert_review: bool = False
    expert_score: Optional[float] = None
    expert_corrections: Optional[List[int]] = None  # Corrected action sequence
    
    # Context
    timestamp: str = None
    training_level: str = "intermediate"  # novice, intermediate, advanced, expert
    previous_experience: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RLFeedbackDatabase:
    """Database manager for RL feedback collection and analysis"""
    
    def __init__(self, db_path: str = "rl_feedback.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the feedback database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                scenario_type TEXT NOT NULL,
                scenario_source TEXT NOT NULL,
                
                actions_taken TEXT NOT NULL,  -- JSON array
                q_values TEXT NOT NULL,       -- JSON array of arrays
                episode_rewards TEXT NOT NULL,  -- JSON array
                episode_length INTEGER NOT NULL,
                final_reward REAL NOT NULL,
                success_rate REAL NOT NULL,
                
                difficulty_rating INTEGER NOT NULL,
                ai_helpfulness INTEGER NOT NULL,
                scenario_realism INTEGER NOT NULL,
                confidence_level INTEGER NOT NULL,
                
                what_worked_well TEXT,
                what_was_confusing TEXT,
                suggested_improvements TEXT,
                additional_comments TEXT,
                
                expert_review BOOLEAN DEFAULT FALSE,
                expert_score REAL,
                expert_corrections TEXT,  -- JSON array
                
                timestamp TEXT NOT NULL,
                training_level TEXT NOT NULL,
                previous_experience TEXT,
                
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Action-level feedback for detailed analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                state_vector TEXT NOT NULL,  -- JSON array
                action_taken INTEGER NOT NULL,
                q_values TEXT NOT NULL,      -- JSON array
                reward_received REAL NOT NULL,
                user_agreement INTEGER,     -- 1-5 scale: how much user agrees with action
                alternative_action INTEGER, -- What action user would have chosen
                feedback_text TEXT,
                timestamp TEXT NOT NULL,
                
                FOREIGN KEY (session_id) REFERENCES rl_feedback (session_id)
            )
        """)
        
        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                test_scenario_id TEXT NOT NULL,
                success_rate REAL NOT NULL,
                average_reward REAL NOT NULL,
                episode_length REAL NOT NULL,
                evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("✅ RL Feedback database initialized")
    
    def store_feedback(self, feedback: RLFeedbackData) -> int:
        """Store feedback data in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rl_feedback (
                session_id, user_id, scenario_id, scenario_type, scenario_source,
                actions_taken, q_values, episode_rewards, episode_length, 
                final_reward, success_rate,
                difficulty_rating, ai_helpfulness, scenario_realism, confidence_level,
                what_worked_well, what_was_confusing, suggested_improvements, 
                additional_comments, expert_review, expert_score, expert_corrections,
                timestamp, training_level, previous_experience
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.session_id, feedback.user_id, feedback.scenario_id,
            feedback.scenario_type, feedback.scenario_source,
            json.dumps(feedback.actions_taken),
            json.dumps(feedback.q_values),
            json.dumps(feedback.episode_rewards),
            feedback.episode_length, feedback.final_reward, feedback.success_rate,
            feedback.difficulty_rating, feedback.ai_helpfulness, 
            feedback.scenario_realism, feedback.confidence_level,
            feedback.what_worked_well, feedback.what_was_confusing,
            feedback.suggested_improvements, feedback.additional_comments,
            feedback.expert_review, feedback.expert_score,
            json.dumps(feedback.expert_corrections) if feedback.expert_corrections else None,
            feedback.timestamp, feedback.training_level, feedback.previous_experience
        ))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Stored RL feedback: {feedback_id}")
        return feedback_id
    
    def store_action_feedback(self, session_id: str, step_number: int, 
                            state_vector: List[float], action_taken: int,
                            q_values: List[float], reward_received: float,
                            user_agreement: int = None, alternative_action: int = None,
                            feedback_text: str = "") -> int:
        """Store detailed feedback for individual actions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO action_feedback (
                session_id, step_number, state_vector, action_taken,
                q_values, reward_received, user_agreement, 
                alternative_action, feedback_text, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, step_number, json.dumps(state_vector), action_taken,
            json.dumps(q_values), reward_received, user_agreement,
            alternative_action, feedback_text, datetime.now().isoformat()
        ))
        
        action_feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return action_feedback_id
    
    def get_recent_feedback(self, days: int = 7) -> List[Dict]:
        """Get recent feedback for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM rl_feedback 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        """, (cutoff_date,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_performance_by_scenario_type(self, scenario_type: str) -> Dict:
        """Get performance analytics for a specific scenario type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_sessions,
                AVG(final_reward) as avg_reward,
                AVG(success_rate) as avg_success_rate,
                AVG(episode_length) as avg_episode_length,
                AVG(difficulty_rating) as avg_difficulty,
                AVG(ai_helpfulness) as avg_ai_helpfulness,
                AVG(scenario_realism) as avg_realism,
                AVG(confidence_level) as avg_confidence
            FROM rl_feedback 
            WHERE scenario_type = ?
        """, (scenario_type,))
        
        result = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        if result:
            return dict(zip(columns, result))
        return {}


class RLFeedbackAnalyzer:
    """Analyzes RL feedback data to identify improvement opportunities"""
    
    def __init__(self, database: RLFeedbackDatabase):
        self.db = database
    
    def analyze_action_patterns(self) -> Dict:
        """Analyze patterns in action selection and user feedback"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get action-level feedback with disagreements
        cursor.execute("""
            SELECT action_taken, user_agreement, alternative_action, 
                   COUNT(*) as frequency
            FROM action_feedback 
            WHERE user_agreement IS NOT NULL
            GROUP BY action_taken, user_agreement, alternative_action
            ORDER BY frequency DESC
        """)
        
        action_patterns = []
        for row in cursor.fetchall():
            action_patterns.append({
                'action_taken': row[0],
                'user_agreement': row[1],
                'alternative_action': row[2],
                'frequency': row[3]
            })
        
        conn.close()
        return {'action_patterns': action_patterns}
    
    def identify_improvement_areas(self) -> Dict:
        """Identify areas where the model needs improvement"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Find scenarios with low success rates
        cursor.execute("""
            SELECT scenario_type, scenario_source,
                   AVG(success_rate) as avg_success,
                   AVG(final_reward) as avg_reward,
                   COUNT(*) as session_count
            FROM rl_feedback
            GROUP BY scenario_type, scenario_source
            HAVING COUNT(*) >= 3
            ORDER BY avg_success ASC
        """)
        
        low_performance_scenarios = []
        for row in cursor.fetchall():
            low_performance_scenarios.append({
                'scenario_type': row[0],
                'scenario_source': row[1],
                'avg_success_rate': row[2],
                'avg_reward': row[3],
                'session_count': row[4]
            })
        
        # Find actions with frequent disagreements
        cursor.execute("""
            SELECT action_taken, 
                   AVG(user_agreement) as avg_agreement,
                   COUNT(*) as feedback_count
            FROM action_feedback
            WHERE user_agreement IS NOT NULL
            GROUP BY action_taken
            HAVING COUNT(*) >= 5
            ORDER BY avg_agreement ASC
        """)
        
        problematic_actions = []
        for row in cursor.fetchall():
            problematic_actions.append({
                'action': row[0],
                'avg_agreement': row[1],
                'feedback_count': row[2]
            })
        
        conn.close()
        
        return {
            'low_performance_scenarios': low_performance_scenarios[:5],
            'problematic_actions': problematic_actions[:10]
        }
    
    def generate_training_data(self, min_agreement: int = 4) -> Dict:
        """Generate training data from high-quality feedback"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get high-quality action examples
        cursor.execute("""
            SELECT af.state_vector, af.action_taken, af.reward_received
            FROM action_feedback af
            JOIN rl_feedback rf ON af.session_id = rf.session_id
            WHERE af.user_agreement >= ? AND rf.success_rate >= 0.8
        """, (min_agreement,))
        
        training_examples = []
        for row in cursor.fetchall():
            state = json.loads(row[0])
            action = row[1]
            reward = row[2]
            
            training_examples.append({
                'state': state,
                'action': action,
                'reward': reward
            })
        
        # Get expert corrections
        cursor.execute("""
            SELECT actions_taken, expert_corrections, final_reward
            FROM rl_feedback
            WHERE expert_review = TRUE AND expert_corrections IS NOT NULL
        """)
        
        expert_corrections = []
        for row in cursor.fetchall():
            original_actions = json.loads(row[0])
            corrected_actions = json.loads(row[1])
            reward = row[2]
            
            expert_corrections.append({
                'original_actions': original_actions,
                'corrected_actions': corrected_actions,
                'reward': reward
            })
        
        conn.close()
        
        return {
            'training_examples': training_examples,
            'expert_corrections': expert_corrections,
            'total_examples': len(training_examples),
            'expert_examples': len(expert_corrections)
        }


class RLFeedbackIntegration:
    """Integrates feedback system with Enhanced DQN training"""
    
    def __init__(self, feedback_db: RLFeedbackDatabase, 
                 analyzer: RLFeedbackAnalyzer):
        self.feedback_db = feedback_db
        self.analyzer = analyzer
    
    def should_trigger_retraining(self, threshold_sessions: int = 50) -> bool:
        """Check if enough feedback has been collected to trigger retraining"""
        recent_feedback = self.feedback_db.get_recent_feedback(days=14)
        return len(recent_feedback) >= threshold_sessions
    
    def prepare_retraining_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare feedback data for model retraining"""
        training_data = self.analyzer.generate_training_data()
        
        if not training_data['training_examples']:
            raise ValueError("No training examples available")
        
        # Convert to tensors
        states = []
        actions = []
        rewards = []
        
        for example in training_data['training_examples']:
            states.append(example['state'])
            actions.append(example['action'])
            rewards.append(example['reward'])
        
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        logger.info(f"✅ Prepared {len(states)} training examples from feedback")
        
        return states_tensor, actions_tensor, rewards_tensor
    
    def update_model_performance(self, model_version: str, test_results: Dict):
        """Track model performance after retraining"""
        conn = sqlite3.connect(self.feedback_db.db_path)
        cursor = conn.cursor()
        
        for scenario_id, results in test_results.items():
            cursor.execute("""
                INSERT INTO model_performance 
                (model_version, test_scenario_id, success_rate, average_reward, episode_length)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_version, scenario_id, 
                results['success_rate'], results['average_reward'], 
                results['episode_length']
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Updated performance metrics for model {model_version}")


if __name__ == "__main__":
    # Example usage and testing
    print("🔄 Shipboard Fire Response RL Feedback System")
    print("=" * 60)
    
    # Initialize components
    db = RLFeedbackDatabase()
    analyzer = RLFeedbackAnalyzer(db)
    integration = RLFeedbackIntegration(db, analyzer)
    
    print("✅ RL Feedback system initialized")
    print("📊 Database tables created")
    print("🔍 Analytics engine ready")
    print("🤖 Model integration prepared")
    print("\n🎯 System ready to collect and analyze RL feedback!")
