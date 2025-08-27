# 🔄 Shipboard Fire Response RL Feedback System

## Overview

The feedback system provides a comprehensive framework for collecting user feedback, analyzing performance patterns, and continuously improving the Enhanced DQN model through real-world usage data.

## 🏗️ System Architecture

### Core Components

1. **RLFeedbackData** - Structured data collection for RL episodes
2. **RLFeedbackDatabase** - SQLite storage with optimized schema 
3. **RLFeedbackAnalyzer** - Pattern analysis and improvement identification
4. **FeedbackEnabledTrainer** - Integrated training with feedback collection
5. **Feedback API** - REST endpoints for web integration

## 🚀 Quick Start

### Basic Usage

```python
from shipboard_rl.feedback import (
    RLFeedbackDatabase, 
    RLFeedbackAnalyzer,
    FeedbackEnabledTrainer
)
from shipboard_rl.core.enhanced_dqn import EnhancedDQNAgent, EnhancedFireResponseEnvironment

# Initialize components
agent = EnhancedDQNAgent(state_dim=20, action_dim=8)
environment = EnhancedFireResponseEnvironment()
trainer = FeedbackEnabledTrainer(agent, environment)

# Start feedback session
session_id = trainer.start_feedback_session(user_id="user123")

# Train episode with feedback collection
episode_results = trainer.train_episode_with_feedback()

# Collect user feedback
user_ratings = {
    'difficulty_rating': 3,    # 1-5 scale
    'ai_helpfulness': 4,       # 1-5 scale  
    'scenario_realism': 4,     # 1-5 scale
    'confidence_level': 3      # 1-5 scale
}

feedback_id = trainer.collect_episode_feedback(
    user_id="user123",
    episode_results=episode_results,
    user_ratings=user_ratings
)
```

### Web API Usage

Start the feedback API server:

```bash
python -m shipboard_rl.feedback.feedback_api
```

API endpoints available at `http://localhost:5000`:

- `POST /api/rl-feedback/session/start` - Start feedback session
- `POST /api/rl-feedback/submit` - Submit episode feedback
- `GET /api/rl-feedback/analytics/summary` - Get performance analytics
- `GET /feedback/form` - Web-based feedback form

### Demo Script

Run the comprehensive demo:

```bash
python demo_feedback_system.py
```

## 📊 Data Collection Framework

### Episode-Level Data
- **Actions Taken**: Sequence of actions selected by the agent
- **Q-Values**: Q-value distributions at each decision point
- **Rewards**: Step-by-step and cumulative rewards
- **Success Metrics**: Episode length, final reward, success rate

### User Feedback Data
- **Quantitative Ratings**: Difficulty, AI helpfulness, scenario realism, confidence (1-5 scale)
- **Qualitative Feedback**: What worked well, confusing aspects, improvement suggestions
- **Context Information**: Training level, previous experience, scenario details

### Action-Level Data
- **State Vectors**: Environmental state at each step
- **User Agreement**: How much user agrees with AI action choice (1-5 scale)
- **Alternative Actions**: What action user would have chosen instead
- **Detailed Comments**: Step-specific feedback text

## 🔍 Analytics & Insights

### Performance Analysis
```python
# Analyze recent performance
performance = trainer.analyze_recent_performance(days=7)
print(f"Average success rate: {performance['overall_performance']['avg_success_rate']}")
print(f"Average reward: {performance['overall_performance']['avg_reward']}")

# Get improvement recommendations
improvements = trainer.get_improvement_recommendations()
print(f"Low performance scenarios: {improvements['low_performance_scenarios']}")
print(f"Problematic actions: {improvements['problematic_actions']}")
```

### Automated Retraining
```python
# Check if retraining is recommended
should_retrain, metrics = trainer.should_retrain_model()

if should_retrain:
    # Retrain model using feedback data
    results = trainer.retrain_from_feedback(
        min_agreement=4,      # Use high-quality feedback only
        training_epochs=10    # Number of fine-tuning epochs
    )
    print(f"Retraining completed: {results}")
```

## 🌐 Web Integration

### Starting a Feedback Session (JavaScript)
```javascript
// Start feedback session
const response = await fetch('/api/rl-feedback/session/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        user_id: 'user123',
        scenario_id: 'galley_fire_001',
        scenario_type: 'galley_fire',
        scenario_source: 'nfpa_1500'
    })
});

const data = await response.json();
const sessionId = data.session_id;
```

### Logging Actions During Training
```javascript
// Log each action taken during episode
await fetch('/api/rl-feedback/session/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        session_id: sessionId,
        action: selectedAction,
        q_values: qValueArray,
        reward: receivedReward,
        state: currentState
    })
});
```

### Submitting Final Feedback
```javascript
// Submit comprehensive episode feedback
await fetch('/api/rl-feedback/submit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        session_id: sessionId,
        final_reward: episodeReward,
        success_rate: successRate,
        difficulty_rating: 3,
        ai_helpfulness: 4,
        scenario_realism: 4,
        confidence_level: 3,
        what_worked_well: "Logical action progression",
        what_was_confusing: "Some actions seemed suboptimal",
        suggested_improvements: "Better emergency handling"
    })
});
```

## 🔧 Configuration Options

### Database Configuration
```python
# Custom database location
db = RLFeedbackDatabase(db_path="custom_feedback.db")

# Memory-only database for testing
db = RLFeedbackDatabase(db_path=":memory:")
```

### Retraining Thresholds
```python
# Custom retraining criteria
trainer.should_retrain_model()  # Uses defaults:
# - Minimum 20 feedback sessions in 14 days
# - Average success rate < 0.7
# - Average user satisfaction < 3.0

# Custom thresholds in implementation
should_retrain = (
    len(recent_feedback) >= 50 and  # More feedback required
    avg_success < 0.8 and           # Higher success threshold
    avg_satisfaction < 3.5          # Higher satisfaction threshold
)
```

### Training Parameters
```python
# Fine-tune retraining parameters
results = trainer.retrain_from_feedback(
    min_agreement=4,        # Only use high-agreement feedback
    training_epochs=20,     # More epochs for better convergence
)
```

## 📈 Expected Outcomes

### Model Improvement Metrics
- **Performance Enhancement**: 10-15% improvement in success rates after retraining
- **User Satisfaction**: Increased AI helpfulness ratings over time
- **Scenario Coverage**: Better performance across diverse fire response scenarios
- **Expert Alignment**: Higher agreement between AI actions and expert corrections

### Feedback Quality Indicators
- **Response Rate**: Target 80%+ feedback submission rate
- **Data Completeness**: All required ratings provided
- **Qualitative Insights**: Actionable improvement suggestions
- **Expert Validation**: Regular expert review of AI recommendations

## 🔒 Privacy & Security

### Data Protection
- **User Anonymization**: Optional anonymous feedback collection
- **Data Encryption**: SQLite database encryption support
- **Access Control**: API authentication for production use
- **Data Retention**: Configurable feedback data lifecycle

### Compliance Considerations
- **GDPR Compliance**: User consent and data deletion capabilities
- **Training Standards**: Alignment with NFPA, USCG, and Maritime guidelines
- **Audit Trail**: Complete tracking of feedback and model changes

## 🛠️ Installation & Dependencies

### Required Dependencies
```bash
pip install torch numpy sqlite3 flask flask-cors
```

### Optional Dependencies
```bash
pip install pandas matplotlib seaborn  # For advanced analytics
```

### Development Setup
```bash
git clone https://github.com/mbubulka/shipboard-fire-response-rl.git
cd shipboard-fire-response-rl
pip install -e .
```

## 📚 Examples & Use Cases

### Training Enhancement
Use feedback to improve training on specific scenario types:

```python
# Focus on problematic scenarios
galley_performance = db.get_performance_by_scenario_type("galley_fire")
if galley_performance['avg_success_rate'] < 0.7:
    # Generate additional training data for galley fires
    training_data = analyzer.generate_training_data(min_agreement=4)
    # Retrain with focus on this scenario type
```

### Expert Review Integration
Incorporate expert corrections into training:

```python
# Add expert review to feedback
feedback = RLFeedbackData(
    # ... other fields ...
    expert_review=True,
    expert_score=8.5,
    expert_corrections=[0, 1, 3, 2, 4]  # Corrected action sequence
)
```

### Real-time Adaptation
Continuous model improvement during operation:

```python
# Monitor feedback and adapt
while training_active:
    episode_results = trainer.train_episode_with_feedback()
    
    if episode % 10 == 0:
        should_retrain, _ = trainer.should_retrain_model()
        if should_retrain:
            trainer.retrain_from_feedback()
```

## 🎯 Best Practices

### Feedback Collection
1. **Timing**: Collect feedback immediately after episode completion
2. **Completeness**: Ensure all required ratings are provided
3. **Context**: Include relevant scenario and user information
4. **Balance**: Mix quantitative ratings with qualitative insights

### Model Improvement
1. **Quality Gates**: Only use high-agreement feedback for retraining
2. **Validation**: Test retrained models on held-out scenarios
3. **Incremental**: Make gradual improvements rather than large changes
4. **Monitoring**: Track model performance before and after updates

### Production Deployment
1. **Scalability**: Use proper database systems for high-volume feedback
2. **Monitoring**: Implement logging and alerting for the feedback system
3. **Backup**: Regular backup of feedback data and model checkpoints
4. **Testing**: Thorough testing of retraining pipeline before deployment

---

For more information, see the [main project README](../README.md) and [technical documentation](docs/).
