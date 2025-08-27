#!/usr/bin/env python3
"""
Flask API for RL Feedback Collection and Analysis
Provides REST endpoints for the Shipboard Fire Response RL system
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import uuid
from datetime import datetime
from pathlib import Path
import logging

from .feedback_system import (
    RLFeedbackDatabase, 
    RLFeedbackAnalyzer, 
    RLFeedbackIntegration,
    RLFeedbackData
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize feedback system
feedback_db = RLFeedbackDatabase()
feedback_analyzer = RLFeedbackAnalyzer(feedback_db)
feedback_integration = RLFeedbackIntegration(feedback_db, feedback_analyzer)

# In-memory session storage (use Redis in production)
active_sessions = {}


@app.route('/api/rl-feedback/session/start', methods=['POST'])
def start_feedback_session():
    """Start a new RL feedback session"""
    try:
        data = request.get_json()
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'user_id': data.get('user_id', 'anonymous'),
            'scenario_id': data.get('scenario_id'),
            'scenario_type': data.get('scenario_type'),
            'scenario_source': data.get('scenario_source', 'nfpa_1500'),
            'started_at': datetime.now().isoformat(),
            'actions': [],
            'q_values': [],
            'rewards': [],
            'states': []
        }
        
        active_sessions[session_id] = session_data
        
        logger.info(f"✅ Started RL feedback session: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Feedback session started'
        })
        
    except Exception as e:
        logger.error(f"❌ Error starting session: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rl-feedback/session/action', methods=['POST'])
def log_action():
    """Log an action taken during the RL episode"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        
        # Store action data
        session['actions'].append(data.get('action'))
        session['q_values'].append(data.get('q_values', []))
        session['rewards'].append(data.get('reward', 0.0))
        session['states'].append(data.get('state', []))
        
        # Optional: Store detailed action feedback
        if data.get('user_agreement') is not None:
            feedback_db.store_action_feedback(
                session_id=session_id,
                step_number=len(session['actions']) - 1,
                state_vector=data.get('state', []),
                action_taken=data.get('action'),
                q_values=data.get('q_values', []),
                reward_received=data.get('reward', 0.0),
                user_agreement=data.get('user_agreement'),
                alternative_action=data.get('alternative_action'),
                feedback_text=data.get('feedback_text', '')
            )
        
        return jsonify({
            'success': True,
            'message': 'Action logged successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error logging action: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rl-feedback/submit', methods=['POST'])
def submit_feedback():
    """Submit comprehensive feedback for completed RL episode"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_sessions[session_id]
        
        # Validate required fields
        required_fields = [
            'difficulty_rating', 'ai_helpfulness', 
            'scenario_realism', 'confidence_level'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create feedback object
        feedback = RLFeedbackData(
            session_id=session_id,
            user_id=session.get('user_id', 'anonymous'),
            scenario_id=session.get('scenario_id'),
            scenario_type=session.get('scenario_type'),
            scenario_source=session.get('scenario_source'),
            
            # RL metrics from session
            actions_taken=session.get('actions', []),
            q_values=session.get('q_values', []),
            episode_rewards=session.get('rewards', []),
            episode_length=len(session.get('actions', [])),
            final_reward=data.get('final_reward', 0.0),
            success_rate=data.get('success_rate', 0.0),
            
            # User ratings
            difficulty_rating=data['difficulty_rating'],
            ai_helpfulness=data['ai_helpfulness'],
            scenario_realism=data['scenario_realism'],
            confidence_level=data['confidence_level'],
            
            # Qualitative feedback
            what_worked_well=data.get('what_worked_well', ''),
            what_was_confusing=data.get('what_was_confusing', ''),
            suggested_improvements=data.get('suggested_improvements', ''),
            additional_comments=data.get('additional_comments', ''),
            
            # Expert review
            expert_review=data.get('expert_review', False),
            expert_score=data.get('expert_score'),
            expert_corrections=data.get('expert_corrections'),
            
            # Context
            training_level=data.get('training_level', 'intermediate'),
            previous_experience=data.get('previous_experience', '')
        )
        
        # Store feedback
        feedback_id = feedback_db.store_feedback(feedback)
        
        # Clean up session
        del active_sessions[session_id]
        
        # Check if retraining should be triggered
        should_retrain = feedback_integration.should_trigger_retraining()
        
        response = {
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Thank you for your feedback!',
            'will_improve_model': True
        }
        
        if should_retrain:
            response['retraining_triggered'] = True
            response['message'] += ' Your feedback will help retrain the model.'
        
        logger.info(f"✅ Feedback submitted: {feedback_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rl-feedback/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get summary analytics of recent feedback"""
    try:
        days = request.args.get('days', 7, type=int)
        recent_feedback = feedback_db.get_recent_feedback(days=days)
        
        if not recent_feedback:
            return jsonify({
                'message': 'No recent feedback available',
                'total_sessions': 0
            })
        
        # Calculate summary statistics
        total_sessions = len(recent_feedback)
        avg_success_rate = sum(f['success_rate'] for f in recent_feedback) / total_sessions
        avg_reward = sum(f['final_reward'] for f in recent_feedback) / total_sessions
        avg_episode_length = sum(f['episode_length'] for f in recent_feedback) / total_sessions
        
        # Rating averages
        avg_difficulty = sum(f['difficulty_rating'] for f in recent_feedback) / total_sessions
        avg_helpfulness = sum(f['ai_helpfulness'] for f in recent_feedback) / total_sessions
        avg_realism = sum(f['scenario_realism'] for f in recent_feedback) / total_sessions
        avg_confidence = sum(f['confidence_level'] for f in recent_feedback) / total_sessions
        
        # Scenario type breakdown
        scenario_types = {}
        for feedback in recent_feedback:
            scenario_type = feedback['scenario_type']
            if scenario_type not in scenario_types:
                scenario_types[scenario_type] = {'count': 0, 'avg_success': 0}
            scenario_types[scenario_type]['count'] += 1
            scenario_types[scenario_type]['avg_success'] += feedback['success_rate']
        
        # Calculate averages for scenario types
        for scenario_type in scenario_types:
            count = scenario_types[scenario_type]['count']
            scenario_types[scenario_type]['avg_success'] /= count
        
        return jsonify({
            'summary': {
                'total_sessions': total_sessions,
                'avg_success_rate': round(avg_success_rate, 3),
                'avg_final_reward': round(avg_reward, 3),
                'avg_episode_length': round(avg_episode_length, 1),
                'avg_difficulty_rating': round(avg_difficulty, 2),
                'avg_ai_helpfulness': round(avg_helpfulness, 2),
                'avg_scenario_realism': round(avg_realism, 2),
                'avg_confidence_level': round(avg_confidence, 2)
            },
            'scenario_breakdown': scenario_types,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting analytics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rl-feedback/improvement-areas', methods=['GET'])
def get_improvement_areas():
    """Get areas where the model needs improvement"""
    try:
        improvement_data = feedback_analyzer.identify_improvement_areas()
        action_patterns = feedback_analyzer.analyze_action_patterns()
        
        return jsonify({
            'improvement_areas': improvement_data,
            'action_patterns': action_patterns,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error analyzing improvements: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rl-feedback/trigger-retraining', methods=['POST'])
def trigger_retraining():
    """Manually trigger model retraining based on recent feedback"""
    try:
        # Check if enough feedback is available
        recent_feedback = feedback_db.get_recent_feedback(days=30)
        
        if len(recent_feedback) < 10:
            return jsonify({
                'success': False,
                'error': 'Insufficient feedback data for retraining',
                'feedback_count': len(recent_feedback),
                'minimum_required': 10
            }), 400
        
        # Prepare training data
        try:
            training_data = feedback_integration.prepare_retraining_data()
            states, actions, rewards = training_data
            
            # In production, this would queue a background job
            job_id = str(uuid.uuid4())
            
            logger.info(f"✅ Retraining queued: {job_id}")
            
            return jsonify({
                'success': True,
                'message': 'Model retraining queued successfully',
                'job_id': job_id,
                'feedback_count': len(recent_feedback),
                'training_examples': len(states),
                'estimated_completion': '2-4 hours'
            })
            
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Training data preparation failed: {str(e)}'
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Error triggering retraining: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/rl-feedback/scenario-performance/<scenario_type>', methods=['GET'])
def get_scenario_performance(scenario_type):
    """Get performance analytics for a specific scenario type"""
    try:
        performance_data = feedback_db.get_performance_by_scenario_type(scenario_type)
        
        if not performance_data:
            return jsonify({
                'message': f'No data available for scenario type: {scenario_type}',
                'scenario_type': scenario_type
            }), 404
        
        return jsonify({
            'scenario_type': scenario_type,
            'performance': performance_data,
            'retrieved_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting scenario performance: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/feedback/form')
def feedback_form():
    """Serve a simple feedback collection form"""
    form_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>RL Training Feedback</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .rating { display: flex; gap: 5px; }
            .rating input { width: auto; }
        </style>
    </head>
    <body>
        <h1>🤖 RL Training Feedback</h1>
        <form id="feedbackForm">
            <div class="form-group">
                <label>Session ID:</label>
                <input type="text" name="session_id" required placeholder="Enter session ID">
            </div>
            
            <div class="form-group">
                <label>Difficulty Rating (1-5):</label>
                <div class="rating">
                    <input type="radio" name="difficulty_rating" value="1" required> 1
                    <input type="radio" name="difficulty_rating" value="2"> 2
                    <input type="radio" name="difficulty_rating" value="3"> 3
                    <input type="radio" name="difficulty_rating" value="4"> 4
                    <input type="radio" name="difficulty_rating" value="5"> 5
                </div>
            </div>
            
            <div class="form-group">
                <label>AI Helpfulness (1-5):</label>
                <div class="rating">
                    <input type="radio" name="ai_helpfulness" value="1" required> 1
                    <input type="radio" name="ai_helpfulness" value="2"> 2
                    <input type="radio" name="ai_helpfulness" value="3"> 3
                    <input type="radio" name="ai_helpfulness" value="4"> 4
                    <input type="radio" name="ai_helpfulness" value="5"> 5
                </div>
            </div>
            
            <div class="form-group">
                <label>Scenario Realism (1-5):</label>
                <div class="rating">
                    <input type="radio" name="scenario_realism" value="1" required> 1
                    <input type="radio" name="scenario_realism" value="2"> 2
                    <input type="radio" name="scenario_realism" value="3"> 3
                    <input type="radio" name="scenario_realism" value="4"> 4
                    <input type="radio" name="scenario_realism" value="5"> 5
                </div>
            </div>
            
            <div class="form-group">
                <label>Confidence Level (1-5):</label>
                <div class="rating">
                    <input type="radio" name="confidence_level" value="1" required> 1
                    <input type="radio" name="confidence_level" value="2"> 2
                    <input type="radio" name="confidence_level" value="3"> 3
                    <input type="radio" name="confidence_level" value="4"> 4
                    <input type="radio" name="confidence_level" value="5"> 5
                </div>
            </div>
            
            <div class="form-group">
                <label>What worked well?</label>
                <textarea name="what_worked_well" rows="3"></textarea>
            </div>
            
            <div class="form-group">
                <label>What was confusing?</label>
                <textarea name="what_was_confusing" rows="3"></textarea>
            </div>
            
            <div class="form-group">
                <label>Suggested improvements:</label>
                <textarea name="suggested_improvements" rows="3"></textarea>
            </div>
            
            <button type="submit">Submit Feedback</button>
        </form>
        
        <script>
            document.getElementById('feedbackForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {};
                
                for (let [key, value] of formData.entries()) {
                    if (key.includes('rating') || key.includes('helpfulness') || key.includes('realism') || key.includes('confidence')) {
                        data[key] = parseInt(value);
                    } else {
                        data[key] = value;
                    }
                }
                
                try {
                    const response = await fetch('/api/rl-feedback/submit', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    if (result.success) {
                        alert('✅ Thank you for your feedback!');
                        e.target.reset();
                    } else {
                        alert('❌ Error: ' + result.error);
                    }
                } catch (error) {
                    alert('❌ Failed to submit feedback: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(form_html)


@app.route('/api/rl-feedback/status', methods=['GET'])
def get_system_status():
    """Get current status of the RL feedback system"""
    try:
        recent_feedback = feedback_db.get_recent_feedback(days=7)
        total_sessions = len(recent_feedback)
        
        # Check if retraining is recommended
        should_retrain = feedback_integration.should_trigger_retraining()
        
        return jsonify({
            'status': 'operational',
            'recent_sessions': total_sessions,
            'retraining_recommended': should_retrain,
            'active_sessions': len(active_sessions),
            'system_version': '1.0.0',
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting RL Feedback API Server")
    print("🔗 Available endpoints:")
    print("   POST /api/rl-feedback/session/start")
    print("   POST /api/rl-feedback/session/action") 
    print("   POST /api/rl-feedback/submit")
    print("   GET  /api/rl-feedback/analytics/summary")
    print("   GET  /api/rl-feedback/improvement-areas")
    print("   POST /api/rl-feedback/trigger-retraining")
    print("   GET  /feedback/form")
    print("   GET  /api/rl-feedback/status")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
