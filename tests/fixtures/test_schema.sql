-- Test database schema for shipboard RL system
-- This file is used by CI/CD pipeline for test database setup

CREATE TABLE IF NOT EXISTS feedback_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    scenario_type VARCHAR(100) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    status ENUM('active', 'completed', 'aborted') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback_actions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    action_data JSON NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    action_id INT NULL,
    rating INT CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    feedback_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (action_id) REFERENCES feedback_actions(id) ON DELETE SET NULL
);

-- Insert test data
INSERT IGNORE INTO feedback_sessions (session_id, user_id, scenario_type, status) VALUES
('test_session_1', 'test_user_1', 'dca_training', 'completed'),
('test_session_2', 'test_user_2', 'emergency_response', 'active');

INSERT IGNORE INTO feedback_actions (session_id, action_type, action_data) VALUES
('test_session_1', 'text_response', '{"user_action": "Deploy foam system", "ai_recommendation": "Use water first"}'),
('test_session_1', 'choice_selection', '{"selected": "option_a", "available": ["option_a", "option_b", "option_c"]}');
