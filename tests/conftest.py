"""Test configuration and fixtures for Shipboard Fire Response AI System"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from shipboard_ai.config.settings import Config
from shipboard_ai.api.server import create_app


@pytest.fixture
def test_config():
    """Create a test configuration"""
    config = Config()
    config.TESTING = True
    config.DATABASE_URL = "sqlite:///:memory:"
    config.SECRET_KEY = "test-secret-key"
    config.WTF_CSRF_ENABLED = False
    return config


@pytest.fixture
def app(test_config):
    """Create a test Flask application"""
    app = create_app(test_config)
    return app


@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a test CLI runner"""
    return app.test_cli_runner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_database():
    """Mock database connection"""
    with patch('mysql.connector.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn, mock_cursor


@pytest.fixture
def sample_training_data():
    """Sample training data for testing"""
    return {
        "scenarios": [
            {
                "id": 1,
                "scenario_type": "galley_fire",
                "location": "galley",
                "fire_type": "cooking_oil",
                "severity": "moderate",
                "response_actions": [
                    "isolate_power",
                    "apply_wet_chemical",
                    "ventilate_space"
                ],
                "expected_outcome": "fire_suppressed",
                "training_standards": ["NFPA_1500", "USCG_CG022"]
            },
            {
                "id": 2,
                "scenario_type": "engine_room_fire",
                "location": "engine_room",
                "fire_type": "fuel_oil",
                "severity": "high",
                "response_actions": [
                    "emergency_shutdown",
                    "activate_co2_system",
                    "evacuate_personnel"
                ],
                "expected_outcome": "fire_suppressed",
                "training_standards": ["NFPA_1670", "Maritime_RVSS"]
            }
        ]
    }


@pytest.fixture
def mock_enhanced_dqn():
    """Mock Enhanced DQN model"""
    with patch('shipboard_ai.core.enhanced_dqn.EnhancedDQN') as mock_dqn:
        mock_model = Mock()
        mock_model.predict.return_value = [0.8, 0.2, 0.5, 0.9]
        mock_model.train_step.return_value = {"loss": 0.123, "accuracy": 0.87}
        mock_dqn.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_training_manager():
    """Mock Training Manager"""
    with patch('shipboard_ai.training.trainer.TrainingManager') as mock_trainer:
        mock_manager = Mock()
        mock_manager.train_model.return_value = {
            "epochs": 100,
            "final_loss": 0.05,
            "accuracy": 0.92,
            "training_time": "2h 15m"
        }
        mock_trainer.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def api_headers():
    """Standard API headers for testing"""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }


@pytest.fixture(scope="session")
def test_database_url():
    """Database URL for integration tests"""
    return os.getenv('TEST_DATABASE_URL', 'sqlite:///:memory:')


@pytest.fixture
def db_session(test_database_url):
    """Database session for testing"""
    engine = create_engine(test_database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestDataHelper:
    """Helper class for test data generation"""
    
    @staticmethod
    def create_fire_scenario(scenario_type="galley_fire", severity="moderate"):
        """Create a test fire scenario"""
        return {
            "scenario_type": scenario_type,
            "location": "test_location",
            "fire_type": "test_fire",
            "severity": severity,
            "response_actions": ["test_action_1", "test_action_2"],
            "expected_outcome": "fire_suppressed",
            "training_standards": ["NFPA_1500"]
        }
    
    @staticmethod
    def create_training_feedback(rating=4, comments="Good response"):
        """Create test training feedback"""
        return {
            "scenario_id": 1,
            "user_id": "test_user",
            "rating": rating,
            "comments": comments,
            "response_time": 120,
            "actions_taken": ["isolate_power", "apply_suppression"],
            "timestamp": "2024-01-01T12:00:00Z"
        }


@pytest.fixture
def test_data_helper():
    """Test data helper fixture"""
    return TestDataHelper()
