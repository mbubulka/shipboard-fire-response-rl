# API Documentation

## Overview

The Shipboard Fire Response AI System provides a comprehensive REST API for integrating with training systems, accessing AI predictions, and managing training data.

## Base URL

```
http://localhost:5000/api/v1/
```

## Authentication

Currently, the API uses basic authentication. Include your API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Health Check

#### GET /health

Check if the API is running and healthy.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00Z",
    "version": "1.0.0",
    "database": "connected",
    "model": "loaded"
}
```

### Fire Response Prediction

#### POST /predict

Get AI-powered fire response recommendations.

**Request Body:**
```json
{
    "scenario": {
        "location": "galley",
        "fire_type": "cooking_oil",
        "severity": "moderate",
        "available_equipment": ["wet_chemical", "co2", "water"],
        "personnel_count": 3,
        "evacuation_status": "partial"
    },
    "environmental_factors": {
        "wind_speed": 15,
        "temperature": 25,
        "humidity": 60,
        "sea_state": 3
    }
}
```

**Response:**
```json
{
    "prediction_id": "pred_123456",
    "recommended_actions": [
        {
            "action": "isolate_power",
            "priority": 1,
            "confidence": 0.95,
            "estimated_time": "30 seconds"
        },
        {
            "action": "apply_wet_chemical",
            "priority": 2,
            "confidence": 0.88,
            "estimated_time": "2 minutes"
        }
    ],
    "risk_assessment": {
        "current_risk": "moderate",
        "projected_risk": "low",
        "time_critical": false
    },
    "confidence_score": 0.92,
    "model_version": "enhanced_dqn_v1.0"
}
```

### Training Data Management

#### GET /training/scenarios

Retrieve training scenarios.

**Query Parameters:**
- `scenario_type` (optional): Filter by scenario type
- `difficulty` (optional): Filter by difficulty level
- `standards` (optional): Filter by safety standards
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
    "scenarios": [
        {
            "id": 1,
            "scenario_type": "galley_fire",
            "location": "galley",
            "fire_type": "cooking_oil",
            "severity": "moderate",
            "difficulty": "intermediate",
            "training_standards": ["NFPA_1500", "USCG_CG022"],
            "expected_actions": ["isolate_power", "apply_wet_chemical"],
            "learning_objectives": ["proper_suppression", "safety_procedures"]
        }
    ],
    "total_count": 150,
    "page_info": {
        "limit": 50,
        "offset": 0,
        "has_next": true
    }
}
```

#### POST /training/scenarios

Create a new training scenario.

**Request Body:**
```json
{
    "scenario_type": "engine_room_fire",
    "location": "engine_room",
    "fire_type": "fuel_oil",
    "severity": "high",
    "difficulty": "advanced",
    "description": "High-intensity fuel fire in main engine room",
    "training_standards": ["NFPA_1670", "Maritime_RVSS"],
    "expected_actions": ["emergency_shutdown", "activate_co2"],
    "learning_objectives": ["emergency_procedures", "system_shutdown"],
    "environmental_conditions": {
        "temperature": 45,
        "visibility": "poor",
        "noise_level": "high"
    }
}
```

### Feedback System

#### POST /feedback

Submit training feedback for model improvement.

**Request Body:**
```json
{
    "session_id": "session_123",
    "scenario_id": 1,
    "user_id": "trainee_456",
    "actions_taken": [
        {
            "action": "isolate_power",
            "timestamp": "2024-01-01T12:01:00Z",
            "success": true,
            "time_taken": 25
        }
    ],
    "outcome": {
        "fire_suppressed": true,
        "casualties": 0,
        "property_damage": "minimal",
        "completion_time": 180
    },
    "rating": 4,
    "comments": "Good response time, proper procedure followed",
    "instructor_notes": "Excellent situational awareness"
}
```

**Response:**
```json
{
    "feedback_id": "fb_789",
    "status": "recorded",
    "model_updated": true,
    "performance_score": 85,
    "areas_for_improvement": [
        "communication_protocols",
        "equipment_handling"
    ]
}
```

### Model Training

#### POST /training/start

Start a new training session for the AI model.

**Request Body:**
```json
{
    "training_type": "incremental",
    "data_sources": ["feedback", "scenarios", "standards"],
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    },
    "validation_split": 0.2
}
```

**Response:**
```json
{
    "training_id": "train_123",
    "status": "started",
    "estimated_duration": "2 hours",
    "progress_url": "/training/progress/train_123"
}
```

#### GET /training/progress/{training_id}

Check training progress.

**Response:**
```json
{
    "training_id": "train_123",
    "status": "in_progress",
    "progress_percent": 45,
    "current_epoch": 45,
    "total_epochs": 100,
    "current_loss": 0.123,
    "current_accuracy": 0.87,
    "estimated_time_remaining": "1 hour 15 minutes",
    "metrics": {
        "training_loss": 0.123,
        "validation_loss": 0.156,
        "training_accuracy": 0.87,
        "validation_accuracy": 0.84
    }
}
```

### Analytics and Reporting

#### GET /analytics/performance

Get model performance analytics.

**Query Parameters:**
- `start_date` (optional): Start date for analytics (YYYY-MM-DD)
- `end_date` (optional): End date for analytics (YYYY-MM-DD)
- `metric_type` (optional): Type of metrics to retrieve

**Response:**
```json
{
    "performance_metrics": {
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.91,
        "f1_score": 0.90
    },
    "usage_statistics": {
        "total_predictions": 1250,
        "average_response_time": "150ms",
        "error_rate": 0.02
    },
    "training_effectiveness": {
        "improvement_rate": 0.15,
        "user_satisfaction": 4.2,
        "completion_rate": 0.88
    }
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes and error messages.

### Error Response Format

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid scenario data provided",
        "details": {
            "field": "fire_type",
            "issue": "Unknown fire type 'plasma'"
        },
        "request_id": "req_123456"
    }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

API requests are rate-limited to:
- 100 requests per minute for prediction endpoints
- 1000 requests per hour for data retrieval endpoints
- 10 requests per minute for training endpoints

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

## SDK Examples

### Python SDK Usage

```python
from shipboard_ai import ShipboardAIClient

client = ShipboardAIClient(api_key="your_api_key")

# Get prediction
scenario = {
    "location": "galley",
    "fire_type": "cooking_oil",
    "severity": "moderate"
}

prediction = client.predict(scenario)
print(f"Recommended actions: {prediction.recommended_actions}")

# Submit feedback
feedback = {
    "scenario_id": 1,
    "rating": 4,
    "comments": "Good response"
}

result = client.submit_feedback(feedback)
```

### JavaScript SDK Usage

```javascript
import ShipboardAI from 'shipboard-ai-js';

const client = new ShipboardAI({
    apiKey: 'your_api_key',
    baseUrl: 'http://localhost:5000/api/v1'
});

// Get prediction
const scenario = {
    location: 'galley',
    fire_type: 'cooking_oil',
    severity: 'moderate'
};

const prediction = await client.predict(scenario);
console.log('Recommended actions:', prediction.recommended_actions);
```

## Webhooks

Configure webhooks to receive real-time notifications about training progress, model updates, and system events.

### Webhook Configuration

```json
{
    "url": "https://your-system.com/webhooks/shipboard-ai",
    "events": ["training.completed", "model.updated", "prediction.made"],
    "secret": "webhook_secret_key"
}
```

### Webhook Payload Example

```json
{
    "event": "training.completed",
    "timestamp": "2024-01-01T12:00:00Z",
    "data": {
        "training_id": "train_123",
        "model_version": "v1.1.0",
        "performance_metrics": {
            "accuracy": 0.94,
            "improvement": 0.02
        }
    },
    "signature": "sha256=..."
}
```
