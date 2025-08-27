# Contributing to Shipboard Fire Response AI

Thank you for your interest in contributing to the Shipboard Fire Response AI System! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Security Considerations](#security-considerations)

## Code of Conduct

This project adheres to a code of conduct that ensures a welcoming and inclusive environment for all contributors. Please read and follow our Code of Conduct.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- MySQL 8.0+ (for database features)
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/shipboard-fire-response-ai.git
   cd shipboard-fire-response-ai
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install development dependencies
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Run Tests**
   ```bash
   pytest
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve existing documentation or add new guides
5. **Testing**: Add test cases or improve test coverage
6. **Performance**: Optimize existing code for better performance

### Before You Start

1. **Check Existing Issues**: Look through existing issues and pull requests to avoid duplicates
2. **Create an Issue**: For significant changes, create an issue first to discuss the approach
3. **Follow Standards**: Ensure your code follows our coding standards and guidelines

### Coding Standards

#### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: Maximum 88 characters (Black default)
- **Imports**: Use absolute imports when possible
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Use type hints for all public functions

#### Code Formatting

We use automated code formatting:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

#### Example Code Style

```python
"""Module for fire response prediction algorithms."""

from typing import Dict, List, Optional, Tuple
import logging

import torch
import numpy as np


class FireResponsePredictor:
    """Predicts optimal fire response actions using AI."""
    
    def __init__(
        self, 
        model_path: str, 
        confidence_threshold: float = 0.8
    ) -> None:
        """Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file.
            confidence_threshold: Minimum confidence for predictions.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model: Optional[torch.nn.Module] = None
        
    def predict(
        self, 
        scenario: Dict[str, any]
    ) -> Tuple[List[str], float]:
        """Predict fire response actions.
        
        Args:
            scenario: Dictionary containing scenario details.
            
        Returns:
            Tuple of (predicted_actions, confidence_score).
            
        Raises:
            ValueError: If scenario data is invalid.
        """
        if not self._validate_scenario(scenario):
            raise ValueError("Invalid scenario data")
            
        # Implementation here
        pass
```

### Commit Message Guidelines

Use conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(prediction): add multi-scenario batch prediction
fix(database): resolve connection timeout issues
docs(api): update endpoint documentation
test(dqn): add unit tests for attention mechanism
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Ensure all documentation is updated
2. **Add Tests**: Include tests for new functionality
3. **Check Coverage**: Maintain or improve test coverage
4. **Update Changelog**: Add entry to CHANGELOG.md
5. **Rebase**: Rebase your branch on the latest main

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer reviews the code
3. **Testing**: All tests must pass
4. **Documentation**: Documentation must be complete and accurate
5. **Approval**: Maintainer approval required before merging

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

#### Unit Tests

```python
"""Unit tests for fire response predictor."""

import pytest
from unittest.mock import Mock, patch

from shipboard_ai.prediction.predictor import FireResponsePredictor


class TestFireResponsePredictor:
    """Test cases for FireResponsePredictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create a test predictor instance."""
        return FireResponsePredictor(
            model_path="test_model.pth",
            confidence_threshold=0.8
        )
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.confidence_threshold == 0.8
        assert predictor._model is None
    
    @pytest.mark.parametrize("scenario,expected", [
        ({"location": "galley"}, True),
        ({}, False),
        ({"invalid": "data"}, False)
    ])
    def test_scenario_validation(self, predictor, scenario, expected):
        """Test scenario validation with different inputs."""
        result = predictor._validate_scenario(scenario)
        assert result == expected
```

#### Integration Tests

```python
"""Integration tests for the complete prediction pipeline."""

import pytest
import json

from shipboard_ai.api.server import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app({'TESTING': True})
    return app.test_client()


class TestPredictionAPI:
    """Test prediction API integration."""
    
    def test_prediction_endpoint(self, client):
        """Test complete prediction workflow."""
        scenario = {
            "location": "galley",
            "fire_type": "cooking_oil",
            "severity": "moderate"
        }
        
        response = client.post(
            '/api/v1/predict',
            data=json.dumps(scenario),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'recommended_actions' in data
        assert 'confidence_score' in data
```

### Test Coverage

Maintain minimum 80% test coverage:

```bash
# Run tests with coverage
pytest --cov=shipboard_ai --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation Standards

### Code Documentation

- **Modules**: Include module-level docstrings
- **Classes**: Document purpose, attributes, and usage
- **Methods**: Use Google-style docstrings with Args, Returns, Raises
- **Type Hints**: Include type hints for all public APIs

### API Documentation

- Update API documentation for any endpoint changes
- Include request/response examples
- Document error codes and responses
- Provide SDK usage examples

### User Documentation

- Keep README.md updated
- Update installation instructions
- Include usage examples
- Document configuration options

## Security Considerations

### Security Guidelines

1. **Sensitive Data**: Never commit API keys, passwords, or sensitive data
2. **Input Validation**: Validate all inputs thoroughly
3. **SQL Injection**: Use parameterized queries
4. **Dependencies**: Keep dependencies updated
5. **Authentication**: Implement proper authentication for APIs

### Security Review

Security-sensitive changes require additional review:

- Authentication/authorization changes
- Database schema modifications
- API endpoint additions
- Dependency updates
- Configuration changes

### Reporting Security Issues

**Do not** create public issues for security vulnerabilities. Instead:

1. Email security concerns to: security@shipboard-ai.com
2. Include detailed description and reproduction steps
3. Allow time for assessment and fix before disclosure

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: All contributors listed
- **Release Notes**: Significant contributions mentioned
- **Documentation**: Contributors credited in relevant sections

## Getting Help

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For security issues and private matters

### Resources

- **Documentation**: https://shipboard-ai.readthedocs.io
- **API Reference**: https://api.shipboard-ai.com/docs
- **Examples**: https://github.com/shipboard-ai/examples

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Shipboard Fire Response AI System! Your contributions help make maritime safety training more effective and accessible.
