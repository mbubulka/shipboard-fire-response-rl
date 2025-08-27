from setuptools import setup, find_packages

setup(
    name="shipboard-fire-response-rl",
    version="1.0.0",
    description="Advanced Reinforcement Learning system for shipboard fire response training and decision support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shipboard Fire Response RL Team",
    author_email="contact@shipboard-ai.com",
    url="https://github.com/mbubulka/shipboard-fire-response-rl",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "mysql-connector-python>=8.1.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
        "rich>=13.5.0",
        "tqdm>=4.66.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0"
        ],
        "aws": [
            "boto3>=1.28.0",
            "botocore>=1.31.0"
        ],
        "ml": [
            "scipy>=1.11.0",
            "joblib>=1.3.0",
            "plotly>=5.15.0"
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "shipboard-rl=cli:main"
        ]
    },
    package_data={
        "shipboard_rl": [
            "data/*.json",
            "data/*.yaml",
            "models/*.pth"
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        "Topic :: Safety/Security"
    ],
    keywords=[
        "artificial-intelligence",
        "fire-response",
        "maritime-safety",
        "deep-learning",
        "emergency-response",
        "training-simulation",
        "shipboard-operations"
    ],
    project_urls={
        "Documentation": "https://github.com/mbubulka/shipboard-fire-response-rl/wiki",
        "Bug Tracker": "https://github.com/mbubulka/shipboard-fire-response-rl/issues",
        "Source Code": "https://github.com/mbubulka/shipboard-fire-response-rl",
    }
)
