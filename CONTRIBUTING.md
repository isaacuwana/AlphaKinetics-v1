# CONTRIBUTING.md (Contribution guidelines)
# Contributing to AlphaKinetics

Thank you for your interest in contributing to AlphaKinetics! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/isaacuwana/AlphaKinetics.git
cd AlphaKinetics
```

2. Create a virtual environment:
```bash
python -m venv alphakinetics-env
source alphakinetics-env/bin/activate  # On Windows: alphakinetics-env\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev,notebooks]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```