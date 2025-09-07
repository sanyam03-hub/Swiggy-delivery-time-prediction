# Contributing to Swiggy Delivery Time Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/swiggy-delivery-time-prediction.git
   cd swiggy-delivery-time-prediction
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate test data:**
   ```bash
   python src/data/data_generator.py
   ```

4. **Train models:**
   ```bash
   python src/models/simple_trainer.py
   ```

## 📝 Contribution Types

We welcome the following types of contributions:

### 🐛 Bug Fixes
- Fix issues in existing code
- Improve error handling
- Performance optimizations

### ✨ New Features
- New machine learning models
- Additional data sources
- Dashboard enhancements
- API improvements

### 📚 Documentation
- README improvements
- Code comments
- API documentation
- Tutorials and examples

### 🧪 Testing
- Unit tests
- Integration tests
- Performance tests
- Data validation tests

## 🔧 Code Guidelines

### Python Style
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Maximum line length: 100 characters

### Machine Learning Best Practices
- Document model assumptions
- Include feature importance analysis
- Validate data preprocessing steps
- Use proper cross-validation

### Testing
- Write tests for new functionality
- Ensure tests pass before submitting PR
- Include edge cases in tests

## 📋 Pull Request Process

1. **Update documentation** if needed
2. **Add or update tests** for your changes
3. **Ensure all tests pass:**
   ```bash
   python -m pytest tests/ -v
   ```
4. **Check code style:**
   ```bash
   flake8 src/ --max-line-length=100
   ```
5. **Create pull request** with:
   - Clear title and description
   - Link to related issues
   - Screenshots for UI changes

## 🏷️ Commit Message Format

Use clear, descriptive commit messages:

```
type(scope): description

Examples:
feat(models): add neural network model
fix(api): handle missing parameters
docs(readme): update installation instructions
test(models): add unit tests for preprocessor
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

## 🚨 Reporting Issues

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (OS, Python version, etc.)
6. **Error messages** and stack traces

## 💡 Feature Requests

For feature requests, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** clearly
3. **Explain the use case** and benefits
4. **Provide examples** if applicable

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private inquiries

## 🏆 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing! 🎉