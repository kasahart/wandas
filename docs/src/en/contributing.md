# Contributing Guide

Thank you for your interest in contributing to the Wandas project. This guide explains how to contribute to the project.

## Types of Contributions

You can contribute to the Wandas project in the following ways:

- Bug reports and feature requests
- Documentation improvements
- Bug fixes
- New feature implementations
- Test additions and improvements
- Performance optimizations

## Setting Up the Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wandas.git
cd wandas
```

### 2. Set Up a Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Development Workflow

### 1. Create a New Branch

Create a new branch for new features or bug fixes:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Code Style

Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for code. We use the following tools to ensure code quality:

- **Ruff**: Code linter and formatter
- **mypy**: Static type checking

Before committing your code, run the following commands to check your code style:

```bash
# Linting
ruff check wandas tests

# Type checking
mypy wandas tests
```

### 3. Testing

Always add tests for new features or bug fixes. Tests are run using `pytest`:

```bash
pytest
```

To generate a coverage report:

```bash
pytest --cov=wandas tests/
```

### 4. Documentation

Code changes require documentation updates:

- Write [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings for functions and classes
- Update or add appropriate documentation pages (tutorials, how-tos, API reference) for new features
- Add sample code to the `examples/` directory as needed

To build and check the documentation:

```bash
cd docs
mkdocs serve
```

Then visit <http://localhost:8000> in your browser to check.

### 5. Creating a Pull Request

Once your changes are complete, create a pull request (PR):

1. Commit your changes and push to the remote repository

   ```bash
   git add .
   git commit -m "Descriptive commit message"
   git push origin your-branch-name
   ```

2. Create a pull request on the GitHub repository page
3. In the PR description, include what was changed, what issues were resolved, and how to test it

## Review Process

All PRs are reviewed through the following process:

1. Automated CI tests must pass
2. Code review by at least one maintainer
3. Requested changes and responses as needed
4. Merge approval

## Communication

You can communicate for questions or discussions through:

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and discussions
- Project mailing list (if available)

## Code of Conduct

All participants in the project are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Treat other contributors with respect and engage in cooperative and constructive communication.
