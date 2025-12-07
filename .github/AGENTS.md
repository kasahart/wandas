# GitHub Configuration

This directory contains GitHub-specific configuration files for the Wandas repository.

## 📋 Copilot Instructions

This repository is configured with comprehensive GitHub Copilot instructions to help AI coding agents work effectively on the codebase.

### Main Instruction File

- **`copilot-instructions.md`**: Primary instructions for Copilot coding agents
  - Provides an overview of the Wandas architecture (frames, processing, I/O, visualization)
  - Documents development workflow and commands (testing, linting, type checking)
  - Establishes design principles (immutability, metadata preservation, Dask laziness)
  - Defines error handling patterns and testing expectations
  - Specifies role-based guidelines for planners, implementers, and reviewers

### Custom Agents

The `agents/` directory contains specialized agent definitions for different development tasks:

- **`wandas-planner.agent.md`**: Read-only planning agent that maps requirements to affected modules
- **`wandas-implementer.agent.md`**: Implementation agent focused on TDD and metadata preservation
- **`wandas-reviewer.agent.md`**: Review agent that validates frame immutability and test coverage
- **`wandas-publisher.agent.md`**: Publishing agent for git operations and PR creation

### Additional Instructions

The `instructions/` directory contains detailed prompts for specific design areas:

- **`frames-design.prompt.md`**: Guidelines for working with frame data structures
- **`processing-api.prompt.md`**: Patterns for processing module implementations
- **`io-contracts.prompt.md`**: I/O handling and file format specifications
- **`testing-workflow.prompt.md`**: Testing strategy and patterns
- **`agent-maintenance.prompt.md`**: Guidelines for maintaining agent definitions

## 🔧 How Copilot Uses These Instructions

When working with this repository:

1. **Copilot coding agents** automatically read `copilot-instructions.md` to understand project conventions
2. **Custom agents** can be invoked for specialized tasks (planning, implementation, review)
3. **Instruction prompts** provide deeper guidance for specific areas of the codebase

## 📝 Maintaining Copilot Instructions

When updating the codebase architecture or development workflow:

1. Update `copilot-instructions.md` if project-wide conventions change
2. Update specific agent definitions if role responsibilities evolve
3. Add or update instruction prompts when introducing new design patterns
4. Keep examples in sync with actual code patterns in the repository

## 🚀 For Contributors

These instructions help ensure consistent code quality and adherence to project patterns. When contributing:

- Review `copilot-instructions.md` to understand the project structure and conventions
- Follow the documented patterns for frames, processing modules, and I/O
- Run the specified commands for testing, linting, and type checking
- Maintain immutability, metadata preservation, and lazy evaluation principles

For more information on using Copilot instructions, see:
- [GitHub Copilot documentation](https://docs.github.com/en/copilot)
- [Best practices for Copilot coding agent in your repository](https://gh.io/copilot-coding-agent-tips)

## 📁 Other Files in This Directory

- **`workflows/`**: GitHub Actions workflows for CI/CD
- **`release-drafter.yml`**: Configuration for automated release notes
