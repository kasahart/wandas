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
  - Specifies role-based guidelines for planners, implementers, reviewers, and PR-only publication handoff

### Custom Agents

The `agents/` directory contains specialized agent definitions for different development tasks:

- **`wandas-planner.agent.md`**: Preferred first-stop read-only planning agent for substantive work that maps requirements to affected modules or `.github/` customization and workflow files when this agent is exposed in the current runtime
- **`wandas-implementer.agent.md`**: Implementation agent for approved plans or direct, clearly scoped follow-up work when the role request, prior handoff, or validation context is already clear
- **`wandas-reviewer.agent.md`**: Read-only review agent for completed implementations or direct review requests when the role request, prior handoff, or validation context is already clear
- **`wandas-publisher.agent.md`**: PR-only publishing agent for branch, stage, commit, push, and pull request create or update

### Active Instructions

The `instructions/` directory contains the active `.instructions.md` guidance files used by agents:

- **`frames-design.instructions.md`**: Guidelines for working with frame data structures
- **`processing-api.instructions.md`**: Patterns for processing module implementations
- **`io-contracts.instructions.md`**: I/O handling and file format specifications
- **`testing-workflow.instructions.md`**: Testing strategy and patterns
- **`agent-maintenance.instructions.md`**: Guidelines for maintaining agent definitions
- **`test-grand-policy.instructions.md`**: Cross-cutting test quality policy
- **`test-frames-policy.instructions.md`**: Frame-specific test patterns
- **`test-processing-policy.instructions.md`**: Processing-layer test patterns
- **`test-io-policy.instructions.md`**: I/O test patterns
- **`test-visualization-policy.instructions.md`**: Visualization test patterns

Archived review notes are kept outside `instructions/` so they do not present as live agent guidance.

## 🔧 How Copilot Uses These Instructions

When working with this repository:

1. **Copilot coding agents** automatically read `copilot-instructions.md` to understand project conventions
2. **Custom agents** can be invoked for specialized tasks (planning, implementation, review, PR publication); first check which agents are actually exposed in the current runtime before requiring a planner-first workflow
3. **Active instruction files** provide deeper guidance for specific areas of the codebase

## 📝 Maintaining Copilot Instructions

When updating the codebase architecture or development workflow:

1. Update `copilot-instructions.md` if project-wide conventions change
2. Update specific agent definitions if role responsibilities evolve
3. Add or update active `.instructions.md` files when introducing new design patterns
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
- **`workflows/release-drafter.yml`**: GitHub Actions workflow that updates GitHub Release drafts from merged PR data
- **`release-drafter.yml`**: Release Drafter configuration consumed by the workflow above
- **`workflows/cd.yml`**: Tag-driven package publication and GitHub Release workflow
