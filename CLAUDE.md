# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is a Python project exploring LLM interpretability — creating examples that demonstrate how to investigate and understand how LLMs work.

## Project Status

Early-stage. No build tooling or source code exists yet. The `.gitignore` is set up for a Python project and references tools like pytest, ruff, mypy, and various package managers (uv, poetry, pdm, pixi).

## Getting Started

As tooling is added, document commands here. Likely conventions based on the `.gitignore`:

- **Package manager**: uv, poetry, or pdm (TBD)
- **Testing**: pytest
- **Linting/formatting**: ruff
- **Type checking**: mypy

## Architecture

No source structure exists yet. When building out the project, consider:

- A top-level `src/interpretability/` package for reusable code
- Jupyter notebooks or Marimo apps for interactive exploration examples
- Example scripts that demonstrate specific interpretability techniques
