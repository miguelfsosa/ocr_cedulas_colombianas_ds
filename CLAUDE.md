# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is an early-stage deep learning project applied to banking, bootstrapped from academic reading material. The `lecturas/` folder contains PDF source material (currently Chapter 2 of a deep learning for banking textbook) from which implementations are derived.

The working language for this project is **Spanish** — comments, variable names, documentation, and communication with the user should reflect this unless the user specifies otherwise.

## Project Structure (Intended)

As the project grows from the readings, the expected structure is:

```
deeplearning_banking/
├── lecturas/          # Source PDFs and reading material
├── notebooks/         # Jupyter notebooks for exploration and prototyping
├── src/               # Production-ready Python source code
├── data/              # Datasets (raw, processed)
├── models/            # Saved model weights and checkpoints
└── tests/             # Unit and integration tests
```

## Stack Conventions

When building out this project, default to:
- **Python** as the primary language
- **PyTorch** or **TensorFlow/Keras** for deep learning (confirm with user on first use)
- **Jupyter notebooks** for exploration; refactor mature code into `src/`
- `pyproject.toml` or `requirements.txt` for dependency management

## Workflow

When given a PDF from `lecturas/`, the expected workflow is:
1. Read and summarize the chapter's key concepts and methodologies
2. Identify implementable components (architectures, algorithms, pipelines)
3. Propose an implementation plan before writing code
4. Implement in a notebook first, then refactor to `src/` if appropriate
