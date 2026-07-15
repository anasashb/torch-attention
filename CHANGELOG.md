# Changelog

All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-15

### Changed

- Renamed the distribution from `torch-attention` to `adlers`.
- Renamed the Python import package from `torch_attention` to `adlers`.
- Rebranded the project as ADLERS.

## [0.2.0] - 2026-07-11

### Added

- Added an `sdpa` backend for `ScaledDotProductAttention`, delegating to
  PyTorch's native scaled dot-product attention implementation.
- Exported `ScaledDotProductAttention` from the package root.

### Changed

- Documented selectable scaled dot-product attention backends.
- Made attention validation helpers static methods.

### Fixed

- Required attention masks to use `torch.bool` so mask semantics stay
  consistent across backends.
- Used the distribution name when reading package version metadata.

## [0.1.1] - 2026-05-22

### Changed

- Moved the package to a `src/` layout.
- Switched the build backend from Poetry to uv.
- Replaced the Poetry lockfile with a uv lockfile.
- Modernized dependency metadata and added uv dependency groups.
- Added separate CPU and CUDA PyTorch dependency groups for local development.
- Replaced flake8 with Ruff linting.
- Tightened mypy configuration.
- Added pre-commit hooks for Ruff, isort, and Black.
- Added tox environments for linting, type checking, building, and tests.
- Added terminal coverage reporting to pytest.

### Removed

- Removed flake8 configuration.
