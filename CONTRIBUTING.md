# Contributing to DeepGEMM

Thank you for your interest in contributing to DeepGEMM! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- NVIDIA GPU with SM90 (Hopper) or SM100 (Blackwell) architecture
- CUDA Toolkit 12.3+ (12.9+ recommended for best performance)
- Python 3.8+
- PyTorch 2.1+
- C++20 compatible compiler

### Development Setup

1. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
   cd DeepGEMM
   ```

2. Set up the development environment:
   ```bash
   ./develop.sh
   ```

3. Run tests to verify your setup:
   ```bash
   python tests/test_core.py
   python tests/test_fp8.py
   python tests/test_bf16.py
   ```

## How to Contribute

### Reporting Bugs

- Use the [Bug Report template](https://github.com/deepseek-ai/DeepGEMM/issues/new?template=bug_report.yml)
- Include your environment details (GPU, CUDA version, PyTorch version)
- Provide a minimal reproducible example if possible
- Include the full error message and stack trace

### Suggesting Features

- Use the [Feature Request template](https://github.com/deepseek-ai/DeepGEMM/issues/new?template=feature_request.yml)
- Describe the problem you're trying to solve
- Explain your proposed solution
- Consider if you'd be willing to implement it

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   python tests/test_core.py
   python tests/test_fp8.py
   python tests/test_bf16.py
   ```

4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")

5. **Push and create a Pull Request**:
   - Provide a clear description of your changes
   - Link any related issues
   - Be responsive to review feedback

## Code Style

### C++ Code

- Use C++17/C++20 features where appropriate
- Follow the existing formatting in the codebase
- Use meaningful variable and function names
- Add comments for complex logic

### Python Code

- Follow PEP 8 style guidelines
- Use type hints where possible
- Keep functions focused and modular

## Project Structure

```
DeepGEMM/
├── csrc/                    # C++ source code
│   ├── apis/                # Public API implementations
│   ├── jit_kernels/         # JIT kernel implementations
│   │   ├── heuristics/      # Configuration heuristics
│   │   └── impls/           # Kernel implementations
│   └── utils/               # Utility functions
├── deep_gemm/               # Python package
│   ├── include/             # Header files for JIT
│   ├── testing/             # Testing utilities
│   └── utils/               # Python utilities
├── tests/                   # Test files
├── third-party/             # Third-party dependencies (CUTLASS, fmt)
└── scripts/                 # Build and utility scripts
```

## Testing

- All new features should include tests
- Run the full test suite before submitting PRs
- Tests are located in the `tests/` directory

## Questions?

- Open a [Question issue](https://github.com/deepseek-ai/DeepGEMM/issues/new?template=question.yml)
- Check existing issues and discussions

## License

By contributing to DeepGEMM, you agree that your contributions will be licensed under the MIT License.
