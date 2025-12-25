# Makefile for DeepGEMM development
# Run 'make help' to see available commands

.PHONY: help build develop install clean test lint format check all

# Default target
help:
	@echo "DeepGEMM Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Build:"
	@echo "  make build      - Build the package"
	@echo "  make develop    - Build for development (creates symlinks)"
	@echo "  make install    - Build and install the package"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-fp8   - Run FP8 GEMM tests"
	@echo "  make test-bf16  - Run BF16 GEMM tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint       - Run linters (ruff)"
	@echo "  make format     - Format code (ruff)"
	@echo "  make check      - Run all checks (lint + format check)"
	@echo "  make typos      - Check for typos"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make clean-all  - Remove all generated files"
	@echo ""
	@echo "Other:"
	@echo "  make all        - Build, check, and test"

# Build targets
build:
	python setup.py build

develop: clean
	./develop.sh

install: clean
	./install.sh

# Test targets
test:
	python -m pytest tests/ -v

test-fp8:
	python tests/test_fp8.py

test-bf16:
	python tests/test_bf16.py

test-attention:
	python tests/test_attention.py

# Code quality targets
lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

format-check:
	ruff format --check .
	ruff check .

check: format-check lint

typos:
	typos .

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f deep_gemm/*.so
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf stubs/
	rm -f deep_gemm/include/cute
	rm -f deep_gemm/include/cutlass

# Combined targets
all: build check test
