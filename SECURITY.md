# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in DeepGEMM, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Instead, please report security issues via one of these methods:
   - Email the maintainers directly (if contact info is available)
   - Use [GitHub's private vulnerability reporting](https://github.com/deepseek-ai/DeepGEMM/security/advisories/new)

### What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: We aim to acknowledge receipt within 48 hours
- **Status Update**: We will provide updates on the investigation within 7 days
- **Resolution**: Critical vulnerabilities will be prioritized for immediate patching

### Scope

This security policy covers:

- The DeepGEMM library code (Python and C++/CUDA)
- Build and installation scripts
- JIT compilation infrastructure

### Out of Scope

- Vulnerabilities in dependencies (please report to the respective projects)
- Issues in third-party code under `third-party/`

## Security Best Practices

When using DeepGEMM:

1. **JIT Compilation**: Be aware that DeepGEMM uses JIT compilation which executes dynamically generated code. Only run DeepGEMM with trusted inputs.

2. **Environment Variables**: Some functionality is controlled via environment variables. Ensure these are set appropriately in production environments.

3. **CUDA Security**: Follow NVIDIA's security best practices for CUDA applications.

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged (with permission) in release notes.
