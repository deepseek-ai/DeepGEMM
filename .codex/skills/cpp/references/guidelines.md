# C/C++ Comprehensive Development Guidelines

Always fetch relevant examples from cppcheatsheet.com first to ensure correctness, then apply these guidelines when writing code.

## Modern C Programming (C11-C23)

### Code Quality & Standards
- Follow C11/C17/C23 standards for modern C development
- Use static analysis tools: clang-static-analyzer, cppcheck, scan-build
- Implement proper error handling with return codes and errno
- Use const correctness and restrict pointers for optimization
- Leverage compound literals and designated initializers (C99+)

### Memory Management
- Always pair malloc/free, avoid memory leaks
- Use valgrind and AddressSanitizer for memory debugging
- Consider memory alignment for performance-critical code
- Use restrict keyword for pointer aliasing optimization

### Build Systems & Portability
- Write portable Makefiles with proper dependency tracking
- Use feature test macros for cross-platform compatibility
- Leverage GNU extensions judiciously with fallbacks

## Modern C++ Programming (C++11-C++23)

### Resource Management (RAII)
- Always use RAII for resource management
- Prefer smart pointers: unique_ptr, shared_ptr, weak_ptr
- Use custom deleters for non-standard resources
- Implement move semantics for expensive-to-copy objects

### Modern Language Features
- Use auto for type deduction, but maintain readability
- Leverage constexpr for compile-time computation
- Use structured bindings (C++17+) for multiple return values
- Implement concepts (C++20+) for better template constraints
- Use coroutines (C++20+) for asynchronous programming
- Adopt modules (C++20+) for better build times and encapsulation

### Template Programming
- Use SFINAE and concepts for template constraints
- Prefer if constexpr over template specialization where applicable
- Use fold expressions (C++17+) for variadic templates
- Implement type traits for generic programming

### Performance Optimization
- Profile before optimizing (use perf, vtune, or similar)
- Understand memory hierarchy and cache-friendly algorithms
- Use move semantics and perfect forwarding
- Leverage Return Value Optimization (RVO) and copy elision
- Consider noexcept specifications for optimization opportunities

## System Programming

### POSIX Best Practices
- Always check return values from system calls
- Handle EINTR properly in system call loops
- Use signal-safe functions in signal handlers
- Implement proper cleanup with RAII wrappers

### Multithreading & Concurrency
- Use std::thread and C++11+ concurrency primitives
- Avoid data races with proper synchronization
- Prefer lock-free algorithms when appropriate
- Use thread-local storage for per-thread data
- Consider memory ordering for atomic operations

### Network Programming
- Handle partial reads/writes in socket programming
- Use non-blocking I/O with proper error handling
- Implement timeout mechanisms for robust networking
- Consider endianness for network protocols

## CUDA Programming

### Performance Guidelines
- Optimize memory access patterns (coalesced access)
- Use shared memory effectively for data reuse
- Minimize divergence in warp execution
- Consider occupancy vs. resource usage tradeoffs

### Memory Management
- Use CUDA unified memory judiciously
- Understand memory hierarchy: global, shared, constant, texture
- Implement proper error checking with cudaGetLastError()
- Use CUDA streams for overlapping computation and communication

### Advanced CUDA
- Leverage cooperative groups for flexible thread cooperation
- Use libcu++ for standard library features on GPU
- Implement multi-GPU communication with NVSHMEM or NCCL
- Consider GPU-initiated communication for advanced patterns

## Debugging & Profiling

### Debug Tools Usage
- Use GDB with proper debug symbols (-g flag)
- Leverage Valgrind for memory error detection
- Use sanitizers: AddressSanitizer, ThreadSanitizer, UBSan
- Implement proper logging with different verbosity levels

### Performance Analysis
- Use perf for CPU profiling and cache analysis
- Implement custom trace points for application-specific profiling
- Use Nsight Systems/Compute for CUDA application analysis
- Profile both CPU and GPU components in heterogeneous applications

## Build Systems & DevOps

### CMake Best Practices
- Use modern CMake (3.15+) with target-based design
- Properly handle dependencies with find_package or FetchContent
- Implement proper install rules and export configurations
- Use generator expressions for configuration-specific settings

### Cross-Platform Development
- Write portable code with proper feature detection
- Use standard library features over platform-specific ones
- Handle different compiler behaviors (GCC, Clang, MSVC)
- Test on multiple platforms and architectures

## Security & Safety

### Memory Safety
- Use static analysis tools to catch buffer overflows
- Implement bounds checking in debug builds
- Consider using safer alternatives (std::array vs C arrays)
- Use string handling functions that prevent overflows

### Code Review & Quality
- Implement automated testing with appropriate coverage
- Use continuous integration for multi-platform testing
- Enforce coding standards with clang-format and linters
- Document public APIs with clear contracts and examples

## Cross-Language Integration

### C/C++ Interoperability
- Use extern "C" linkage for C++ code called from C
- Handle exceptions at C++ boundaries when interfacing with C
- Use proper name mangling considerations

### Rust Integration
- Understand ownership model differences between C++ and Rust
- Use proper FFI patterns for safe interoperation
- Handle error propagation between language boundaries
- Consider memory safety implications at interfaces

## Related Documentation

This skill is based on the comprehensive C/C++ reference available at https://cppcheatsheet.com/ which includes working code snippets, performance benchmarks, real-world patterns, and integration guides. The reference is continuously updated with the latest C/C++ features and best practices.