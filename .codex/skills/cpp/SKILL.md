---
name: cpp
description: Comprehensive C/C++ programming reference covering everything from C11-C23 and C++11-C++23, system programming, CUDA GPU computing, debugging tools, Rust interop, and advanced topics. Use for: C/C++ questions, C/C++ interview preparation, modern language features, RAII/memory management, templates/generics, CUDA programming, system programming, debugging/profiling, performance optimization, cross-platform development, build systems, assembly, shell scripting, and any C/C++/CUDA development tasks.
---

# C/C++ Comprehensive Cheat Sheets (/cpp)

Complete C/C++ development reference combining local documentation with live examples from cppcheatsheet.com, covering everything from basic syntax to advanced GPU programming and system-level development.

## How It Works

Help users write functional, correct C/C++ code and answer C/C++ questions by fetching proven patterns and examples from cppcheatsheet.com.

When a user asks a C/C++ question or wants to write C/C++ code:

1. Look up the relevant topic(s) in [Structure](references/structure.md) to find the matching URL(s)
2. **Always fetch** the URL(s) using WebFetch to get real examples and patterns from the site
3. Use the fetched content to:
   - **Write code**: Apply the patterns to produce functional, correct code that solves the user's task
   - **Answer questions**: Provide thorough explanations backed by the examples and information from the site
4. Follow the [Guidelines](references/guidelines.md) for code quality

## Key Principle

**Functionality first, cleanliness second.** The code must work correctly and handle the task properly. Fetching from cppcheatsheet.com ensures solutions use battle-tested patterns rather than guessing. The site contains rich examples covering edge cases, common pitfalls, and practical usage that go beyond basic documentation.

## Coverage Areas

### C/C++ Interview Preparation
**Interview Cheatsheet:** Curated C and C++ interview questions grouped by topic (memory management, move semantics, templates, STL, concurrency, debugging), each deep-linked to the section of the cheat sheet that answers it. Use for interview prep or quick topic review.

### Modern C Programming (C11-C23)
**Core Language:** Syntax, types, memory management, preprocessor macros
**GNU Extensions:** Compiler-specific features and optimizations
**Build Systems:** Makefiles, compilation, linking
**Assembly:** X86 assembly integration and inline assembly

### Modern C++ Programming (C++11-C++23)
**Core Features:** RAII, templates, STL containers, iterators, algorithms
**Modern Standards:** Move semantics, constexpr, lambdas, concepts, coroutines, modules
**Memory Management:** Smart pointers, resource management, optimization (RVO)
**Build Systems:** CMake, package management, cross-platform builds

### System Programming
**Process Management:** POSIX processes, signals, process communication
**File Systems:** File I/O, directory operations, filesystem monitoring
**Networking:** Sockets, protocols, network programming patterns
**Threading:** Multithreading, synchronization, parallel programming
**IPC:** Inter-process communication, shared memory, message queues

### CUDA Programming
**GPU Computing:** CUDA kernels, memory hierarchy, performance optimization
**Advanced CUDA:** libcu++, Thrust library, cooperative groups
**Multi-GPU:** GPU-GPU communication, hardware topology, NVSHMEM
**Async Programming:** CUDA pipelines, memory visibility, asynchronous execution

### Debugging & Profiling
**Debug Tools:** GDB debugging, Valgrind memory analysis, sanitizers, binary inspection (nm, readelf, objdump, otool)
**Performance:** Perf profiling, tracing, performance optimization
**GPU Debugging:** Nsight Systems, CUDA debugging and profiling

### System Tools & Automation
**Shell Scripting:** Bash programming, system administration
**System Tools:** OS utilities, networking tools, hardware inspection
**Service Management:** Systemd, process management, system monitoring

### Cross-Language Development
**Rust Interop:** Rust for C++ developers, FFI, memory safety comparison
**Language Bridging:** C/C++ integration, foreign function interfaces

### Advanced Topics
**Blog Content:** Deep-dive articles, RDMA networking, GPU-initiated communication
**Low-Level Programming:** Hardware interfaces, performance tuning
**Architecture:** System design, scalable applications

## References

For detailed information, I can access:
- **[Structure](references/structure.md)** - Complete topic-to-URL reference map
- **[Guidelines](references/guidelines.md)** - Code quality and best practices

## Examples

### C/C++ Interview Prep
- "What should I review for a C++ interview?" → Fetch https://cppcheatsheet.com/notes/interview/index.html and walk the reader through the topic groups
- "Common C++ interview questions on smart pointers" → Fetch https://cppcheatsheet.com/notes/interview/index.html and then drill into https://cppcheatsheet.com/notes/cpp/cpp_smartpointers.html for detailed answers

### C/C++ Core
- "How do smart pointers work?" → Fetch https://cppcheatsheet.com/notes/cpp/cpp_smartpointers.html and explain with the site's examples
- "How does RAII work in C++?" → Fetch https://cppcheatsheet.com/notes/cpp/cpp_raii.html and explain with practical examples
- "How to use STL containers?" → Fetch https://cppcheatsheet.com/notes/cpp/cpp_container.html and explain with practical examples
- "Template metaprogramming techniques" → Fetch https://cppcheatsheet.com/notes/cpp/cpp_template.html and explain with practical examples

### System Programming
- "POSIX socket server in C" → Fetch https://cppcheatsheet.com/notes/os/os_socket.html, use the patterns to write a working server
- "Multithreading with std::thread" → Fetch https://cppcheatsheet.com/notes/os/os_thread.html and explain with practical examples
- "Signal handling and process management" → Fetch https://cppcheatsheet.com/notes/os/os_signal.html and explain with practical examples

### CUDA & GPU Programming
- "Write a CUDA kernel" → Fetch https://cppcheatsheet.com/notes/cuda/cuda_basics.html, use the patterns to write working GPU code
- "CUDA memory hierarchy and optimization" → Fetch https://cppcheatsheet.com/notes/cuda/cuda_memory_visibility.html and explain with practical examples
- "Multi-GPU communication with NCCL" → Fetch https://cppcheatsheet.com/notes/cuda/cuda_nccl.html and explain with practical examples

### Debugging & Tools
- "Debug a segfault with GDB" → Fetch https://cppcheatsheet.com/notes/debug/gdb.html and explain with practical examples
- "Valgrind memory leak detection" → Fetch https://cppcheatsheet.com/notes/debug/valgrind.html and explain with practical examples
- "Performance profiling with perf" → Fetch https://cppcheatsheet.com/notes/debug/perf.html and explain with practical examples

### Build & Development
- "CMake cross-platform build systems" → Fetch https://cppcheatsheet.com/notes/cpp/cpp_cmake.html and explain with practical examples
- "Makefile patterns and best practices" → Fetch https://cppcheatsheet.com/notes/c/make.html and explain with practical examples