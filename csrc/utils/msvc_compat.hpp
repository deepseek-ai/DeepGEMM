#pragma once

// MSVC compatibility layer for DeepGEMM
// This header provides POSIX-compatible macros and includes for Windows/MSVC builds

#if defined(_MSC_VER)
    #define DG_MSVC 1
#else
    #define DG_MSVC 0
#endif

#if DG_MSVC
    // POSIX compatibility for Windows
    #include <process.h>
    #include <io.h>
    #include <direct.h>

    // Map POSIX functions to Windows equivalents
    #define popen  _popen
    #define pclose _pclose
    #define getpid _getpid

    // WEXITSTATUS: On Windows, pclose returns the exit code directly
    #define WEXITSTATUS(x) (x)

    // Include ciso646 for and/or/not keywords (C++ alternative tokens)
    // In C++17 and later, these are built-in, but MSVC requires this header
    // unless /permissive- is used consistently
    #include <ciso646>

    // Disable some MSVC warnings that are noisy for this codebase
    #pragma warning(disable: 4244) // conversion from 'type1' to 'type2', possible loss of data
    #pragma warning(disable: 4267) // conversion from 'size_t' to 'type', possible loss of data
    #pragma warning(disable: 4996) // deprecated functions

#else
    // POSIX systems
    #include <unistd.h>
    #include <sys/wait.h>
#endif
