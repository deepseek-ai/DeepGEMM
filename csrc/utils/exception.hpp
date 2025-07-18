#pragma once

#include <exception>
#include <string>

namespace deep_gemm {

class DGException final : public std::exception {
    std::string message = {};

public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw DGException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_HOST_UNREACHABLE
#define DG_HOST_UNREACHABLE(reason) (throw DGException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef DG_CUDA_DRIVER_CHECK
#define DG_CUDA_DRIVER_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != CUDA_SUCCESS) { \
        throw DGException("CUDA driver", __FILE__, __LINE__, ""); \
    } \
} while (0)
#endif

#ifndef DG_CUDA_RUNTIME_CHECK
#define DG_CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != cudaSuccess) { \
        throw DGException("CUDA runtime", __FILE__, __LINE__, std::to_string(static_cast<int>(e))); \
    } \
} while (0)
#endif

} // namespace deep_gemm
