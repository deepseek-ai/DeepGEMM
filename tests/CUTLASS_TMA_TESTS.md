# CUTLASS TMA Performance Test Suite

## 概述

这个测试套件基于 CUTLASS 的 [`copy_sm90_tma.hpp`](../third-party/cutlass/include/cute/arch/copy_sm90_tma.hpp) 文件，使用**真正的 TMA 硬件指令**进行性能测试。

## 🎯 测试的 TMA 操作

### 1. TMA LOAD 操作
- **`SM90_TMA_LOAD_1D::copy()`** - 1D TMA 加载
- **`SM90_TMA_LOAD_2D::copy()`** - 2D TMA 加载
- **`SM90_TMA_LOAD_3D::copy()`** - 3D TMA 加载
- **`SM90_TMA_LOAD_4D::copy()`** - 4D TMA 加载
- **`SM90_TMA_LOAD_5D::copy()`** - 5D TMA 加载

### 2. TMA STORE 操作
- **`SM90_TMA_STORE_1D::copy()`** - 1D TMA 存储
- **`SM90_TMA_STORE_2D::copy()`** - 2D TMA 存储
- **`SM90_TMA_STORE_3D::copy()`** - 3D TMA 存储
- **`SM90_TMA_STORE_4D::copy()`** - 4D TMA 存储
- **`SM90_TMA_STORE_5D::copy()`** - 5D TMA 存储

### 3. TMA REDUCE ADD 操作
- **`SM90_TMA_REDUCE_ADD_1D::copy()`** - 1D TMA 归约加法
- **`SM90_TMA_REDUCE_ADD_2D::copy()`** - 2D TMA 归约加法
- **`SM90_TMA_REDUCE_ADD_3D::copy()`** - 3D TMA 归约加法
- **`SM90_TMA_REDUCE_ADD_4D::copy()`** - 4D TMA 归约加法
- **`SM90_TMA_REDUCE_ADD_5D::copy()`** - 5D TMA 归约加法

### 4. BULK COPY 操作
- **`SM90_BULK_COPY_G2S::copy()`** - Global 到 Shared 批量复制
- **`SM90_BULK_COPY_S2G::copy()`** - Shared 到 Global 批量复制

## 🔧 TMA 操作接口

### TMA LOAD 接口
```cpp
SM90_TMA_LOAD_1D::copy(
    void const* desc_ptr,     // TMA descriptor 指针
    uint64_t* mbar_ptr,       // Memory barrier 指针
    uint64_t cache_hint,      // Cache 提示
    void* smem_ptr,           // Shared memory 目标指针
    int32_t const& crd0       // 1D 坐标
);
```

### TMA STORE 接口
```cpp
SM90_TMA_STORE_1D::copy(
    void const* desc_ptr,     // TMA descriptor 指针
    void const* smem_ptr,     // Shared memory 源指针
    int32_t const& crd0       // 1D 坐标
);
```

### TMA REDUCE ADD 接口
```cpp
SM90_TMA_REDUCE_ADD_1D::copy(
    void const* desc_ptr,     // TMA descriptor 指针
    void const* smem_ptr,     // Shared memory 源指针
    int32_t const& crd0       // 1D 坐标
);
```

### BULK COPY 接口
```cpp
SM90_BULK_COPY_G2S::copy(
    void const* gmem_ptr,     // Global memory 源指针
    uint64_t* mbar_ptr,       // Memory barrier 指针
    void* smem_ptr,           // Shared memory 目标指针
    int32_t load_bytes        // 复制字节数
);
```

## 🏗️ 实现特点

### 1. 真正的 TMA 硬件指令
- 直接调用 CUTLASS 提供的 TMA 操作
- 使用内联汇编执行真正的 PTX TMA 指令
- 支持 SM90+ 架构的硬件加速

### 2. 完整的测试框架
- **GPU 预热**：确保测试结果稳定
- **多种数据大小**：测试不同负载下的性能
- **精确计时**：使用 CUDA Events 进行精确测量
- **带宽计算**：自动计算内存带宽

### 3. 内存管理
- **自动内存分配**：RAII 风格的 GPU 内存管理
- **数据预取**：支持 cache 预热测试
- **Barrier 同步**：正确的 TMA barrier 管理

## 📊 测试指标

### 性能指标
- **延迟 (Latency)**：操作完成时间 (ms)
- **带宽 (Bandwidth)**：数据传输速率 (GB/s)
- **吞吐量 (Throughput)**：每秒操作数

### 测试场景
- **不同数据大小**：1KB - 1MB
- **不同维度**：1D, 2D, 3D, 4D, 5D
- **Cache 行为**：Warm cache vs Cold cache
- **Bank Conflict**：有冲突 vs 无冲突访问模式

## 🚀 运行测试

### 前提条件
- **GPU 架构**：SM90+ (H100, H200)
- **CUDA 版本**：12.0+
- **编译器**：支持 C++17

### 编译和运行
```bash
# 进入测试目录
cd tests

# 运行测试脚本
./run_cutlass_tma_tests.sh
```

### 手动编译
```bash
# 创建构建目录
mkdir -p build && cd build

# 配置 CMake
cmake -DCMAKE_BUILD_TYPE=Release ../tests

# 编译
make cutlass_tma_performance_test -j$(nproc)

# 运行
./cutlass_tma_performance_test
```

## 📈 预期输出

```
CUTLASS TMA Performance Test Suite
===================================
GPU: NVIDIA H100 PCIe
Compute Capability: 9.0

============================================================
Testing with 1024 elements (4.00 KB)
============================================================

=== TMA LOAD Performance Test (Elements: 1024) ===
TMA LOAD 1D: 0.12 ms, Bandwidth: 167.23 GB/s
TMA LOAD 2D: 0.15 ms, Bandwidth: 133.78 GB/s

=== TMA STORE Performance Test (Elements: 1024) ===
TMA STORE 1D: 0.10 ms, Bandwidth: 200.45 GB/s
TMA STORE 2D: 0.13 ms, Bandwidth: 154.32 GB/s

=== TMA REDUCE ADD Performance Test (Elements: 1024) ===
TMA REDUCE ADD 1D: 0.14 ms, Bandwidth: 143.56 GB/s
TMA REDUCE ADD 2D: 0.17 ms, Bandwidth: 118.23 GB/s

=== BULK COPY Performance Test (Elements: 1024) ===
BULK COPY G2S: 0.08 ms, Bandwidth: 250.12 GB/s
BULK COPY S2G: 0.09 ms, Bandwidth: 222.89 GB/s
```

## 🔍 与之前实现的区别

### 之前的实现（模拟）
```cpp
// 使用普通 CUDA 内存操作模拟
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
```

### 现在的实现（真正的 TMA）
```cpp
// 使用真正的 CUTLASS TMA 硬件指令
SM90_TMA_LOAD_1D::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, coord);
```

## 🎯 测试价值

1. **真实性能**：测量真正的 TMA 硬件性能
2. **架构对比**：量化 TMA vs 传统内存操作的性能差异
3. **优化指导**：为实际应用提供性能调优参考
4. **硬件验证**：验证 TMA 硬件功能的正确性

## 📝 注意事项

1. **架构要求**：只能在 SM90+ GPU 上运行真正的 TMA 指令
2. **编译标志**：需要正确的 CUDA 编译标志和架构设置
3. **内存对齐**：TMA 操作对内存对齐有特殊要求
4. **Barrier 同步**：必须正确使用 memory barrier 进行同步

## 🔗 相关文件

- [`cutlass_tma_performance_test.cu`](cutlass_tma_performance_test.cu) - 主测试文件
- [`copy_sm90_tma.hpp`](../third-party/cutlass/include/cute/arch/copy_sm90_tma.hpp) - CUTLASS TMA 操作定义
- [`run_cutlass_tma_tests.sh`](run_cutlass_tma_tests.sh) - 自动化测试脚本
- [`CMakeLists.txt`](CMakeLists.txt) - 构建配置