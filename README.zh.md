# DeepGEMM

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

DeepGEMM 是一个统一、高性能的 Tensor Core 算子库，汇集了现代大语言模型的核心计算原语——GEMM（通用矩阵乘法，包括 FP8、FP4、BF16）、重叠通信的融合 MoE（Mega MoE）、用于闪电索引器（Lightning Indexer）的 MQA 打分算子、超连接（HyperConnection, HC）等——并整合至单一清晰的 CUDA 代码库中。所有算子均通过 DeepJIT 在运行时动态即时编译（JIT），安装时无需繁琐的 CUDA 编译过程。

DeepGEMM 借鉴了 [CUTLASS](https://github.com/nvidia/cutlass) 和 [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 的部分核心设计理念，但避免了对繁复模板和代数系统（algebras）的过度依赖。该库以极致简洁为设计初衷，仅包含有限的核心算子函数，使其成为学习 NVIDIA GPU 算子优化技术的绝佳、纯净的开源资源。

尽管设计轻量精简，DeepGEMM 在各种矩阵维度（shapes）下的性能均能达到甚至超越经过专家精心手工调优的库。

## 最新动态

- 2026.09.30
  - DeepGEMM 昇腾版本现已发布！详情请查阅 [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend/)。
  - 新增更多优化，包括局部性域（locality domain）特性，详情请查阅 [#462](https://github.com/deepseek-ai/DeepGEMM/pull/462)。
- 2026.09.10：稀疏索引器（Sparse Indexer）、Mega Gate、Mega mHC、DeepJIT、MoE 与索引器优化等。
    - 详情请查阅 [#432](https://github.com/deepseek-ai/DeepGEMM/pull/432)。
- 2026.04.16：Mega MoE、FP8xFP4 GEMM、FP4 索引器、PDL、更快的 JIT 编译等。
    - 详情请查阅 [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304)。
    - Mega MoE 基准测试结果请参考 [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316)。
- 2025.09.28：DeepGEMM 现已支持用于 DeepSeek v3.2 闪电索引器的打分算子（加权 ReLU MQA logits）。
    - 详情请查阅 [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200)。
- 2025.07.20：DeepGEMM 现已全面支持 SM90/SM100，并通过低 CPU 开销的 JIT C++ 模块完成全量重构。
    - 由于 NVCC 12.9 会自动进行 FFMA 交错（interleaving），所有后续优化将不再需要支持。
    - 详情请查阅 [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112)。
- 2025.05.14：DeepGEMM 现已提供用于稠密与 MoE 反向传播的权重梯度算子！详情请查阅 [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95)。
- 2025.04.18：DeepGEMM 在 H800 上达到最高 **1550 TFLOPS**！详情请查阅 [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74)、[#78](https://github.com/deepseek-ai/DeepGEMM/pull/78)、[#81](https://github.com/deepseek-ai/DeepGEMM/pull/81)、[#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) 与 [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526)。

## 快速上手

### 环境要求

- NVIDIA SM90 或 SM100 架构 GPU
- Python 3.8 或更高版本
- 支持 C++20 `<format>` 的编译器与标准库
- CUDA Toolkit 12.9 或更高版本
- PyTorch 2.3 或更高版本
- CUTLASS 4.0 或更高版本（可通过 Git submodule 克隆）

### 开发环境配置

```bash
# 必须克隆子模块
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# 链接核心头文件并构建 C++ 扩展
cat develop.sh
./develop.sh
```

### 安装

```bash
cat install.sh
./install.sh
```

随后在您的 Python 项目中导入 `deep_gemm` 即可开始使用！

## 接口说明

#### 使用须知

本库为 NVIDIA GPU 提供了高度优化的 GEMM 算子，遵循命名规范：`D = C + A @ B`。输入形状的内存布局为 NT（A 矩阵不转置，B 矩阵转置）。SM90 实现仅支持 NT 内存布局（行优先、列优先），而 SM100 实现支持所有内存布局（NT、TN、NN、TT）。例如，`fp8_gemm_nt` 将计算 `D = C + A @ B.T`。

在这两种架构下，左操作数（LHS）的缩放因子（Scaling Factor, SF）均必须采用 TMA 对齐且转置的布局。SM90 与 SM100 的缩放因子数据格式有所不同：

- SM90 要求缩放因子采用 FP32 格式。
- SM100 要求缩放因子采用紧凑的 [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats) 格式，将 4 个 UE8M0 打包为一个单精度整型 `torch.int`。

请注意，输入矩阵转置或 FP8 类型转换等操作须由用户自行处理，请在前置算子中独立实现或进行融合。虽然本库提供了一些基础的 PyTorch 工具函数，但直接调用可能会降低性能；本项目的核心聚焦点在于 GEMM 算子本身的极限优化。

#### 普通稠密 GEMM（非分组）

要执行基础的非分组 FP8 GEMM，请调用 `fp8_gemm_{nt, nn, tn, tt}` 函数。更多详细信息请参阅函数文档。

#### 分组 GEMM（连续布局 Contiguous Layout）

与 CUTLASS 中的传统分组 GEMM 不同，DeepGEMM 仅对 M 轴进行分组，而 N 和 K 轴必须保持固定。该设计专门针对 MoE 模型中所有专家共享相同形状的场景进行了定制。对于训练前向传播或推理预填充（Prefilling），每个专家可能处理不同数量的 token，我们将这些 token 拼接为一个单一连续张量，称为“连续（contiguous）”布局。请注意，每个专家的分段必须对齐到 GEMM 的 M 块大小（`get_mk_alignment_for_contiguous_layout()`）。更多详细信息请参阅 `m_grouped_fp8_gemm_{nt, nn}_contiguous` 函数文档。

我们还提供了用于 MoE 权重反向传播的 K 轴分组 API（此时 M 和 N 轴必须保持固定），详情请参阅 `k_grouped_fp8_gemm_tn_contiguous`。

#### 分组 GEMM（掩码布局 Masked Layout）

在推理解码（Decoding）阶段，当启用 CUDA Graph 且 CPU 无法预知每个专家接收到的 token 数量时，本库支持基于掩码的分组 GEMM。通过传入一个 mask 张量，算子将仅计算有效部分。

请使用 `m_grouped_fp8_gemm_nt_masked` 实现该功能并查阅相关文档。一个典型使用示例是将 [DeepEP](https://github.com/deepseek-ai/DeepEP) 低延迟算子的输出直接作为输入传入。

#### 索引器 V3.2 MQA 算子

该算子系列包含两个版本：非分页（用于预填充阶段）和分页（用于解码阶段）。
以非分页版本 `fp8_fp4_mqa_logits` 为例，其主要输入参数包括：

- `q`：一个 `(q_data, q_sf)` 元组；SM100 接受 MXFP4/MXFP8 数据以及打包的 UE8M0 缩放因子
- `kv`：一个 `(kv_data, kv_sf)` 元组，逻辑维度为 `[seq_len_kv, head_dim]`
- `weights`：形状为 `[seq_len, num_heads]` 的张量（在 SM100 上为 BF16）
- `cu_seq_len_k_start` 与 `cu_seq_len_k_end`：形状为 `[seq_len]` 的整型张量
- `max_seqlen_k`：任意 query 行的最大有效 KV 跨度

输出被压缩为 `[seq_len, max_seqlen_k]`；第 `i` 行从第 0 列开始存储其有效 KV 跨度。
对于 `q` 中的每个 token `i`，算子将遍历区间 `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])` 内的所有 token `j`，
并按如下方式计算对应的压缩 logit：

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)  # [head_dim]
out_ij = q[i, :, :] @ kv_j  # [num_heads]
out_ij = out_ij.relu() * weights[i, :]  # [num_heads]
out_ij = out_ij.sum()  # 标量
```

更多详细信息及分页版本 `fp8_fp4_paged_mqa_logits`，请参阅 `tests/test_attention.py`。

#### Mega MoE

Mega MoE 将 EP dispatch、线性层 1 与线性层 2（FP8xFP4 或 FP8xFP8）、SwiGLU 激活以及 EP combine 彻底融合重叠为一个单一超级算子（Mega-kernel），实现了 NVLink 通信与 Tensor Core 计算的深度重叠。该算子需配合对称内存（Symmetric Memory）进行多进程启动。使用示例：

```python
# 分配对称内存缓冲区
# 说明：需要 PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
    mma_type='fp8xfp4',  # 对于 FP8 路由专家权重请使用 'fp8xfp8'
)

# 将权重（FP4 或带 UE8M0 缩放因子的 FP8）转换为所需布局
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

# （可选）将权重本地化至局部性域（locality domains）中
transformed_l1 = (deep_gemm.localize(transformed_l1[0]), transformed_l1[1])
transformed_l2 = (deep_gemm.localize(transformed_l2[0]), transformed_l2[1])
deep_gemm.destroy_localizer()

# 在每次调用前将输入复制至缓冲区中
# 您亦可将这些复制操作融合至前序算子中
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# 运行融合的 Mega MoE 算子
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

包含多进程设置与性能基准测试的完整示例，请参阅 `tests/test_mega_moe.py`。

#### 实用工具函数

除上述核心算子外，本库还提供了一系列实用工具函数：

- `deep_gemm.set_num_sms` / `get_num_sms`：设置/获取使用的最大 SM 数量
- `deep_gemm.set_tc_util` / `get_tc_util`：设置/获取估算的 Tensor Core 利用率比例
- `deep_gemm.set_pdl` / `get_pdl`：启用/禁用程序化依赖启动（Programmatic Dependent Launch, PDL）
- `deep_gemm.use_deterministic_algorithms`：启用/禁用确定性算法
- `deep_gemm.set_mk_alignment_for_contiguous_layout` / `get_mk_alignment_for_contiguous_layout`：设置/获取连续布局的分组级 M/K 对齐值
- `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout`：获取理论上的最小 M/K 对齐值
- `deep_gemm.set_ignore_compile_dims`：配置 JIT 编译时忽略的维度
- `deep_gemm.set_block_size_multiple_of`：约束分块尺寸（block size）为给定值的倍数
- `deep_gemm.transform_sf_into_required_layout`：将缩放因子转换为所需布局
- `deep_gemm.get_tma_aligned_size`：获取所需的 TMA 对齐大小
- `deep_gemm.get_mn_major_tma_aligned_tensor`：获取 MN-major TMA 对齐的张量
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`：获取 MN-major TMA 对齐的张量（将 FP32 打包为 UE8M0）
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`：K-grouped GEMM 打包算子

本库还提供了一些常用的环境变量配置：

未设置时，每个 `DG_JIT_*` 变量都会自动回退至对应的全局 `DJ_JIT_*` 变量。

- 常规配置
    - `DG_JIT_DEBUG`：`0` 或 `1`，启用 JIT 调试功能，包括编译器命令与 PTXAS 输出、加载耗时报告、代码行信息以及 PTX/SASS 转储；默认为 `0`
    - `DG_PRINT_CONFIGS`：`0` 或 `1`，打印每个矩阵形状所选取的配置，默认为 `0`
- JIT 缓存
    - `DG_JIT_CACHE_DIR`：字符串，编译后算子的缓存目录（或用冒号 `:` 分隔的目录列表）；查找时自前向后检索所有路径（首次命中优先），缓存未命中时编译至首个路径，默认为 `$HOME/.dj`
- 编译器选择
    - `DG_JIT_NVCC_COMPILER`：字符串，NVCC 编译器路径；未指定时将依次通过 `CUDA_HOME`、`CUDA_PATH`、`which nvcc` 以及 `/usr/local/cuda` 自动定位 CUDA
    - `DG_JIT_CPP_STANDARD`：整型，C++ 标准版本，默认为 `20`
- 编译器输出
    - `DG_JIT_PRINT_COMPILER_COMMAND`：`0` 或 `1`，打印编译执行命令，默认为 `0`
    - `DG_JIT_PTXAS_VERBOSE`：`0` 或 `1`，显示详细的 PTXAS 编译输出，默认为 `0`
    - `DG_JIT_CHECK_NO_SPILLS`：`0` 或 `1`，断言编译后算子不存在寄存器溢出（spill），默认为 `0`
    - `DG_JIT_CHECK_NO_LOCAL_MEMORY`：`0` 或 `1`，断言编译后算子不使用局部内存（local memory），默认为 `0`
    - `DG_JIT_PRINT_LOAD_TIME`：`0` 或 `1`，打印算子加载耗时，默认为 `0`
- 调试与性能分析
    - `DG_JIT_WITH_LINEINFO`：`0` 或 `1`，为性能分析工具嵌入源码行信息，默认为 `0`
    - `DG_JIT_DUMP_ASM`：`0` 或 `1`，同时转储 PTX 和 SASS 汇编，默认为 `0`
    - `DG_JIT_DUMP_PTX`：`0` 或 `1`，转储 PTX 输出，默认为 `0`
    - `DG_JIT_DUMP_SASS`：`0` 或 `1`，转储 SASS 输出，默认为 `0`
    - `DG_COMM_KERNEL_DEBUG`：`0` 或 `1`，在每次 Mega MoE 调用前清零对称缓冲区以便于调试，默认为 `0`
    - `DG_USE_NVIDIA_TOOLS`：`0` 或 `1`，在外部 NVIDIA 分析工具下运行时跳过内部性能分析，默认为 `0`
- 构建选项
    - `DG_SKIP_CUDA_BUILD`：`0` 或 `1`，在安装时跳过 CUDA 扩展构建，默认为 `0`
    - `DG_FORCE_BUILD`：`0` 或 `1`，强制本地编译而不是下载预构建 Wheel 包，默认为 `0`

更多使用示例与技术细节，请参阅[测试代码](tests)或查阅对应的 Python 文档。

## 致谢

DeepGEMM 受 [CUTLASS](https://github.com/nvidia/cutlass) 项目启发而诞生。在此向所有优秀的开发者致以崇高的敬意与感谢！

## 许可证

本代码仓库基于 [MIT License](LICENSE) 开源许可证发布。

## 引用

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient BLAS kernel library on GPU},
      author={Chenggang Zhao and Zhean Xu and Liang Zhao and Jiashi Li and Chenhao Xu and Anyi Xu and Shengyu Liu and Kexing Zhou and Kuai Yu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月14日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
