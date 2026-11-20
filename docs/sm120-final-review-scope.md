# PR501 最终候选范围与评审入口

本候选沿用已经完成 GPU 验收的专用融合内核、舍入修复、配置、白名单和回退规则，补齐原集成工作区遗漏的 TP 本地 I=384 输入支持及示例打包文件。随后合并上游 `8c8a92076c4de39e5ed29459e55fea3eaca6256c`，适配新增 epilogue 接口。没有重调参数或改变数值容差；合并后的 `643f4f3` 已完成独立干净构建、正确性/数值回归、core/TP 四种 sanitizer 和双卡完整模型复测；详见文末新增证据。

## 评审顺序

| 层次 | 内容 | 应用补丁 |
|---|---|---|
| 核心接口 | per-call M tile、gate/up 顺序、Python stub、回归测试 | 01 |
| 私有框架输入 | direct expert IDs、安全 padding、TP 本地 I=384、包结构 | 02 |
| 框架示例 | vLLM 0.32.0 默认关闭适配器、独立 TP2 白名单 | 03 |
| 说明材料 | 当前 PR 正文、支持边界、示例使用说明 | 04 |

补丁 01–04 基于 PR501 原 head `8fb87090aff4bfe6c8da6080cc4caa3525fd8979`，对应四个独立提交；之后以单独的 merge commit 集成上游。原集成工作区及其真实 Git index 保留，发布通过隔离候选分支完成。

准确性修复已在原 PR501 中；补丁 01–04 是它之后的增量。冻结 review bundle 的 diff 与 CPU 回执描述合并上游之前的候选，不代表当前提交。当前 masked BF16 GEMM 保留上游 positional `epilogue=None`，并保留本 PR keyword-only `sm120_m_tile=0`。stub 生成器改用 pybind 声明的 Python 参数名，避免误导调用者使用 C++ 内部名 `epilogue_class`。合并后 9 项核心 CPU 检查与 4 项 TP 策略检查通过；stub AST 有效，示例打包和默认关闭导入检查通过。上述为先前本地检查；后续新增 GPU 验证见文末，不能把两次检查混记为一次运行。

## 机制与正式范围

两段 CTA 内融合：L1 直接读 X，完成 gate/up GEMM 与 SwiGLU，保留 BF16 H；L2 做 down GEMM，并在 CTA 内按路由顺序加权 combine。每个计算 warp 负责一位专家，另外四个 warp 负责 stage 搬运，线程数为 `(bundle+4)*32`。不引入跨 CTA 队列、全 grid barrier 或新的 TP 通信。

BF16 中点问题采用 zero-C K16 MMA、显式 FP32 累加和 TwoSum 邻近中点重算；生产不使用 FP64。保留原 FP64 reference、BF16 中间边界和原容差。通过有限回归覆盖不构成任意输入的正确舍入证明。

公开 `single_token` 策略只接受以下 T=1、RTX 5090/170 SM 配置；配置三元组为 BN/stages/experts-per-CTA。

| D / I / E / k | L1 | L2 |
|---|---|---|
| 2048 / 768 / 128 / 8 | 32/3/2 | 16/4/8 |
| 4096 / 1536 / 128 / 8 | 16/2/8 | 32/2/8 |
| 2048 / 1408 / 60 / 4 | 16/3/4 | 16/4/4 |
| 2048 / 1408 / 64 / 6 | 16/3/2 | 16/4/6 |

T>1、未知键、其他设备在该策略下回退。压力测试退化的 3584/2560/E64/k8 仍拒绝。旧的 `validated` 策略支持表未删减，也未把它的 T=2/4 覆盖混算成新单 token 策略的证据。

## 框架与 TP 边界

- 适配器默认关闭，仅接受 vLLM 0.32.0、BF16 和受支持布局；异步调度、DP/EP、DBO/EPLB 等保守回退。
- PP2 每 rank 的本地 I=768；TP2 的每个专家 gate/up 对应中间通道一起切分，本地 I=384，L2 返回局部 Y，交给框架原有 all-reduce。
- TP2 是示例中的独立 opt-in，未加入公开策略；要求 `(T,E,k,D,I,tp)=(1,128,8,2048,384,2)`。
- 实测 PP2 为 slot 路径，L1=32/3/2、L2=16/4/8；实测 TP2 为 direct-ID 路径，显式设置 L1=16/4/2、L2=16/2/8。
- TP 环境变量不设置时，源码默认是 16/3/2 与 16/3/8；不能用这些默认值复现上述实测结果。本候选不擅自修改默认值。
- 每层拥有工作区；模型 forward 串行，不允许多个独立 stream 并发共享。计数器只计 dispatch/capture，不计 Graph replay。
- 输入专家 ID 要求合法专家不重复；padding 会映射到保留行并在 combine 中排除。该约定并非 host 上逐值校验。

## 历史快照的结果、质量与未完成范围

完整原始 BF16 Qwen3-Coder-30B-A3B-Instruct、双 5090、batch=1、串行请求、mode0、decode Graph：历史等输出整模型场景请求延迟 PP2 降低 2.27–7.22%，TP2 降低 4.82–8.30%。HTTP 的 72 行中有 56 行等输出，分别降低 1.00–3.00% 和 2.23–5.84%；16 行不同输出单列，不计入等输出收益。各实现两个独立进程位于同一会话，未证明跨会话稳定性或吞吐优势。

最终补充质量验收 11/11 完成，`PASS_SCOPED_NLL`：英文、中文、代码各 48 个独立来源、每来源 32 个共同固定前缀位置，每并行 4608 个位置。事先冻结 50,000 次配对来源簇 bootstrap、六比较 Bonferroni 调整和 log(1.01) 工程界限。六项区间上界均低于界限，九项硬控制通过。原始 logits 本地独立复算一致。

该判定只支持预定 NLL 工程界限，不能写“语义无损”。PP2/TP2 top-1 分别有 69/74 个位置不同，一致率 98.50%/98.39%，最大 KL 为 0.342/0.289。历史自然生成分叉和严格数值闸门失败保留。首个全局 MoE 层同输入 BM32/fused 的 H/局部 Y 逐位一致；未观察 native 内部投影、累加及外层 all-reduce，不能写“根因已完全定位”。

已完成当前有限范围的研究验收。更多完整模型、任务 EM/F1/pass@k、batch>1、并发吞吐、跨会话服务稳定性是后续扩展，不能由本轮结果代替。适配器继续默认关闭。

## 证据索引与源码身份

路径均相对于本地研究根目录 `D:/code/NJU/AI_Infra_Researce`；原始证据与冻结交付不覆写。

| 证据 | 位置 |
|---|---|
| 实测 source 与 harness | `deliverables/sm120-final-acceptance-v1-20261120/measured/` |
| 历史整模型/HTTP 原始证据 | `log/2026-11-20/batch1-final1-live/final-download2/b1f1/verified/b1f1/` |
| 最终质量原始 logits/日志 | `log/2026-11-20/final-quality-live/final-download/extracted/final-quality-2/` |
| 质量结果、独立复算、定位 | `deliverables/sm120-final-acceptance-completed-20261120/` 中的 REPORT、analysis、secondary-metrics、localization-summary、local-remote-reanalysis-audit |
| 历史中文论文 | 同上 `paper/main.pdf` 与 `paper/main.tex`（25 页、16 图） |
| 代码审查整理版论文 | `deliverables/sm120-pr501-review-20261120/paper/`（26 页、19 图、13 表；当前复测置于第4.2节） |
| 当前候选与 CPU 回执 | `deliverables/sm120-pr501-review-ready-20261120/` |
| 上游合并、发布身份与当前 CPU 回执 | `deliverables/sm120-pr501-publish-20261120/` |

关键 SHA256：

- 实测包 manifest：`676eb83a3784082fbf03cc9f7cd2508c34aaeb55b43fd60462e1fda4b35f012a`。
- 实测扩展：`e5274e4a8c9aac8747258e7ed8fa1abca84986c3885bf56424d6cdb37437d2d6`。
- 最终质量冻结协议：`56f4fc7643bd91babd5b3a44176b478e57c7bc1dcf2ebde5186bb03c92eba5b2`。
- 最终质量原始归档：`89cd9b7ae7e8345e7de1a627ac39a6a33db8f321242cc4e97d06b3ae2a6e13d1`。

候选对所选 242 个核心/测试/适配器文件核对 LF 归一化内容，逐文件记录 raw hash；原始实测字节保留在冻结包。本候选统一 Git diff 换行。新 GPU 构建使用 gitlink 固定的 CUTLASS/DeepJIT，拥有独立扩展 SHA；旧扩展与旧质量验收身份保留。

## 上游合并后的 GPU 验证（2026-11-20）

GPU 执行 head 为 `643f4f3ce4991889b702e814210fe754b7f36ec6`，tree 为 `1c557cbf5fb0e848a8d35d0428340a66241f58d8`。随后的 `f5a77a3` 只更新本文与验证 JSON；该提交的内核、API、测试、适配器与配置相对 GPU 执行 head 的字节不变。之后的主机端审查修订单列于下节。

新扩展 SHA256：`a89993de11376f4ad9d15d04246df925a60476d09e575998ddcacad9a6555aa9`。证据归档 SHA256：`0dd5bb1a33b287beecdd1c50607d05ea26479e6e8dad9400faa235608b0f2056`。本地完整交付位于 `deliverables/sm120-pr501-gpu-20261120/`；机器可读摘要为 [合并后验证](validation/sm120-pr501-postmerge-20261120.json)。

- 干净构建、9项 Linux CPU 合同、30项公开正确性、42项原始 SwiGLU 合同、4项原 FP64/中点/抵消回归通过；原容差未改。
- core 与 TP 的 memcheck/synccheck/racecheck/initcheck 全部干净，racecheck 无 hazard、error 或 warning；布局/direct-ID 4项和实际 positional epilogue/keyword-only tile ABI 通过。
- TP 两 rank 各9个 slot/int64/int32 记录的 BM32 局部结果、改变输入 Graph 与 NCCL 外层求和逐位一致；TP 本地 I=384 的回归通过。
- 单层20行全部披露：四个支持尺寸、两个种子，相对每行实测最快分离组合，warm 为 1.147–1.997×，512 MiB 缓存干扰为 1.081–1.448×。Qwen2 尺寸继续回退，不计作融合收益。
- 完整 BF16 Qwen3-Coder-30B、batch=1、上下文32/128/512、生成32/128，PP2/TP2共12场景均跨实现/重复等 token；请求墙钟延迟PP2 降低 0.84–3.10%；TP2 生成128 token时降低 1.72–5.55%，生成32 token时变慢 0.49–7.57%。每实现2个引擎进程，每场景每进程5轮，同一主机会话，不是吞吐、HTTP或跨会话稳定性结论。

TP2 短请求回退保留。上下文512/生成32时，原生/融合首 token 中位延迟99.18/123.44ms，首末token间隔179.67/177.49ms；decode间隔的节省未抵消首token增加。不能由此断言CPU/prefill/通信/包装器中的具体根因。核心正确性通过不等于完整服务性能验收，TP适配器继续默认关闭。

相对历史对应进程/提示的48项核对，token相同48项，已记录top-5 logprob完全相同48项。历史 `PASS_SCOPED_NLL` 仍属于旧扩展；本轮没有重跑其三领域完整语料，不能写成当前提交的自动质量通过。

## 主机端代码审查修订（2026-11-21）

审查 `f5a77a3` 时发现，direct-ID 输入地址不满足私有接口要求的16字节对齐时，适配器仍进入融合后触发断言；同时权重预检遗漏 down 权重的 CUDA 设备一致性。已补齐这两个元数据检查。direct-ID 不对齐时在任何融合/共享专家调用前回到原生路径；slot 准备仍接受连续的偏移 ID，因为其内部输出拥有对齐的地址。两段权重必须位于同一 CUDA 设备才可进入融合候选。

新增 `tests/test_sm120_vllm_adapter.py` 在 CPU 上执行实际注册与分派代码，以元数据替身检查 int32/int64 偏移 ID、slot/direct 正常分派顺序、配置、共享专家调用顺序、权重设备拒绝和默认关闭。它不替代真实 vLLM ABI 或 GPU 算术测试。修订前能复现 direct-ID 断言及错误的权重资格判定；修订后新增4项与原stub/策略9项合计13项在 Windows 与 Linux 各自通过。机器可读记录见 [主机端审查](validation/sm120-pr501-host-review-20261121.json)。

该修订只增加主机端资格与回退检查，未修改 C++、CUDA、算术、配置、框架通信及合法输入的 GPU 发射顺序；主机检查有所增加，旧计时不自动成为修订后的新计时。2026-11-20的GPU/模型结果仍按原执行 head 报告，不将这次CPU检查写作新GPU或新服务验收。适配器仍默认关闭，TP2短请求回退仍待定位。
