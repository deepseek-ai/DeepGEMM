# PR501 最终候选范围与评审入口

本候选沿用已经完成 GPU 验收的数学实现、配置、白名单和回退规则，补齐原集成工作区遗漏的 TP 本地 I=384 输入支持及示例打包文件。唯一源码整理为换行统一；没有重调参数或改变数值容差。本轮不使用 GPU。

## 评审顺序

| 层次 | 内容 | 应用补丁 |
|---|---|---|
| 核心接口 | per-call M tile、gate/up 顺序、Python stub、回归测试 | 01 |
| 私有框架输入 | direct expert IDs、安全 padding、TP 本地 I=384、包结构 | 02 |
| 框架示例 | vLLM 0.32.0 默认关闭适配器、独立 TP2 白名单 | 03 |
| 说明材料 | 当前 PR 正文、支持边界、示例使用说明 | 04 |

补丁基于 PR501 head `8fb87090aff4bfe6c8da6080cc4caa3525fd8979`。本地分支 `codex/pr501-final-review` 已准备候选 diff，无新 commit、push、评论或评审请求。原集成工作区及其真实 Git index 保留。

准确性修复已在现有 PR501 中；补丁 01–04 是它之后的增量。review bundle 的 `full-pr-diff.stat` 按 GitHub 比较的共同祖先给出完整范围，避免把增量误当成整个项目。上游当前 tip 已前进；本次未执行向该 tip 的 merge/rebase，不把补丁适用检查写成上游合并检查通过。

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

## 结果、质量与未完成范围

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
| 中文论文 | 同上 `paper/main.pdf` 与 `paper/main.tex`（25 页、16 图） |
| 当前候选与 CPU 回执 | `deliverables/sm120-pr501-review-ready-20261120/` |

关键 SHA256：

- 实测包 manifest：`676eb83a3784082fbf03cc9f7cd2508c34aaeb55b43fd60462e1fda4b35f012a`。
- 实测扩展：`e5274e4a8c9aac8747258e7ed8fa1abca84986c3885bf56424d6cdb37437d2d6`。
- 最终质量冻结协议：`56f4fc7643bd91babd5b3a44176b478e57c7bc1dcf2ebde5186bb03c92eba5b2`。
- 最终质量原始归档：`89cd9b7ae7e8345e7de1a627ac39a6a33db8f321242cc4e97d06b3ae2a6e13d1`。

候选对所选 242 个核心/测试/适配器文件核对 LF 归一化内容，逐文件记录 raw hash；原始实测字节保留在冻结包。本候选统一 Git diff 换行，不能声称生成的新扩展拥有旧 SHA 或已经重新在 GPU 编译。未初始化的 CUTLASS/DeepJIT 子模块及新构建身份亦需明确，详见本地 bundle README。
