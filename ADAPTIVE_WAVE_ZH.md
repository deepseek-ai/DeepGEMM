# DeepGEMM MegaMoE Adaptive Wave Sizing 中文方案说明

## 1. 背景与优化目标

MegaMoE 将一个 EP rank 上的本地专家分批组织成多个 scheduler wave。每个
wave 内先完成 L1（gate/up）计算，再衔接 L2（down）计算。每个 wave 放入
多少专家会同时影响：

- SM 填充率与尾部 CTA 浪费；
- L1 到 L2 之间的 activation/cache 局部性；
- ring buffer 容量；
- 不同专家 token 数不均衡时的慢 rank 延迟。

上游启发式只能看到平均 tokens/expert，无法区分“均衡地落在所有专家”与
“少数热专家承载大部分 token”这两类分布。本方案的目标是在不改变数值结果、
不扩大未校准配置范围、且不向稳态路径引入每次同步开销的前提下，根据上一
统计窗口的真实路由分布调整 `num_experts_per_wave`。

当前实现是 **B200、FP8×FP4 MegaMoE、显式 opt-in** 的窄范围策略。默认行为
与 upstream 完全一致。

## 2. Baseline 启发式

上游 `get_num_experts_per_wave_for_mega_moe` 主要执行以下步骤：

1. 根据 ring buffer 容量得到每个 wave 可容纳的专家数上限；
2. 用平均 tokens/expert 估计每个专家的 M tile 数；
3. 根据 L1 N tile 数估计每个专家产生的 CTA 数；
4. 使用固定 imbalance factor 估计填满全部 SM 所需的最少专家数；
5. 在一个有限区间内搜索，使最后一个 partial wave 尽量饱满。

这个启发式是可靠的通用 fallback，但平均值会丢失两个重要信号：

- 有多少本地专家实际收到 token；
- 专家 token 数的离散程度（探索阶段使用，最终策略不再依赖该指标）。

Adaptive Wave Sizing 不重写这套逻辑，而是先完整计算 upstream 结果，只在
经过实机校准的条件内替换 wave size；其他情况直接返回 upstream 结果。

## 3. 探索过程与被淘汰的方案

### 3.1 Imbalance-aware `block_m`

第一版思路根据真实专家 token 数缩小 `block_m`，希望减少冷专家的 padding。
虽然它能降低理论 padded rows，但 B200 张量核的 M 维利用率、tile 调度开销和
流水线行为不能由 padded rows 单独描述。完整 block_m 网格校准显示，该策略在
偏斜路由下最差会使慢 rank 回退约 **31%**。

因此最终提交删除了生产用的 imbalance-aware `block_m` 选择器，只保留
`DG_MEGA_MOE_FORCE_BLOCK_M` 作为校准钩子。生产路径继续使用 upstream
block_m。

### 3.2 每次 launch 同步读取路由统计

第二版直接在每次 MegaMoE 调用前将累计 receive stats 从 GPU 拷贝到 CPU。
配置本身能够找到更快的 wave size，但同步 D2H 会破坏 kernel 提交节奏，并可能
影响 GPU 时钟。same-process A/B 中整体约回退 **0.8%**，因此这一实现也被淘汰。

### 3.3 缓存统计窗口

最终实现只在启动阶段取得两个快照来形成有效 delta，之后缓存该分布，每 256
次 launch 刷新一次。这样稳态的绝大多数调用不发生 D2H，也不会因为累计计数
持续增长而反复生成不同 JIT 配置。

## 4. 最终统计采样实现

入口位于 `csrc/apis/mega.hpp`。每个线程维护一个 `thread_local` cache：

- receive-stats device pointer；
- 上一次累计计数 `previous_cumulative`；
- 最近一次有效窗口 delta `cached_delta`；
- 距离上次刷新的调用数；
- 当前 cache 是否有效。

累计统计不能直接用于策略判断，因为其绝对值会随运行时间增长。刷新时对每个
本地专家计算：

```text
delta[e] = current_cumulative[e] - previous_cumulative[e]
```

只有所有 delta 非负且总和大于零时，窗口才有效。以下情况严格 fallback：

- 第一次看到该统计 tensor；
- tensor 指针或长度变化；
- 累计计数回绕或被重置；
- 窗口 delta 总和为零；
- 未提供统计 tensor；
- 未设置启用开关。

有效 delta 会被传入 FP8×FP4 MegaMoE JIT heuristic。API 会先计算
expected tokens/expert；未命中校准 shape 或 tokens 窗口时完全跳过
receive-stat D2H，而不是采样后再由 heuristic fallback。窗口内刷新间隔固定为
256 launch；在缓存命中路径中没有同步 D2H。

## 5. 最终决策规则

定义：

```text
E_local = num_experts / num_ranks
expected_tpe = num_tokens * num_topk / E_local
active_ratio = count(recv[e] > 0) / E_local
```

最终策略的判断顺序如下：

```text
default_wave = upstream_heuristic(...)

if adaptive flag 未启用或 stats 无效:
    return default_wave

if not (127.5 < expected_tpe <= 128.5):
    return default_wave

if active_ratio <= 0.92:
    candidate = 8                 # 稀疏/高偏斜，优先判断
else:
    candidate = default_wave      # balanced 与中等偏斜

if candidate 超过 ring capacity:
    return default_wave
return candidate
```

探索阶段曾为 balanced 路由选择 wave 12。前三轮测量为正，但新增独立 6×30
复验中回退 1.276%，且 0/6 获胜，说明收益不稳定。因此最终生产策略删除 wave
12 分支，只保留跨轮次稳定的 sparse/high-skew wave 8。

在本次固定模型参数下：

```text
num_experts = 256
num_ranks = 8
num_topk = 8
E_local = 32
expected_tpe = tokens_per_rank / 4
```

API 还会严格检查校准 shape：EP 8、256 experts、top-k 8、hidden 7168、
intermediate 2048。第一轮 gate 曾覆盖 **259～514 tokens/rank**。补充测试加入 384（96
tokens/expert）后，高偏斜 wave 8 稳定回退 1.481%（0/6 获胜），证明不能在两个
校准点之间直接插值。最终 gate 收窄到 **127.5 < expected_tpe <= 128.5**；在
本次固定形状下大致对应 512 tokens/rank 附近。窗口之外无论路由偏斜如何都使用
upstream wave size。

## 6. JIT 与配置安全性

`num_experts_per_wave` 是 JIT 配置的一部分。为了避免难以控制的 cache 扩张，
生产策略只增加 wave 8 一个候选值，并且仅在 128 tokens/expert 附近的窄区间
使用。wave 12 仍可通过 benchmark-only force override 做实验，但不进入生产
决策。

调试输出的配置 key 已加入 `block_m` 和 `num_experts_per_wave`，设置
`DG_PRINT_CONFIGS=1` 后可以确认实际选择。以下环境变量只用于校准：

- `DG_MEGA_MOE_FORCE_EXPERTS_PER_WAVE`
- `DG_MEGA_MOE_FORCE_BLOCK_M`

非法值会触发断言，而不会静默落入未知配置。

## 7. 正确性验证

`tests/test_mega_moe.py --validate-config-invariance` 在相同输入上比较：

- upstream 默认配置；
- Adaptive Wave Sizing；
- 6 个 forceable block_m；
- 6 个 forceable wave size。

每个配置检查：

- MegaMoE 输出逐位相等；
- cumulative receive stats 更新逐位相等。

最终源码在作业 `2310250` 上重新编译并通过全部配置一致性门。集群中实际参与
编译/测试的 4 个核心源码文件与最终本地提交的 SHA-256 完全一致。

## 8. 性能测试方法

为了避免不同进程启动、JIT warmup、温度和时钟造成的偏差，A/B 在同一进程内
执行，并逐轮反转顺序：

```text
repeat 0: baseline -> adaptive
repeat 1: adaptive -> baseline
repeat 2: baseline -> adaptive
...
```

最终高置信复验每侧执行 6 次 measurement，每次由 30 个 profiled kernel call
构成。分布式系统延迟必须按以下顺序归约：

1. 每个 repetition 先取 8 个 rank 的最大延迟；
2. 再对 6 个 slowest-rank 样本取中位数。

即 `median_repeat(max_rank(latency))`，而不是
`max_rank(median_repeat(latency))`。后者可能在每轮 straggler rank 不同时低估
真实关键路径。`scripts/bench_adaptive_wave_ab.sh` 已实现正确的统计顺序。

## 9. 从宽 gate 到最终策略的收敛

第一轮补充 sweep（作业 `2310224`）覆盖 64～2048 tokens/rank。它发现原 gate
中的 384/high-skew 点并不能从 wave 8 获益：

| 384 tokens/rank | 当时配置 | baseline (µs) | adaptive (µs) | 观测差异 | 获胜次数 |
|---|---:|---:|---:|---:|---:|
| balanced (`alpha=0.0`) | wave 12 | 250.674 | 251.166 | -0.196% | 3/6 |
| moderate (`alpha=1.0`) | upstream wave 16 | 319.840 | 319.942 | -0.032% | 4/6 |
| high skew (`alpha=1.5`) | wave 8 | 306.322 | 310.927 | **-1.481%** | 0/6 |

因此 tokens/expert gate 从 `64.5 < tpe <= 128.5` 收窄到
`127.5 < tpe <= 128.5`。

随后作业 `2310233` 对收窄 gate 做独立复验。384 已正确 fallback；但
512/balanced 的 wave 12 出现 296.317 → 300.148 µs，即 **-1.276%、0/6
获胜**。虽然 wave 12 在此前三轮曾得到正值，这一轮证明它不够稳定，因此也从
生产策略移除。

512/high-skew 的 wave 8 则跨多轮、多个节点始终为正：

| 作业 | baseline (µs) | wave 8 (µs) | 提升 | 获胜次数 |
|---:|---:|---:|---:|---:|
| `2310099` | 352.300 | 350.079 | 0.634% | 8 个 rank 均为正 |
| `2310121` | 344.084 | 339.452 | 1.365% | 6/6 |
| `2310224` | 356.112 | 348.341 | 2.231% | 6/6 |
| `2310233` | 347.093 | 341.694 | 1.580% | 6/6 |
| `2310238` | 352.834 | 345.672 | **2.072%** | 6/6 |
| `2310250` | 352.642 | 348.225 | **1.268%** | 5/6 |

这六轮的提升范围为 0.634%～2.231%，中位数为 1.473%。最终生产策略只保留
这一条稳定分支。

## 10. 常见 tokens/rank 补充测试

表中 64/128/256/1024/2048 来自作业 `2310224`；384/512 来自精确交付源码作业
`2310250`。两者均为 8×B200、6×30、先取每轮全 rank 最大值再跨轮取中位数。

| tokens/rank | expected TPE | skew alpha | 最终配置 | baseline (µs) | adaptive (µs) | 观测差异 | wins |
|---:|---:|---:|---|---:|---:|---:|---:|
| 64 | 16 | 0.0 | upstream | 201.639 | 199.776 | +0.933% | 3/6 |
| 64 | 16 | 1.0 | upstream | 238.095 | 241.044 | -1.223% | 0/6 |
| 64 | 16 | 1.5 | upstream | 217.293 | 215.671 | +0.752% | 5/6 |
| 128 | 32 | 0.0 | upstream | 214.590 | 216.791 | -1.015% | 2/6 |
| 128 | 32 | 1.0 | upstream | 244.868 | 245.055 | -0.077% | 2/6 |
| 128 | 32 | 1.5 | upstream | 227.633 | 228.458 | -0.361% | 2/6 |
| 256 | 64 | 0.0 | upstream | 232.787 | 233.361 | -0.246% | 3/6 |
| 256 | 64 | 1.0 | upstream | 294.130 | 294.957 | -0.280% | 2/6 |
| 256 | 64 | 1.5 | upstream | 304.789 | 305.301 | -0.168% | 1/6 |
| 384 | 96 | 0.0 | upstream | 256.543 | 258.517 | -0.763% | 3/6 |
| 384 | 96 | 1.0 | upstream | 321.395 | 319.239 | +0.676% | 5/6 |
| 384 | 96 | 1.5 | upstream | 312.940 | 316.131 | -1.009% | 0/6 |
| 512 | 128 | 0.0 | upstream | 295.855 | 298.209 | -0.789% | 1/6 |
| 512 | 128 | 1.0 | upstream | 343.657 | 343.517 | +0.041% | 2/6 |
| 512 | 128 | 1.5 | **wave 8** | 352.642 | 348.225 | **+1.268%** | **5/6** |
| 1024 | 256 | 0.0 | upstream | 496.883 | 496.588 | +0.059% | 3/6 |
| 1024 | 256 | 1.0 | upstream | 609.587 | 613.898 | -0.702% | 3/6 |
| 1024 | 256 | 1.5 | upstream | 634.232 | 633.958 | +0.043% | 4/6 |
| 2048 | 512 | 0.0 | upstream | 831.183 | 830.465 | +0.087% | 3/6 |
| 2048 | 512 | 1.0 | upstream | 1234.500 | 1234.000 | +0.041% | 3/6 |
| 2048 | 512 | 1.5 | upstream | 1321.500 | 1320.000 | +0.114% | 3/6 |

除唯一标为 wave 8 的 512/high-skew 行外，表内 baseline/adaptive 选择的
`block_m` 和 wave size 完全相同，因此这些行的正负值是同配置重复测量的噪声，
不能解释为策略收益或回退。它们给出的实测噪声范围约为 -1.23%～+0.93%。
最终 API 还会在校准窗口外跳过 receive-stat D2H，所以这些 fallback 档没有新增
device synchronization。

## 11. 使用方式

启用策略：

```bash
export DG_MEGA_MOE_ADAPTIVE_WAVE=1
```

运行配置一致性：

```bash
python3 tests/test_mega_moe.py \
  --num-processes 8 --num-tokens 256 --num-max-tokens-per-rank 256 \
  --num-experts 256 --num-topk 8 --hidden 7168 --intermediate-hidden 2048 \
  --skew-alpha 1.5 --validate-config-invariance
```

运行 same-process A/B：

```bash
TOKENS_LIST="128 256 384 512 1024" \
ALPHAS="0.0 1.0 1.5" AB_REPEATS=6 AB_NUM_TESTS=30 \
  bash scripts/bench_adaptive_wave_ab.sh
```

运行强制 wave 网格校准：

```bash
bash scripts/bench_mega_moe_wave_size.sh
```

## 12. 适用范围与后续扩展

当前阈值只由以下配置上的 B200 数据支持：256 experts、top-k 8、EP 8、hidden
7168、intermediate 2048、FP8×FP4。其他模型形状仍有 upstream fallback，但不应
直接把本阈值解释为跨模型最优。

后续扩展应继续遵守同一流程：

1. 先用 forced-wave 网格得到候选；
2. 使用真实 receive distribution，而不是只看理论 Zipf 参数；
3. 使用同进程、顺序交替、全 rank 慢值 A/B；
4. 将新策略限制在实测稳定获胜的参数区域；
5. 对未覆盖区域保持 upstream fallback。

不建议重新引入每 launch D2H，也不建议仅依据 padded-row 模型调整 block_m。
