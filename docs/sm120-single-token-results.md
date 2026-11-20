# Final single-token evidence

Timing: pre-routed expert branch, allocation/JIT/router excluded. Same repaired arithmetic.

| Shape | Condition | Entry latency range (us) | Best ordinary/public range (us) | Entry/control range |
|---|---|---:|---:|---:|
| Qwen3-30B | warm | 26.230–26.490 | 50.214–50.880 | 0.516–0.528 |
| Qwen3-30B | pressure512 | 59.392–59.392 | 86.016–88.064 | 0.674–0.690 |
| Qwen3-235B | warm | 193.277–196.003 | 222.112–225.050 | 0.870–0.871 |
| Qwen3-235B | pressure512 | 223.232–225.280 | 241.664–243.712 | 0.924–0.924 |
| Qwen1.5-MoE | warm | 19.974–20.346 | 38.790–40.282 | 0.505–0.515 |
| Qwen1.5-MoE | pressure512 | 57.344–57.344 | 69.632–71.680 | 0.800–0.824 |
| Qwen2-MoE | warm | 295.130–297.734 | 295.158–297.680 | 1.000–1.000 |
| Qwen2-MoE | pressure512 | 323.584–323.584 | 323.584–323.584 | 1.000–1.000 |
| DeepSeek-V2-Lite | warm | 37.930–39.098 | 64.790–66.362 | 0.585–0.589 |
| DeepSeek-V2-Lite | pressure512 | 79.872–81.920 | 98.304–98.304 | 0.812–0.833 |

Qwen2 is the deliberately rejected fusion boundary: its entry runs ordinary kernels.

| Shape with legacy fusion | Condition | Entry/legacy latency range |
|---|---|---:|
| Qwen3-30B | warm | 1.112–1.140 |
| Qwen3-30B | pressure512 | 0.879–0.879 |
| Qwen3-235B | warm | 0.999–1.000 |
| Qwen3-235B | pressure512 | 1.000–1.000 |

Values above one mean the existing fused configuration was faster in that condition.

Validation: 7 stages passed; real replay: 3072 label/input checks.

All 768 public-entry inputs match ordinary H/Y bitwise. 27 selected inputs passed independent FP64 contract checks. Historical HF inputs outside the original tolerance: 0.

Real replay uses Qwen3 layers 0/4/7 and teacher-forced inputs. It is not full-model serving.

| Real replay boundary | Condition | Median entry/control across recorded cases | Range |
|---|---|---:|---:|
| steady | warm | 0.459 | 0.430–0.503 |
| steady | pressure512 | 0.769 | 0.707–0.833 |
| sequence_average | warm | 0.605 | 0.533–0.717 |
| sequence_average | pressure512 | 0.628 | 0.566–0.722 |

512 MiB interference is a timing condition, not a measured DRAM/cache metric.
Warm samples average ten repeated calls; sequence samples average 32 calls. Neither gives per-request P95.
Single-card/session evidence; no framework, cross-day, or universal correct-rounding claim.

## Evidence identity and reproduction limits

- PR501 base: `87bb4fcc15d4ced85b91649392bc45b727355cf1`.
- Final source archive SHA256: `7da7a3084e14cb7beb454792115de2be496eadd0746b9119c262fcc160d02911`.
- Downloaded final raw evidence SHA256: `a768da9ddcd20e8dfb2db9784f430962d84ca6d9c6d69b8b50a6147ca7d253d5`.
- Native extension SHA256: `5ccdd2925cc5314e44db7b672f1ba25a30b06f7ee63658436c93bf64d1310658`.
- Build reuse was audited across 150 native/build files plus actual dependency tree hashes.
  Only the Python support selection changed between calibration and final execution.
- Four accepted keys retained their configuration before the two final seeds 25301/25309.
  Public-entry performance is separate from calibration seeds 25011/25029 and holdout seeds 25107/25223.
- The original launcher exited 1 when real replay could not import safetensors.
  Its logs are retained. Installing safetensors 0.6.2 in an isolated directory and rerunning only
  replay completed with exit 0. Earlier validation and timing artifacts were not relabeled.
- Production and executed test bodies are unchanged since the final source seal. Subsequent
  edits add reporting, documentation and dependency checks; their source seal is separate.
- Warm steady values average ten repeated calls; pressure steady values time one call.
  A recorded sequence includes GPU copies into stable input/route buffers. Pressure is applied
  before the sequence, not between all 32 recorded calls; these conditions must not be equated.
- Historical HF tolerance success is specific to these 768 T1 inputs and three layers;
  it does not resolve the older T8 SIMT study's separate compatibility failures.

## Measured environment

PyTorch `2.13.0+cu132`, CUDA `13.2`, GPU `NVIDIA GeForce RTX 5090`.

Telemetry recorded with final performance rows:

```text
GPU-936dbb5d-9a2b-e8d2-ae05-fdeec583dda2, NVIDIA GeForce RTX 5090, 610.57.04, 39, 188.31 W, 2782 MHz, 13801 MHz, 2662 MiB, 97 %
```
