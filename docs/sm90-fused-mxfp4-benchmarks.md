# SM90 fused MXFP4 four-way H20 measurements

Measured on 2026-10-14. Full per-round metrics are in
`benchmarks/sm90-fused-mxfp4-h20.csv`.

The default tables use medians of per-sample minimum rank event times.
Maximum-rank tables include start skew; neither is serving latency.
All runs use Graph replay, seed0, cold L2 (256 MiB), 2M-cycle pad,
50 samples/round and three interleaved rounds (middle order reversed).
Canonical RNE E2M1/K32 quantization is fixed in the harness because
historical tree helpers differ at exact midpoints.
FP8 activation quantization helpers are identical across all trees.
Source BF16 weights and routing use identical per-rank seeds.
#383/#440 use FP8 weights; #411/this PR use MXFP4 weights.
Preprocessing, input staging, and shared experts are excluded.

Flash: H4096/I2048/E32 per GPU/topk6.
Pro: H7168/I3072/E48 per GPU/topk6. M means global active tokens.
EP8 uses cap4096/rank and masked SP padding; fused paths receive
a common row bound. Only EP8 this-PR M1 requests split-K.
Single GPU uses cap=max(128,align(M,128)), no split-K; Flash/Pro
run on separate GPUs concurrently, with the same GPU for all variants
within each shape. Hardware: H20-3e, 78 SMs/GPU, CUDA13, torch2.13.

Frozen baseline commits: #383 `572557e7ae9ad5331b81a1c250f141fba2c57962`,
#411 `5d32216686b982a39eabccca9419af430a60cfc2`,
#440 `8cf8180fe6f97eb65ee6638a14a21497cc1cb5e0`.
Candidate base: nv_dev `b64107f2b9599ca76445b7f62eedf66bae1d095b`.
Baseline tracked kernel/host/Python/test sources were checked against
their Git blobs; existing built baseline extensions were reused.
Candidate uses current DeepJIT; baselines use their legacy JIT.

The 30 candidate cubins from these four suites have maximum STACK 16 B and no CALL.ABS.

**EP8, Flash** (us)

| M | #383 FP8 | #440 FP8 | #411 MXFP4 | This PR MXFP4 | Speedup vs #411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 50.2 | 88.9 | 66.8 | **46.4** | 1.44x |
| 2 | 63.8 | 89.6 | 67.5 | **53.0** | 1.27x |
| 4 | 83.3 | 127.6 | 92.5 | **69.1** | 1.34x |
| 8 | 102.7 | 148.8 | 106.2 | **78.8** | 1.35x |
| 16 | 151.7 | 227.1 | 159.4 | **117.0** | 1.36x |
| 32 | 196.1 | 300.9 | 202.5 | **151.5** | 1.34x |
| 64 | 219.1 | 340.6 | 227.0 | **169.2** | 1.34x |
| 128 | 270.0 | 420.4 | 276.4 | **206.3** | 1.34x |
| 256 | 282.0 | 423.8 | 290.4 | **211.3** | 1.37x |
| 512 | 321.9 | 427.2 | 313.7 | **233.0** | 1.35x |
| 1024 | 412.0 | 429.1 | 376.8 | **270.2** | 1.39x |
| 2048 | 467.6 | 454.8 | 456.2 | **410.7** | 1.11x |
| 4096 | 880.6 | 800.1 | 840.3 | **739.1** | 1.14x |
| 8192 | 1485.5 | 1549.4 | 1445.0 | **1316.9** | 1.10x |

**EP8, Pro** (us)

| M | #383 FP8 | #440 FP8 | #411 MXFP4 | This PR MXFP4 | Speedup vs #411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 98.3 | 122.6 | 98.3 | **83.9** | 1.17x |
| 2 | 110.9 | 152.7 | 114.0 | **99.1** | 1.15x |
| 4 | 133.1 | 191.4 | 135.1 | **117.0** | 1.15x |
| 8 | 230.3 | 316.6 | 213.5 | **184.8** | 1.16x |
| 16 | 346.1 | 505.6 | 323.5 | **280.9** | 1.15x |
| 32 | 507.3 | 782.8 | 482.5 | **424.5** | 1.14x |
| 64 | 693.4 | 1065.2 | 642.5 | **568.2** | 1.13x |
| 128 | 957.8 | 1476.5 | 897.3 | **782.2** | 1.15x |
| 256 | 1024.6 | 1523.1 | 957.1 | **806.7** | 1.19x |
| 512 | 1065.8 | 1530.3 | 989.2 | **832.7** | 1.19x |
| 1024 | 1203.1 | 1539.4 | 1125.4 | **899.7** | 1.25x |
| 2048 | 1603.6 | 1555.3 | 1560.4 | **1085.0** | 1.44x |
| 4096 | 2475.8 | 2964.8 | 2441.8 | **2060.5** | 1.19x |
| 8192 | 3915.4 | 4511.0 | 3803.5 | **3425.6** | 1.11x |

**Single GPU, Flash** (us)

| M | #383 FP8 | #440 FP8 | #411 MXFP4 | This PR MXFP4 | Speedup vs #411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 79.0 | 118.1 | 79.5 | **66.2** | 1.20x |
| 2 | 115.1 | 158.5 | 110.6 | **84.0** | 1.32x |
| 4 | 166.0 | 250.4 | 155.4 | **124.4** | 1.25x |
| 8 | 218.8 | 330.9 | 211.5 | **160.8** | 1.32x |
| 16 | 253.1 | 402.7 | 241.2 | **190.2** | 1.27x |
| 32 | 274.5 | 407.3 | 266.7 | **196.8** | 1.36x |
| 64 | 312.1 | 410.2 | 292.8 | **215.2** | 1.36x |
| 128 | 387.1 | 412.0 | 341.9 | **247.6** | 1.38x |
| 256 | 451.1 | 422.0 | 427.2 | **377.9** | 1.13x |
| 512 | 854.5 | 776.6 | 826.1 | **699.4** | 1.18x |
| 1024 | 1399.6 | 1506.0 | 1375.7 | **1268.4** | 1.08x |
| 2048 | 2575.2 | 2539.8 | 2504.3 | **2339.9** | 1.07x |
| 4096 | 4939.7 | 4874.9 | 4861.2 | **4584.8** | 1.06x |
| 8192 | 9625.7 | 9328.1 | 9575.6 | **9050.8** | 1.06x |

**Single GPU, Pro** (us)

| M | #383 FP8 | #440 FP8 | #411 MXFP4 | This PR MXFP4 | Speedup vs #411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 163.7 | 215.7 | 140.8 | **111.5** | 1.26x |
| 2 | 251.7 | 375.9 | 237.4 | **176.8** | 1.34x |
| 4 | 433.5 | 646.9 | 392.0 | **291.1** | 1.35x |
| 8 | 646.7 | 985.1 | 585.5 | **428.9** | 1.37x |
| 16 | 850.5 | 1308.9 | 788.0 | **563.3** | 1.40x |
| 32 | 979.0 | 1496.5 | 921.9 | **643.6** | 1.43x |
| 64 | 1017.5 | 1502.2 | 955.0 | **668.7** | 1.43x |
| 128 | 1132.3 | 1510.2 | 1092.7 | **754.9** | 1.45x |
| 256 | 1571.6 | 1528.2 | 1540.4 | **945.7** | 1.63x |
| 512 | 2348.3 | 2888.3 | 2280.1 | **1951.5** | 1.17x |
| 1024 | 3773.1 | 4337.0 | 3704.7 | **3482.8** | 1.06x |
| 2048 | 6756.4 | 7086.2 | 6630.1 | **6216.6** | 1.07x |
| 4096 | 12725.8 | 12860.9 | 12596.9 | **11960.8** | 1.05x |
| 8192 | 24868.9 | 24766.9 | 24639.8 | **23700.7** | 1.04x |

## Maximum-rank event intervals

**EP8, Flash** (us)

| M | #383 FP8 | #440 FP8 | #411 MXFP4 | This PR MXFP4 | Speedup vs #411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.1 | 94.4 | 75.0 | **53.0** | 1.42x |
| 2 | 91.7 | 114.8 | 77.6 | **57.1** | 1.36x |
| 4 | 93.2 | 162.1 | 98.5 | **73.7** | 1.34x |
| 8 | 132.2 | 184.3 | 112.2 | **82.7** | 1.36x |
| 16 | 160.6 | 233.7 | 170.0 | **126.3** | 1.35x |
| 32 | 205.6 | 306.7 | 219.9 | **162.0** | 1.36x |
| 64 | 228.0 | 345.0 | 238.4 | **180.3** | 1.32x |
| 128 | 279.4 | 428.6 | 306.7 | **215.1** | 1.43x |
| 256 | 290.5 | 431.0 | 311.4 | **219.9** | 1.42x |
| 512 | 333.3 | 434.8 | 322.3 | **242.8** | 1.33x |
| 1024 | 420.9 | 440.4 | 384.2 | **282.7** | 1.36x |
| 2048 | 477.2 | 464.2 | 463.2 | **428.6** | 1.08x |
| 4096 | 888.8 | 806.8 | 846.1 | **747.4** | 1.13x |
| 8192 | 1495.5 | 1563.6 | 1451.0 | **1326.0** | 1.09x |

**EP8, Pro** (us)

| M | #383 FP8 | #440 FP8 | #411 MXFP4 | This PR MXFP4 | Speedup vs #411 |
|---:|---:|---:|---:|---:|---:|
| 1 | 104.7 | 150.4 | 104.4 | **90.5** | 1.15x |
| 2 | 122.7 | 174.1 | 118.3 | **105.4** | 1.12x |
| 4 | 155.8 | 209.3 | 146.4 | **124.2** | 1.18x |
| 8 | 241.1 | 323.0 | 220.7 | **196.1** | 1.13x |
| 16 | 355.3 | 511.4 | 333.5 | **293.4** | 1.14x |
| 32 | 519.5 | 788.8 | 492.0 | **431.2** | 1.14x |
| 64 | 719.2 | 1080.7 | 651.0 | **573.6** | 1.13x |
| 128 | 969.7 | 1482.8 | 922.4 | **800.4** | 1.15x |
| 256 | 1034.0 | 1529.8 | 964.9 | **814.3** | 1.18x |
| 512 | 1077.1 | 1549.9 | 995.1 | **842.8** | 1.18x |
| 1024 | 1219.8 | 1560.3 | 1136.2 | **910.1** | 1.25x |
| 2048 | 1615.5 | 1564.1 | 1569.4 | **1094.4** | 1.43x |
| 4096 | 2485.7 | 2972.4 | 2447.3 | **2067.9** | 1.18x |
| 8192 | 3936.8 | 4541.7 | 3829.5 | **3437.5** | 1.11x |

## Reproduction

Use `tests/bench_sm90_fused_mega_moe.py` with `DG_ROOT` selecting
each separately built tree and a distinct `DG_JIT_CACHE_DIR`.
Use the four --impl labels pr383_fp8/pr440_fp8/pr411_mxfp4/pr440_mxfp4.
The last label denotes this PR.

```bash
python tests/bench_sm90_fused_mega_moe.py --impl pr440_mxfp4 --np 8 \
  --m 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
  --global-m --graph --bound --split-k --cap 4096 --layout tiled \
  --iters 50 --pad-cycles 2000000 --num-sms 78 --seed 0
```

For Pro add --hidden 7168 --ih 3072 --epr 48. For single GPU use
--np 1 --cap 0 and omit --split-k. Repeat all four variants in
three interleaved rounds. These measurements cover one routing seed;
other routing distributions and H100/H200 remain to be evaluated.
