"""Exercise fused and existing non-fused FP8 against the upstream reference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import deep_gemm
import test_mega_moe_sm90 as reference
from deep_gemm.utils.dist import init_dist, dist_print


def worker(local_rank, nr, args):
    rank, nr, group = init_dist(local_rank, nr)
    factory = deep_gemm.get_symm_buffer_for_sm90_mega_moe
    if args.fused:
        def fused_factory(*a, **kw):
            return factory(*a, **kw, fused=True)
        deep_gemm.get_symm_buffer_for_sm90_mega_moe = fused_factory
    for m in args.m:
        for fm in (True, False):
            cfg = dict(num_max_tokens_per_rank=max(m, 128), num_tokens=m,
                       hidden=args.hidden, intermediate_hidden=args.ih,
                       num_experts=args.experts_per_rank * nr, num_topk=6,
                       fast_math=fm)
            reference._run_scenario(f'fp8.fused{args.fused}.m{m}.fm{int(fm)}', cfg,
                                    rank, nr, group, 0.01)
    dist_print('FP8 SUMMARY fails=[]', once_in_node=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--np', type=int, default=1)
    p.add_argument('--m', type=int, nargs='+', default=[1, 64, 256, 1024])
    p.add_argument('--hidden', type=int, default=4096)
    p.add_argument('--ih', type=int, default=2048)
    p.add_argument('--experts-per-rank', type=int, default=32)
    p.add_argument('--fused', action='store_true')
    args = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.np, args), nprocs=args.np)
