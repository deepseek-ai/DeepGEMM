"""SM90 fused MegaMoE ring-buffer check.

For every shape the symm buffers of three layouts must produce the same output bit for bit -- the default layout (a lag-scheduled
ring from 1024 tokens per rank, the full pool below that), a ring sized for one automatically chosen expert wave
(``num_experts_per_wave=-1``) and, where the default is a ring, the full pool under the wave schedule
(``num_experts_per_wave=num_experts_per_rank``, the legacy layout) -- and the first launch must match the reference of
test_mega_moe_sm90.py within its tolerance. Every buffer runs four launches: the full routing, a smaller token count with a
skewed routing, the full routing again, and a decode-sized one, so both launch-parity banks and the ring wrap are exercised on one
allocation. The two-rank shape is the configuration whose requested lag ring exceeds the full pool (the buffer is the pool itself
under the lag schedule). A fourth buffer of the default layout then repeats all four launches with each call's own
``num_tokens_bound`` instead of the buffer capacity, which must not change a single bit either: on a lag ring that holds one wave
of that bound, the decode-sized launch runs the wave order under the bound where it runs the lag order without one, and on the
top-6 shape the smaller launch runs a shorter lag under its bound. A bound above the capacity must be rejected instead.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist
from deep_gemm.testing import calc_diff, get_arch_major

from test_mega_moe_sm90 import _quantize_grouped_fp8_block_128_128, _reference_fused

ACTIVATION_CLAMP = 10.0
# tokens per rank from which the default layout is a lag ring (heuristics kSm90FusedAutoLagMinTokens)
AUTO_LAG_MIN_TOKENS = 1024
# per-rank token bound up to which a call on a lag ring takes the wave schedule instead (heuristics kSm90FusedCallWaveMaxTokens)
CALL_WAVE_MAX_TOKENS = 256
TOKEN_ALIGNMENT = 128


def _shapes(num_ranks):
    out = [
        (f'ring.h1024.t{tokens}', dict(hidden=1024, intermediate_hidden=1024, num_experts=8 * num_ranks, num_topk=2, num_tokens=tokens))
        for tokens in (64, 256, 512, 1024, 2048)
    ]
    out.append(('ring.h4096.experts288.t2048', dict(hidden=4096, intermediate_hidden=2048, num_experts=288, num_topk=8, num_tokens=2048)))
    # top-6 at a 5120-token capacity: the buffer's lag spans four 1024-token batches per local expert and the smaller launch's
    # own bound spans three, so here a bound shortens the lag instead of switching the order, which no other shape's bound does
    out.append(('ring.h1024.topk6.t5120', dict(hidden=1024, intermediate_hidden=1024, num_experts=8 * num_ranks, num_topk=6, num_tokens=5120)))
    # two ranks, 8 local experts: the sizer asks for a 12800-token lag ring, the full pool is 9216 tokens -> clamped to the pool
    out.append(('ring.h1024.t2048.ep2', dict(hidden=1024, intermediate_hidden=1024, num_experts=16, num_topk=2, num_tokens=2048, world=2)))
    return out


def _align(x, a):
    return (x + a - 1) // a * a


def _full_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    # layout::get_num_max_pool_tokens_sm90 (one partial block of kSM90FusedMaxCandidateBlockM 128 per expert, kSM90FusedLCMBlockM 128)
    return _align(num_ranks * num_max_tokens_per_rank * min(num_topk, num_experts_per_rank) + num_experts_per_rank * 127, 128)


def _routing(num_tokens, num_experts, num_topk, seed, num_hot_experts=0):
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda', generator=gen)
    if num_hot_experts:
        # skewed routing: a few experts take most rows, the others a handful
        scores[:, :num_hot_experts] += 4.0
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    return topk_idx, topk_weights


def _run_shape(name, cfg, rank_idx, num_ranks, group, diff_tol, l2_act_sf_gran_k):
    """Runs one shape on `group`; returns (ok, message). No assertion and no collective outside `group`."""
    hidden, intermediate_hidden = cfg['hidden'], cfg['intermediate_hidden']
    num_experts, num_topk, num_tokens = cfg['num_experts'], cfg['num_topk'], cfg['num_tokens']
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = _align(num_tokens, TOKEN_ALIGNMENT)
    seed = rank_idx * 1000 + sum(map(ord, name))
    torch.manual_seed(seed)

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda') * 0.05
    l2_weights_bf16 = torch.randn((num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda') * 0.05
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_weights_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_weights_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1_weights, l2_weights)
    del l1_weights_bf16, l2_weights_bf16

    # launch 1 and 3: every token, the same routing (3 runs on the parity bank of launch 1 again); launch 2: fewer tokens, skewed
    # routing (the smaller call may select a different tile class on the same buffer; the checks do not depend on it)
    launches = [(num_tokens,) + _routing(num_tokens, num_experts, num_topk, seed)]
    num_tokens_2 = max(1, num_tokens * 3 // 4)
    launches.append((num_tokens_2,) + _routing(num_tokens_2, num_experts, num_topk, seed + 1, num_hot_experts=2))
    launches.append(launches[0])
    num_tokens_4 = min(CALL_WAVE_MAX_TOKENS, num_tokens)
    launches.append((num_tokens_4,) + _routing(num_tokens_4, num_experts, num_topk, seed + 2))

    # full pool -> the default layout is a lag ring (or the pool itself when the ring would exceed it), else the pool
    full_pool = _full_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank)
    default_is_ring = num_max_tokens_per_rank >= AUTO_LAG_MIN_TOKENS
    arms = (None, -1, num_experts_per_rank) if default_is_ring else (None, -1)

    def run_launches(buffer, bounds):
        ys = []
        for (n, topk_idx, topk_weights), bound in zip(launches, bounds):
            buffer.x[:n].copy_(x_fp8[0][:n])
            buffer.x_sf[:n].copy_(x_fp8[1][:n])
            buffer.topk_idx[:n].copy_(topk_idx)
            buffer.topk_weights[:n].copy_(topk_weights)
            y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_mega_moe(y, transformed_l1, transformed_l2, buffer, recipe=(128, 128, 128), activation='swiglu',
                                   activation_clamp=ACTIVATION_CLAMP, fast_math=True, num_tokens_bound=bound)
            torch.cuda.synchronize()
            ys.append(y)
        return ys

    outputs, caps = {}, {}
    for num_experts_per_wave in arms:
        buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
            group, num_experts, num_tokens, num_topk, hidden, intermediate_hidden,
            fused=True, num_experts_per_wave=num_experts_per_wave, l2_act_sf_gran_k=l2_act_sf_gran_k)
        outputs[num_experts_per_wave] = run_launches(buffer, [None] * len(launches))
        # the buffer reports the full pool as a ring capacity of 0
        caps[num_experts_per_wave] = (buffer.num_ring_tokens or None, buffer.l2_lag_encoded, buffer.l2_act_sf_gran_k)
        buffer.destroy()
        dist.barrier(group=group)

    # the default layout again, this time telling the kernel each call's own bound instead of letting it assume the capacity
    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group, num_experts, num_tokens, num_topk, hidden, intermediate_hidden,
        fused=True, l2_act_sf_gran_k=l2_act_sf_gran_k)
    bounds = [_align(n, TOKEN_ALIGNMENT) for n, _, _ in launches]
    bounded_outputs = run_launches(buffer, bounds)
    # a bound the buffer cannot receive is rejected on the host, before any rank launches
    try:
        run_launches(buffer, [num_max_tokens_per_rank + TOKEN_ALIGNMENT] * len(launches))
        rejected = False
    except Exception:  # noqa: BLE001 -- the host assertion arrives as whatever the FFI layer wraps it in
        rejected = True
    buffer.destroy()
    dist.barrier(group=group)

    # capacities: the default arm is a lag ring (at most the pool) or the pool, the auto-wave arm a ring, the control arm the pool
    ring_default, lag_default, gran_k = caps[None]
    ring_auto = caps[-1][0]
    problems = []
    if default_is_ring:
        if ring_default is None or ring_default > full_pool or lag_default <= 0:
            problems.append(f'default layout ring={ring_default} lag={lag_default} (expected a lag ring <= pool {full_pool})')
        if caps[num_experts_per_rank][0] != full_pool or caps[num_experts_per_rank][1] != 0:
            problems.append(f'full-pool control ring={caps[num_experts_per_rank][0]} lag={caps[num_experts_per_rank][1]} (expected pool {full_pool}, no lag)')
    elif ring_default is not None:
        problems.append(f'default layout ring={ring_default} (expected the full pool below {AUTO_LAG_MIN_TOKENS} tokens)')
    if ring_auto is None or ring_auto > full_pool:
        problems.append(f'auto-wave ring={ring_auto} (expected a ring <= pool {full_pool})')

    # every arm bit-identical to the default arm on every launch; launch 3 bit-identical to launch 1 within every arm
    bitwise = all(torch.equal(outputs[arm][i], outputs[None][i]) for arm in arms[1:] for i in range(len(launches)))
    replay = all(torch.equal(outputs[arm][0], outputs[arm][2]) for arm in arms)
    bounded = all(torch.equal(bounded_outputs[i], outputs[None][i]) for i in range(len(launches)))
    topk_idx_1, topk_weights_1 = launches[0][1], launches[0][2]
    y_ref = _reference_fused(
        x_fp8[0], x_fp8[1], topk_idx_1, topk_weights_1,
        l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        rank_idx, num_ranks, group, num_experts, num_topk, hidden, intermediate_hidden, ACTIVATION_CLAMP, gran_k)
    diff = calc_diff(outputs[None][0], y_ref)
    ok = bitwise and replay and bounded and rejected and diff < diff_tol and not problems
    control = f' pool={caps[num_experts_per_rank][0]}' if default_is_ring else ''
    msg = (f"ring={ring_auto} default={ring_default or 'full'}{control} lag={lag_default} sf_gran_k={gran_k} "
           f"bitwise={'OK' if bitwise else 'FAIL'} replay={'OK' if replay else 'FAIL'} "
           f"bounds={bounds}<=cap{num_max_tokens_per_rank}:{'OK' if bounded else 'FAIL'} "
           f"over_cap_rejected={'OK' if rejected else 'FAIL'} diff={diff:.4f} (tol={diff_tol:.2f})"
           + (' ' + '; '.join(problems) if problems else ''))
    return ok, msg


def test(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print(f'[SKIP] test_mega_moe_sm90_fused_ring requires SM90; got SM{get_arch_major()}0', once_in_node=True)
        dist.destroy_process_group()
        return

    # the two-rank shapes run on ranks 0 and 1 (every rank takes part in creating the group)
    group_2 = dist.new_group(ranks=[0, 1]) if num_ranks > 2 else None

    failures = []
    for name, cfg in _shapes(num_ranks):
        world = cfg.get('world', num_ranks)
        if world > num_ranks or cfg['num_experts'] % world != 0:
            dist_print(f"  [{name:<24}] SKIP ({cfg['num_experts']} experts on {world} ranks, {num_ranks} available)", once_in_node=True)
            continue
        try:
            if world == num_ranks:
                ok, msg = _run_shape(name, cfg, rank_idx, num_ranks, group, args.diff_tol, args.l2_act_sf_gran_k)
            elif rank_idx < world:
                ok, msg = _run_shape(name, cfg, rank_idx, world, group_2, args.diff_tol, args.l2_act_sf_gran_k)
            else:
                ok, msg = True, ''
        except Exception as ex:   # noqa: BLE001 -- a shape that raises on every rank is reported and the loop goes on; a raise on one rank alone still hangs on the collectives it skipped, as it would without the handler
            ok, msg = False, f'exception: {ex!r}'
        # one verdict for all ranks before anyone prints or moves on: a rank-local failure must not skip a collective
        flag = torch.tensor([0 if ok else 1], dtype=torch.int32, device='cuda')
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        messages = [None] * num_ranks
        dist.all_gather_object(messages, msg if not ok else '')
        ok_all = flag.item() == 0
        failed_ranks = [r for r, m in enumerate(messages) if m]
        dist_print(f"  [{name:<24}] {msg if rank_idx == 0 and msg else ''} {'OK' if ok_all else 'FAIL'}"
                   + (f' ranks {failed_ranks}: {messages[failed_ranks[0]]}' if failed_ranks else ''), once_in_node=True)
        if not ok_all:
            failures.append(name)
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM90 fused MegaMoE ring-buffer bitwise check')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of spawned processes, one per GPU')
    parser.add_argument('--diff-tol', type=float, default=0.07, help='calc_diff tolerance against the reference; default: 0.07')
    parser.add_argument(
        '--l2-act-sf-gran-k', type=int, choices=[64, 128], default=None,
        help='L2 activation-scale K granularity the symm buffers are built with; default: the package default',
    )
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
