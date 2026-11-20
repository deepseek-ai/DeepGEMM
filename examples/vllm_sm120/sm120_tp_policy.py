"""Opt-in TP2 research plans; the frozen public single-token table is unchanged."""
import os

from deep_gemm.small_moe_policy import SmallMoEConfig, select_single_token_config


def select_adapter_config(tokens, experts, topk, hidden, intermediate, tp_size):
    if tp_size == 1:
        return select_single_token_config(tokens, experts, topk, hidden, intermediate)
    if (os.environ.get('SM120_VLLM_TP_ENABLE', '0') != '1' or tp_size != 2
            or (tokens, experts, topk, hidden, intermediate) != (1, 128, 8, 2048, 384)):
        return SmallMoEConfig(reason='outside opt-in TP2 research scope')
    l1 = tuple(int(x) for x in os.environ.get('SM120_TP_L1', '16,3,2').split(','))
    l2 = tuple(int(x) for x in os.environ.get('SM120_TP_L2', '16,3,8').split(','))
    for stage, plan in enumerate((l1, l2)):
        if (len(plan) != 3 or plan[0] not in (8, 16, 32) or plan[1] not in (2, 3, 4)
                or plan[2] not in ((1, 2, 4, 8) if stage == 0 else (8,))):
            raise ValueError('Invalid TP research geometry')
        bn, stages, bundle = plan
        a_bytes = stages*(1 if stage == 0 else bundle)*128
        smem = ((a_bytes+1023)//1024)*1024 + stages*bundle*(2 if stage == 0 else 1)*bn*128
        smem += stages*16 + (0 if stage == 0 else topk*bn*2)
        if smem > 99*1024:
            raise ValueError('TP research geometry exceeds 5090 shared memory')
    return SmallMoEConfig(l1, l2, 'opt-in TP2 local-shard research plan')
