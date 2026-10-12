import pypto.language as pl
from .vision_config import V_DS2_OUT, V_TOKENS_FINAL_PAD, LM_HIDDEN
IN = V_DS2_OUT; OUT = LM_HIDDEN  # 6144, 4096
N = V_TOKENS_FINAL_PAD          # 176 (169 zero-padded to 16-align; see vision_config)

@pl.jit.inline
def vit_projector(
    x: pl.Tensor[[N, IN], pl.BF16],
    w: pl.Tensor[[IN, OUT], pl.BF16],
    out: pl.Tensor[[N, OUT], pl.BF16],
):
    # Tiling: 64 N-block tasks (OUT//64), each owns one [IN,64] weight slice and
    # loops every row group. Rows are split into M=32 tiles with a 2-way unroll
    # (shared weight slice feeds two adjacent tiles) plus static M=16 tails
    # (176 = 2*64 + 32 + 16). K is chunked at 512 with a stage=2 pipeline over
    # the chunk loop. None of these ever reorder a row's K accumulation, so the
    # result stays bit-exact vs a single monolithic matmul.
    for nb_idx in pl.spmd(64, name_hint="vit_proj"):     # OUT // 64
        n0 = nb_idx * 64
        for tg_idx in pl.range(2):
            tg = tg_idx * 64
            acc = pl.matmul(x[tg : tg + 32, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
            acc2 = pl.matmul(x[tg + 32 : tg + 64, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
            for kb in pl.pipeline(11, stage=2):          # chunks 1..11
                k0 = (kb + 1) * 512
                wt = w[k0:k0 + 512, n0:n0 + 64]          # loaded once, used by both tiles
                acc = pl.matmul_acc(acc, x[tg : tg + 32, k0:k0 + 512], wt)
                acc2 = pl.matmul_acc(acc2, x[tg + 32 : tg + 64, k0:k0 + 512], wt)
            out[tg : tg + 32, n0 : n0 + 64] = pl.cast(acc, target_type=pl.BF16, mode="rint")
            out[tg + 32 : tg + 64, n0 : n0 + 64] = pl.cast(acc2, target_type=pl.BF16, mode="rint")
        # tail rows 128..159 (unpaired M=32; separate names: DSL forbids
        # reassigning a name with a different tensor shape)
        acc_u = pl.matmul(x[128:160, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
        for kb_u in pl.pipeline(11, stage=2):            # chunks 1..11
            k0_u = (kb_u + 1) * 512
            acc_u = pl.matmul_acc(acc_u, x[128:160, k0_u:k0_u + 512], w[k0_u:k0_u + 512, n0:n0 + 64])
        out[128:160, n0 : n0 + 64] = pl.cast(acc_u, target_type=pl.BF16, mode="rint")
        # tail rows 160..175 (M=16; separate names: DSL forbids reassigning a
        # name with a different tensor shape)
        acc_t = pl.matmul(x[160:176, 0:512], w[0:512, n0:n0 + 64], out_dtype=pl.FP32)
        for kb_t in pl.pipeline(11, stage=2):            # chunks 1..11
            k0_t = (kb_t + 1) * 512
            acc_t = pl.matmul_acc(acc_t, x[160:176, k0_t:k0_t + 512], w[k0_t:k0_t + 512, n0:n0 + 64])
        out[160:176, n0 : n0 + 64] = pl.cast(acc_t, target_type=pl.BF16, mode="rint")
    return out


# ── torch golden reference (consumed by tests/step3p7/unit/test_vit_projector.py) ─
def golden_vit_projector(x, w):
    import torch
    return (x.float() @ w.float()).to(torch.bfloat16)     # [N, 4096]
