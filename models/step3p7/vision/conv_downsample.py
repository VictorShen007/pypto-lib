import pypto.language as pl
from .vision_config import V_DS1_OUT, V_DS2_OUT, V_GRID_DS1, V_TOKENS_FINAL, V_WIDTH

@pl.jit.inline
def conv_downsample1(
    im2col: pl.Tensor[[V_GRID_DS1*V_GRID_DS1, 9*V_WIDTH], pl.BF16],
    w: pl.Tensor[[9*V_WIDTH, V_DS1_OUT], pl.BF16],
    out: pl.Tensor[[V_GRID_DS1*V_GRID_DS1, V_DS1_OUT], pl.BF16],
):
    n = (V_GRID_DS1 * V_GRID_DS1) // 16; KC = 512
    for tg_idx in pl.spmd(n, name_hint="conv_ds1"):
        tg = tg_idx * 16
        for nb in pl.range(V_DS1_OUT // 128):
            n0 = nb * 128
            acc = pl.matmul(im2col[tg : tg + 16, 0:KC], w[0:KC, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.range(1, 13824 // KC):
                k0 = kb * KC
                acc = pl.matmul_acc(acc, im2col[tg : tg + 16, k0:k0 + KC], w[k0:k0 + KC, n0:n0 + 128])
            out[tg : tg + 16, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
    return out

@pl.jit.inline
def conv_downsample2(
    im2col: pl.Tensor[[V_TOKENS_FINAL, 9*V_DS1_OUT], pl.BF16],
    w: pl.Tensor[[9*V_DS1_OUT, V_DS2_OUT], pl.BF16],
    out: pl.Tensor[[V_TOKENS_FINAL, V_DS2_OUT], pl.BF16],
):
    n = V_TOKENS_FINAL // 16; KC = 512
    for tg_idx in pl.spmd(n, name_hint="conv_ds2"):
        tg = tg_idx * 16
        for nb in pl.range(V_DS2_OUT // 128):
            n0 = nb * 128
            acc = pl.matmul(im2col[tg : tg + 16, 0:KC], w[0:KC, n0:n0 + 128], out_dtype=pl.FP32)
            for kb in pl.range(1, 27648 // KC):
                k0 = kb * KC
                acc = pl.matmul_acc(acc, im2col[tg : tg + 16, k0:k0 + KC], w[k0:k0 + KC, n0:n0 + 128])
            out[tg : tg + 16, n0 : n0 + 128] = pl.cast(acc, target_type=pl.BF16, mode="rint")
    return out


# ── torch golden references (consumed by tests/step3p7/unit/test_conv_downsample.py) ─
# Both downsamplers are im2col(@host) -> matmul; golden is the same matmul in FP32.
def golden_conv_downsample1(im2col, w):
    import torch
    return (im2col.float() @ w.float()).to(torch.bfloat16)   # [V_GRID_DS1^2, V_DS1_OUT]


def golden_conv_downsample2(im2col, w):
    import torch
    return (im2col.float() @ w.float()).to(torch.bfloat16)   # [V_TOKENS_FINAL, V_DS2_OUT]
