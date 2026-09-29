"""MoE kernel constants for Step3.5 prefill-side @pl.jit.inline bodies.

Shared by the 5 shim files (prefill_gate / prefill_dispatch /
prefill_expert_routed / prefill_expert_shared / prefill_combine) and by
``prefill_moe.py``. Centralised here so the shim files and
``prefill_moe.py`` can both import them without a circular dependency
(after the #9 refactor, ``prefill_moe.py`` will pl.inline the shim
bodies, creating a two-way import — the constants must live in a third
module to break the cycle).

Mirrors the decode-side constants in ``gate.py`` / ``expert_routed.py``
/ ``expert_shared.py`` / ``moe.py``, but uses prefill-side names
(BATCH, ROUTER_SCORE_PAD, ROUTER_GATE_K_CHUNK, etc.).
"""

from __future__ import annotations

from .config import (
    BATCH,
    HIDDEN,
    LAYER_DYN,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_ROUTER_SCALING_FACTOR,
    MOE_TOP_K,
    SHARE_EXPERT_DIM_LOCAL,
    TP_WORLD_SIZE,
)
from .dispatch import (
    LOCAL_RECV_MAX as DISPATCH_LOCAL_RECV_MAX,
)
from .dispatch import (
    N_RANKS_PAD as DISPATCH_N_RANKS_PAD,
)
from .dispatch import (
    PER_RANK_BUCKETS as DISPATCH_PER_RANK_BUCKETS,
)

# -----------------------------------------------------------------------------
# Re-exports of config-level MoE topology (kept here so shim bodies have a
# single import site for all MoE constants).
# -----------------------------------------------------------------------------
N_RANKS = TP_WORLD_SIZE
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
N_EXPERTS = MOE_NUM_EXPERTS
TOPK = MOE_TOP_K
INTER = MOE_INTERMEDIATE
SH_INTER_LOCAL = SHARE_EXPERT_DIM_LOCAL
LOCAL_RECV_MAX = DISPATCH_LOCAL_RECV_MAX
PER_RANK_BUCKETS = DISPATCH_PER_RANK_BUCKETS
N_RANKS_PAD = DISPATCH_N_RANKS_PAD
N_ROUTES_PER_RANK = BATCH * TOPK
SH_TP_CHUNK = HIDDEN // TP_WORLD_SIZE
# Per-layer RMS weight stack height (post_rms_weight [LAYER_DYN, HIDDEN]).
# Re-exported so the gate shim (prefill_gate.py) can type its
# ``post_rms_weight`` parameter from the single MoE-constants import site.
LAYER_DYN = LAYER_DYN

# Per-token INT8 activation dequant scale window width (W8A8 dispatch).
# Padded to 8 FP32 cols (=32B, the min UB/GM tile row) so the scale a2a
# window's pl.load / remote_load tiles are 32B-aligned — mirrors
# decode-side moe.py SCALE_W_PAD (DeepSeek v4 dispatch W_PAD). Column 0
# holds the scale; columns 1..7 are zero pad.
SCALE_W_PAD = 8

# -----------------------------------------------------------------------------
# Router (gate) kernel constants — mirrors gate.py / moe.ROUTER_*.
# -----------------------------------------------------------------------------
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
# K_CHUNK=256 (was 512): gate_w tile [256, N_EXPERTS=288] FP32 = 294KB
# fits L1 (512KB); the 512 variant overflowed L1 (575KB w tile alone).
# Math unchanged — kb loop still covers full HIDDEN=4096 (16 iters at
# 256 vs 8 iters at 512). Mirrors the L1-fit constraint that
# _compile_prefill_layer_dense.py is deferred to Phase 17 for (192KB
# UB overflow). Decode-side gate.py keeps its own GATE_K_CHUNK=512
# (not touched per directive).
ROUTER_GATE_K_CHUNK = 256
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = MOE_ROUTER_SCALING_FACTOR
assert TOPK <= ROUTER_TOPK_PAD
assert HIDDEN % ROUTER_GATE_K_CHUNK == 0

# -----------------------------------------------------------------------------
# Routed-expert kernel constants — mirrors expert_routed.py / moe.ROUTED_*.
# -----------------------------------------------------------------------------
ROUTED_GATE_K_CHUNK = 128
ROUTED_GATE_N_CHUNK = 64
ROUTED_H_QUANT_N_CHUNK = 256
ROUTED_DOWN_K_CHUNK = 128
ROUTED_DOWN_N_CHUNK = 64
ROUTED_MAX_TILE = LOCAL_RECV_MAX
# K1 row tiling (task #25): ROUTED_MAX_TILE=1024 rows makes h_tile
# [1024, 1280] FP32 = 5MB overflow UB (184KB). Row-tile to 16 rows:
# h_tile [16, 1280] FP32 = 81920B = 80KB; with the h_bf16 cast
# (also alive in moe_down, [16, 1280] BF16 = 40KB) + weight
# reshape copies ([128, 64] BF16 = 16KB each), moe_down Vec total
# = 160KB < 184KB; moe_gate_up Vec = 96KB. Math unchanged -- the
# rt loop splits 1024 rows into 64 row-tiles of 16 rows each.
# Mirrors decode-side expert_routed.py (h_tile FP32 + cast to BF16
# before down matmul).
#
# PL's directive was ROW_TILE=32, but ROW_TILE=32 with FP32 h_tile
# overflows moe_down (h_tile 160KB + h_bf16 80KB = 240KB > 184KB --
# both alive because h_bf16 = cast(h_tile) reads h_tile). ROW_TILE=16
# is the largest power-of-2 that fits with FP32 h_tile + h_bf16 cast.
#
# K_CHUNK=128 (was 256): the 3D-slice + reshape weight path creates
# a copy in Vec ([K_CHUNK, N_CHUNK] BF16). K_CHUNK=256 made this
# 32KB per tile; K_CHUNK=128 halves it to 16KB, recovering the Vec
# budget that the h_tile + h_bf16 pair consumes. Math unchanged --
# the kb loop still covers full HIDDEN=4096 (32 iters at 128 vs
# 16 iters at 256).
#
# tile_valid clamp (task #25 runtime fix): when n_rows < rt (e.g.
# count=8, rt=16 -> tile_valid=-8), the negative valid_shape stalls
# the cube unit (S1:running-stalled). Scalar pl.maximum doesn't
# exist; the clamp uses x * cast(x > 0, INDEX) == max(x, 0).
# Exposed by K1 (UB fix) -- previously masked by the Vec overflow
# that prevented runtime.
ROUTED_ROW_TILE = 16
assert HIDDEN % ROUTED_GATE_K_CHUNK == 0
assert HIDDEN % ROUTED_DOWN_N_CHUNK == 0
assert INTER % ROUTED_GATE_N_CHUNK == 0
assert INTER % ROUTED_H_QUANT_N_CHUNK == 0
assert INTER % ROUTED_DOWN_K_CHUNK == 0
assert ROUTED_MAX_TILE % ROUTED_ROW_TILE == 0

# -----------------------------------------------------------------------------
# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
# -----------------------------------------------------------------------------
SHARED_GATE_K_CHUNK = 256
SHARED_DOWN_N_CHUNK = 256
# Narrow swiglu N-chunk for the shared-expert gate/up + down projection: the
# full [BATCH,160] Vec-tile cast/clamp is miscompiled (wide-tile tmov misprune
# -> ~45% wrong) because 160 crosses a 128-column block boundary. Compute 5
# narrow [BATCH,32] chunks (5 * 32 = 160) instead, mirroring the working
# routed path and the decode-side moe.py SHARED_SWIGLU_N_CHUNK fix.
SHARED_SWIGLU_N_CHUNK = 32
assert SH_INTER_LOCAL == SHARED_SWIGLU_N_CHUNK * 5
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0

# -----------------------------------------------------------------------------
# Swiglu limits (compile-time constants for the activation split).
# Lifted from prefill_moe.py's _routed_swiglu_limit / _shared_swiglu_limit
# closure variables. The shim bodies split into silu / swiglu7 / swiglu16
# variants at factory build time (each variant is a separate @pl.jit.inline
# body), replacing the runtime ``if _routed_swiglu_step`` branch.
# -----------------------------------------------------------------------------
ROUTED_SWIGLU_LIMIT = 7.0
SHARED_SWIGLU_LIMIT = 16.0
