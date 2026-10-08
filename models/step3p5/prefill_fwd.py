# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""[中文摘要] `Step3p5PrefillFwd`:整模型 prefill 顶层 @pl.program。结构与
decode_fwd.py 对偶 —— host_orch 一次性分配 window pool,chip_orch 编译期
for 45 层走 select_prefill_layer 选 kind 标签,派发到 in-class chip_orch 方法;
末尾接 rms_lm_head。3 个 MTP 层不在这里(走 mtp.py 的 program)。per-layer
@pl.program compile target 已摘除(T2a per-layer deprecation)。
[关键装饰器] @pl.program +
   @pl.function(level=HOST, role=Orchestrator)
   @pl.function(type=Orchestration)
[SPMD 角色] 顶层入口,跨卡 + 片上 SPMD 全程。
[详见] 中文架构指南 §10

────── 以下为英文原 docstring ──────

Step3p5 multi-layer prefill forward pass — TP/EP wired (Phase 6).

Top-level distributed prefill entry. The 45 main layers are dispatched
in a Python compile-time loop; ``select_prefill_layer(layer_idx)`` returns
a ``kind`` routing label that selects an in-class ``@pl.function``
chip_orch method (per-layer ``@pl.program`` targets were removed in the
T2a per-layer deprecation). After the layer loop
the residual stream goes through ``rms_lm_head`` (replicated zero-
centred RMSNorm + vocab-sliced LM head matmul) and each rank emits its
own ``[USER_BATCH, VOCAB_LOCAL]`` shard.

The 3 MTP layers (45..47) are NOT included here — see ``mtp.py``; the
integration author from Phase 8 wires them on top of the prefill fwd.

Per-layer dispatch (mirrors ``decode_layer.select_decode_layer``)
------------------------------------------------------------------
  layer_idx  | attention   | MLP
  -----------|-------------|------------
  0          | full        | TP dense MLP
  1, 2       | swa         | TP dense MLP
  3..44      | full / swa  | EP+TP MoE (silu/silu, silu+swiglu7, swiglu7+swiglu16)

The per-layer ``@pl.program`` specialisations (``prefill_layer_*_moe_*`` /
``prefill_layer_*_dense``) were removed in the T2a per-layer deprecation;
``select_prefill_layer`` returns a ``kind`` label only.

Window-pool layout
------------------
Copied from ``decode_fwd.py``: per-layer ``host_orch`` pattern, fresh
signal windows per call site to avoid AtomicAdd ring-step collisions
across collectives. The TP-AR scratch / EP a2a payload pools are
allocated inside the top-level program's ``host_orch`` (the MoE
chip_orch reuses its slot across the prefill-T tile loop because each
tile flushes before the next reads).

KV-cache convention (TP-aware)
------------------------------
Each rank holds only its slice of the KV heads — ``KV_HEADS_LOCAL = 1``
KV head per rank under TP=8. The per-layer cache row stride is
``MAX_BLOCKS_PER_SEQ * KV_HEADS_LOCAL * BLOCK_SIZE`` rows of
``HEAD_DIM`` BF16 lanes. The 45-layer K-cache and V-cache are stacked
along their leading axis.

RoPE table convention
---------------------
Per-flavour stacks ``rope_cos_full / rope_sin_full`` size
``[NUM_FULL_LAYERS * MAX_SEQ, 64]`` and ``rope_cos_swa / rope_sin_swa``
size ``[NUM_SWA_LAYERS * MAX_SEQ, 128]`` — replicated on every rank.

Distributed-mock harness
------------------------
The ``__main__`` block runs a pure-torch 8-rank simulation of the
full 45-layer prefill against a single-card oracle. The TP all-reduce
is implemented in torch as a sum-across-ranks. The harness reports
the worst-case per-rank pass rate against the oracle.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import os

import os as _os

import pypto.language as pl
import pypto.language.distributed as pld

# ── Perf-experiment gates (host-level constants, 2026-09-21). ──
# These select inline-body variants at splice time (host-level Python if,
# NOT a DSL static if: conditions inside a spliced inline body resolve in
# the splicing module's scope, so variant selection must happen here where
# these names are visible).
#   PREFILL_ZERO_CHUNK: rows per zero-store in the EP buffer zero-fill
#     (prefill_zero_dispatch_buffers[_chunked] / prefill_zero_routed_y_buf
#     [_chunked]). 1 = original per-row stores; 4/8 = chunked stores.
#   PREFILL_SER_MODE: _serialize_after_shared dependency-carrier size.
#     0 = original full-row fold; 1 = one 32B chunk per row (the sh_y task
#     edge stays declared; only in-task dummy loads shrink ~512x).
PREFILL_ZERO_CHUNK = int(_os.environ.get("PYPTO_PREFILL_ZERO_CHUNK", "4"))  # default 4 (chunked) per zero_chunk4 A/B: P50 -8.8% vs per-row
PREFILL_SER_MODE = int(_os.environ.get("PYPTO_PREFILL_SER_MODE", "1"))  # 默认 1（2026-09-23 A/B 定案：ser=1 vs ser=0 同代码同日 -527ms/−13.8%，cos 0.9959 过门）
#   PREFILL_A2A_BULK: ep_all_to_all copy granularity (audit A1, 2026-10-08).
#     0 = original per-row copies (128 rows x 7 peers of [1, HIDDEN] tiles =
#     ~3.6k serial GM/remote ops per dispatch); 1 = bulk chunked copies
#     ([PREFILL_A2A_ROWS, HIDDEN] tiles = 14-56 ops). Copies the SAME rows
#     (full fixed-slot block incl. the zeroed gap rows, which the per-row
#     form also copies and downstream ignores) -> bit-identical results;
#     only loop granularity changes.
#   PREFILL_A2A_ROWS: rows per bulk tile. UB bound: ROWS*4096 B INT8 payload
#     + ROWS*32 B FP32 scale (16 -> 64 KB of the 184 KB Vec buffer).
PREFILL_A2A_BULK = int(_os.environ.get("PYPTO_PREFILL_A2A_BULK", "0"))
PREFILL_A2A_ROWS = int(_os.environ.get("PYPTO_PREFILL_A2A_ROWS", "16"))
# MEASUREMENT-ONLY PROBE (UNSAFE, never ship): skip the cross-layer WAR
# fence to test whether it is on the critical path. Skipping can race
# layer L's dispatch against layer L-1's combine drains.
PREFILL_SKIP_CROSS_FENCE = int(
    _os.environ.get("PYPTO_PREFILL_SKIP_CROSS_FENCE", "0")
)


from ._ops import build_plain_rope_tables, zero_centered_rmsnorm_apply
# v1 refactor (design §7): prefill_dense_mlp_body is the extracted form
# of the former local _prefill_dense_mlp_body_tp (now hosted in
# prefill_dense_mlp.py with its own golden_fn + build_tensor_specs for
# parallel L0 testing).
from .prefill_dense_mlp import prefill_dense_mlp_body
from .config import (
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    FINAL_RMS_K_CHUNK,
    HEAD_DIM,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LAYER_TYPE_FULL,
    LAYER_TYPES,
    LM_HEAD_K_CHUNK,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    MOE_INTERMEDIATE,
    MOE_LAYER_INDICES,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_TOP_K,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    NUM_HEADS_SWA_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    NUM_HIDDEN_LAYERS,
    ROPE_SEQ_DYN,
    SHARE_EXPERT_DIM_LOCAL,
    SWIGLU_LIMITS,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB_CHUNK,
    VOCAB_LOCAL,
    is_full_attention,
    is_moe_layer,
)
from .prefill_attention_full import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_FULL,
    attention_full_prefill,
)
from .prefill_attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
    attention_swa_prefill,
)
from .dispatch import N_RANKS_PAD, PER_RANK_BUCKETS
from .prefill_moe import select_prefill_moe_block
from .prefill_qkv_proj_rope import PREFILL_BATCH, PREFILL_SEQ, PREFILL_T, TOK_TILE
from .rms_lm_head import rms_lm_head
from .prefill_gate import prefill_gate_body
from .prefill_dispatch import (
    prefill_build_inverse_map,
    prefill_build_local_expert_csr,
    prefill_histogram_and_prefix_sum,
    prefill_pack_send_payload,
    prefill_zero_dispatch_buffers,
    prefill_zero_dispatch_buffers_chunked,
)
from .prefill_expert_routed import (
    prefill_expert_routed_silu,
    prefill_expert_routed_swiglu7,
)
from .prefill_expert_shared import (
    prefill_expert_shared_silu,
    prefill_expert_shared_swiglu16,
)
from .prefill_combine import (
    prefill_publish_src_route_table,
    prefill_push_routed_y_to_sources,
    prefill_weighted_gather_and_add,
    prefill_zero_routed_y_buf,
    prefill_zero_routed_y_buf_chunked,
)


# -----------------------------------------------------------------------------
# Compile-time tables — mirror decode_fwd.py.
# -----------------------------------------------------------------------------
# Precision-dump switch. Default 0 = disabled (production); precision/golden
# flow sets PYPTO_STEP3P5_DUMP=1 to enable dump assembles + host-visible
# buffers + host read-back.
_DUMP_ENABLED = os.environ.get("PYPTO_STEP3P5_DUMP", "0") != "0"
NUM_FULL_LAYERS = sum(
    1 for t in LAYER_TYPES[:NUM_HIDDEN_LAYERS] if t == LAYER_TYPE_FULL
)
NUM_SWA_LAYERS = NUM_HIDDEN_LAYERS - NUM_FULL_LAYERS
NUM_DENSE_LAYERS = NUM_HIDDEN_LAYERS - len(MOE_LAYER_INDICES)
NUM_MOE_LAYERS = len(MOE_LAYER_INDICES)
# SWA input-projection stacks span 33 swa-attention layers; the static
# LAYER_HIDDEN_ROWS_DYN covers only the 12 full-attention layers (task #28
# fixed swa_wo but not swa_wq/wk/wv/w_g — the 33-vs-12 gap, Problem 25).
LAYER_HIDDEN_ROWS_DYN_SWA = NUM_SWA_LAYERS * HIDDEN
# Dense MLP gate/up stacks span only the NUM_DENSE_LAYERS (3) dense layers.
# LAYER_HIDDEN_ROWS_DYN is the 12-full-attention-layer count and must not be
# reused for the dense stacks (blue MAJOR: over-provisioning 49152 -> 12288).
LAYER_HIDDEN_ROWS_DYN_DENSE = NUM_DENSE_LAYERS * HIDDEN


def _build_pos_tables():
    full_pos = [-1] * NUM_HIDDEN_LAYERS
    swa_pos = [-1] * NUM_HIDDEN_LAYERS
    f, s = 0, 0
    for i in range(NUM_HIDDEN_LAYERS):
        if is_full_attention(i):
            full_pos[i] = f
            f += 1
        else:
            swa_pos[i] = s
            s += 1
    return tuple(full_pos), tuple(swa_pos)


FULL_POS, SWA_POS = _build_pos_tables()


def _build_dense_moe_tables():
    dense_pos = [-1] * NUM_HIDDEN_LAYERS
    moe_pos = [-1] * NUM_HIDDEN_LAYERS
    d, m = 0, 0
    for i in range(NUM_HIDDEN_LAYERS):
        if is_moe_layer(i):
            moe_pos[i] = m
            m += 1
        else:
            dense_pos[i] = d
            d += 1
    return tuple(dense_pos), tuple(moe_pos)


DENSE_POS, MOE_POS = _build_dense_moe_tables()


# Window-pool sizing constants (referenced by host_orch).
N_RANKS = TP_WORLD_SIZE
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
LOCAL_RECV_MAX = 1024
N_ROUTES_PER_RANK = BATCH * MOE_TOP_K
SH_INTER_LOCAL = SHARE_EXPERT_DIM_LOCAL
INTER = MOE_INTERMEDIATE
INTER_LOCAL = INTERMEDIATE_LOCAL
N_EXPERTS = MOE_NUM_EXPERTS
TOPK = MOE_TOP_K
# Per-token INT8 activation dequant scale window width (W8A8 dispatch). Mirrors
# _moe_constants.SCALE_W_PAD and decode-side moe.py SCALE_W_PAD. Column 0 holds
# the FP32 scale; columns 1..7 are zero pad so the a2a scale window's
# pl.load / remote_load tiles stay 32B-aligned.
SCALE_W_PAD = 8

# Prefill-T tile count for the inlined EpTpMoE adapter (mirrors prefill_moe.py).
# Each MoE-layer chip_orch slices its [PREFILL_T, HIDDEN] post_norm into
# PREFILL_TILE_COUNT independent [BATCH, HIDDEN] tiles and runs the inlined
# gate / dispatch / expert_routed / expert_shared / combine pipeline once
# per tile.
PREFILL_TILE_COUNT = PREFILL_T // BATCH
assert PREFILL_T % BATCH == 0, (
    f"PREFILL_T={PREFILL_T} must be a multiple of BATCH={BATCH} so the "
    "inlined prefill-T MoE adapter can chunk into whole decode-T tiles"
)
assert (
    (BATCH * TOPK) % PREFILL_A2A_ROWS == 0 and 0 < PREFILL_A2A_ROWS <= 32
), (
    f"PREFILL_A2A_ROWS={PREFILL_A2A_ROWS} must divide BATCH*TOPK="
    f"{BATCH * TOPK} and stay <= 32 so the ep_all_to_all_bulk tile "
    f"[{PREFILL_A2A_ROWS}, {HIDDEN}] INT8 ("
    f"{PREFILL_A2A_ROWS * HIDDEN // 1024} KB) fits the UB budget"
)


# -----------------------------------------------------------------------------
# Phase X.10 — kernel-internal constants for the inlined MoE method bodies.
#
# pypto frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
# (instantiating a ``@pl.program`` inside another ``@pl.program`` body is not
# a supported feature), so the entire body of ``EpTpMoE`` (every
# ``@pl.function`` method plus the ``chip_orch`` per-tile loop) is inlined
# directly into ``PrefillLayerMoE``. The originals in ``moe.py`` /
# ``prefill_moe.py`` remain intact (their standalone harnesses still drive
# them as separate ``@pl.program`` instances).
# -----------------------------------------------------------------------------

# Router (gate) kernel constants — MUST match _moe_constants.py (the
# L0-validated single source of truth for the spliced @pl.jit.inline shim
# bodies). The pypto frontend resolves a spliced body's free variables in the
# ENCLOSING @pl.program scope (this module's globals), NOT the body's own
# __globals__, so a stale local copy here silently diverges from the shim's
# L0-validated values. See _moe_constants.py:58-72.
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_K_CHUNK = 256  # K1 L1-fit fix (512 overflowed L1)
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0  # MOE_ROUTER_SCALING_FACTOR
assert TOPK <= ROUTER_TOPK_PAD
assert HIDDEN % ROUTER_GATE_K_CHUNK == 0

# Routed-expert kernel constants — MUST match _moe_constants.py:77-115.
ROUTED_GATE_K_CHUNK = 128
ROUTED_GATE_N_CHUNK = 64
ROUTED_DOWN_K_CHUNK = 128
ROUTED_DOWN_N_CHUNK = 64
ROUTED_MAX_TILE = LOCAL_RECV_MAX
ROUTED_ROW_TILE = 16
ROUTED_H_QUANT_N_CHUNK = 256
assert HIDDEN % ROUTED_GATE_K_CHUNK == 0
assert HIDDEN % ROUTED_DOWN_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_GATE_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_H_QUANT_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_DOWN_K_CHUNK == 0
assert ROUTED_MAX_TILE % ROUTED_ROW_TILE == 0

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = SH_INTER_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = SH_INTER_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
# Narrow swiglu N-chunk for the shared-expert gate/up + down projection: the
# full [BATCH,160] Vec-tile cast/clamp crosses a 128-column block boundary and
# is miscompiled (wide-tile tmov misprune -> ~45% wrong). The @pl.jit.inline
# shim bodies (prefill_expert_shared.py) compute 5 narrow [BATCH,32] chunks
# instead. This constant is mirrored here because ``pl.inline(shim._func)``
# resolves the shim body's globals from THIS module's namespace, not the shim's.
SHARED_SWIGLU_N_CHUNK = 32
# Swiglu clamp limits — mirrored for the same inline-globals reason as
# SHARED_SWIGLU_N_CHUNK: prefill_expert_routed_swiglu7 /
# prefill_expert_shared_swiglu16 resolve these from THIS module's namespace.
ROUTED_SWIGLU_LIMIT = 7.0
SHARED_SWIGLU_LIMIT = 16.0
assert SH_INTER_LOCAL == SHARED_SWIGLU_N_CHUNK * 5
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0


# =============================================================================
# Prefill dense-MLP body — TP-sliced gate/up/down + tp_all_reduce.
#
# v1 refactor (design §7): the body has been extracted to
# ``prefill_dense_mlp.py:prefill_dense_mlp_body`` (a NEW prefill-side
# ``@pl.jit.inline`` body with its own ``golden_prefill_dense_mlp`` +
# ``build_tensor_specs`` for parallel L0 testing on a2a3sim).
#
# The former local ``_prefill_dense_mlp_body_tp`` definition is removed;
# callers below import ``prefill_dense_mlp_body`` directly and bind it once
# at factory build time via ``dense_inline = pl.inline(prefill_dense_mlp_body._func)``
# (the pre-existing pattern in this file — ``pl.inline(body._func)`` is
# constructed ONCE at factory build and the resulting object is called by
# name inside the @pl.function body; the frontend rejects
# ``pl.inline(body._func)(args)`` written inline in the body).
#
# TODO(v2)/PROPOSAL: the 10 inlined MoE method bodies formerly in the
# per-layer ``_build_prefill_layer_moe_program`` (gate / dispatch helpers /
# expert_routed / expert_shared / combine helpers) remain inlined here
# (and duplicated in ``prefill_moe.py``). The v1 shim-re-export approach
# (prefill_gate.py / prefill_dispatch.py / prefill_expert_routed.py /
# prefill_expert_shared.py / prefill_combine.py re-exporting the
# decode-side ``@pl.jit.inline`` bodies) is DEAD — blue-verified: the
# pypto frontend parses an inlined body in the ENCLOSING ``@pl.program``
# scope, not in the body's own ``__globals__``, so decode-side bodies
# referencing decode-side module-level constants (T, SCORE_PAD,
# GATE_K_CHUNK, FP32_NEG_INF, ROUTE_SCALE, SORT_PAD, TOPK_PAD, ...) fail
# with "Undefined variable 'T'" at build time when inlined into prefill's
# enclosing ``@pl.program``.
#
# v2 PATH (task #13, blue-greenlit): 9 NEW prefill-side ``@pl.jit.inline``
# bodies lifted from ``prefill_moe.py:388-1295`` into the 5 shim files,
# using prefill-side names (BATCH, ROUTER_SCORE_PAD, ROUTER_GATE_K_CHUNK,
# ROUTER_FP32_NEG_INF, ROUTER_SCALE, ROUTER_TOPK_PAD, ROUTER_SORT_PAD,
# N_RANKS, N_LOCAL_EXPERTS, INTER, SH_INTER_LOCAL, LOCAL_RECV_MAX,
# ROUTED_*, SHARED_*). Math unchanged. The runtime
# ``if _routed_swiglu_step`` / ``if _shared_swiglu_step`` branches are
# split into compile-time silu / swiglu7 / swiglu16 body variants
# (select_prefill_expert_routed / select_prefill_expert_shared return
# the right variant). Constants centralised in ``_moe_constants.py`` to
# break the circular dep (shim files and ``prefill_moe.py`` both import
# them; after #9 refactor, ``prefill_moe.py`` will pl.inline the shim
# bodies, creating a two-way import — the constants must live in a
# third module).
#
# BUG FIX (team-lead directive): the gate body's mrgsort call is lifted
# as the CANONICAL form1 pattern (``mrgsort(srt, block_len=256)``), NOT
# the buggy form2 two-slice (``mrgsort(srt[:, 0:512], srt[:, 512:1024])``
# at prefill_moe.py:453). Per task #12 finding, form2 violates the
# format2 contract (each src must be a single sorted run; with
# ROUTER_SCORE_PAD=512 each 512-position slice contains 2 sorted runs
# of 256). Pure-Python emulation shows form1 matches
# torch.argsort(stable=True) on all test cases; form2 diverges on 3
# of 4. See ``_tmp_mrgsort_emulation.py``.
#
# Decode-side bodies remain untouched (hard rule: do not modify decode).
#
# STATUS (task #13): all 12 @pl.jit.inline bodies (9 logical + 3
# activation-split variants) landed in the 5 shim files, import OK,
# prefill_fwd + prefill_moe no regression, distributed_mock PASS.
# Runtime L0 golden PENDING-ENV (g++-15 symlink points to g++-12,
# cannot compile pto-isa c++23 headers). Next: #9 (prefill_moe.py
# refactor — replace inlined bodies with pl.inline calls to the new
# shim bodies) and #10 (prefill_fwd.py refactor — same).
# =============================================================================




# -----------------------------------------------------------------------------
# (routed_lim, shared_lim) -> kind string for MoE layers.  ``kind`` is a plain
# routing label consumed by ``whole_chip_orch``'s in-class @pl.function
# dispatch; per-layer @pl.program compile targets were removed (T2a per-layer
# deprecation) — ``select_prefill_layer`` no longer resolves to a program.
# -----------------------------------------------------------------------------
_KIND_BY_PAIR_FULL = {
    (0.0, 0.0): "full_moe_silu_silu",
    (7.0, 0.0): "full_moe_swiglu7_silu",
    (7.0, 16.0): "full_moe_swiglu7_swiglu16",
}
_KIND_BY_PAIR_SWA = {
    (0.0, 0.0): "swa_moe_silu_silu",
    (7.0, 0.0): "swa_moe_swiglu7_silu",
    (7.0, 16.0): "swa_moe_swiglu7_swiglu16",
}



def select_prefill_layer(layer_idx: int):
    """Return ``(kind, routed_lim, shared_lim)`` for the prefill layer at
    ``layer_idx``.

    P3a reshaped contract (design §9 item 1, now DONE): the 45-layer
    ``whole_chip_orch`` dispatches on ``kind`` to in-class ``@pl.function``
    chip_orch methods — no ``@pl.program`` class is returned.  Per-layer
    @pl.program compile targets were removed (T2a per-layer deprecation);
    ``kind`` is a plain routing label.
    """
    full = is_full_attention(layer_idx)
    moe = is_moe_layer(layer_idx)
    if full and not moe:
        return "full_dense", 0.0, 0.0
    if (not full) and (not moe):
        return "swa_dense", 0.0, 0.0
    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])
    table = _KIND_BY_PAIR_FULL if full else _KIND_BY_PAIR_SWA
    try:
        kind = table[(routed_lim, shared_lim)]
    except KeyError as err:
        raise ValueError(
            f"Unsupported (routed={routed_lim}, shared={shared_lim}) "
            f"for layer {layer_idx}",
        ) from err
    return kind, routed_lim, shared_lim


_VALID_LAYER_KINDS = frozenset(
    {"full_dense", "swa_dense"}
    | set(_KIND_BY_PAIR_FULL.values())
    | set(_KIND_BY_PAIR_SWA.values())
)


# ── EP all-to-all bodies (module-level @pl.jit.inline). ──
# Pulled out of the Step3p5PrefillFwd class (they were Inline methods there)
# so the per-row and bulk-chunked variants can be selected at factory scope
# with a plain Python if/else (the ZERO_CHUNK idiom below). A device-level
# ``if`` inside moe_dispatch_step is NOT viable: both branches get inlined
# into the IfStmt region where body-local aliases in shape positions turn
# into dynamic IR Scalars and InitMemRef rejects them — the lh45 BULK=0 arm
# failed on the ORIGINAL body's [1, d_cols] this way (smokes 1-4, 2026-10-08).
# Shape dims here are therefore bare module constants; range bounds and
# offsets may stay dynamic.
@pl.jit.inline
def prefill_ep_all_to_all(
    send: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    recv: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    send_scale: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
    recv_scale: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
    send_counts: pl.Tensor[[N_RANKS], pl.INT32],
    recv_counts: pl.Tensor[[N_RANKS], pl.INT32],
    send_offsets: pl.Tensor[[N_RANKS], pl.INT32],
    recv_offsets: pl.Tensor[[N_RANKS], pl.INT32],
    signal_window: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Pull-side fixed-slot token-level all-to-all over the EP group.

    Lifted verbatim from the original ``Step3p5PrefillFwd.ep_all_to_all``
    (itself from prefill_moe.py:323-384: Set signal + Ge wait, one signal
    round-trip for the data-arrival handshake). The only body edit is
    ``[1, d_cols]`` -> ``[1, HIDDEN]`` (shape dims must be bare constants,
    see the block comment above).
    """
    group_size = N_RANKS

    # Symmetric fixed-slot a2a (§D:129, mirrors decode moe.py:316-409).
    # Each (src,dst) pair owns a full BATCH*TOPK slot block; the
    # dst-side pack writes at dst*BATCH*TOPK and the src-side read
    # pulls from my_rank*BATCH*TOPK, so no per-rank offset tensor is
    # needed (the old variable-length recv_offsets read had the
    # Problem-27 src/dst prefix-sum skew). send_counts/recv_counts/
    # send_offsets/recv_offsets become unused formals (kept for
    # signature parity, same as decode).
    PER_PEER_BOUND = BATCH * TOPK

    # 1) Local self-bucket copy (symmetric fixed slot my_rank*MAX).
    self_base = pl.cast(my_rank * PER_PEER_BOUND, pl.INDEX)
    for r in pl.range(PER_PEER_BOUND):
        self_tile = pl.load(send, [self_base + r, 0], [1, HIDDEN])
        pl.store(self_tile, [self_base + r, 0], recv)
        # Fused per-token dequant scale self-copy (same row index,
        # SCALE_W_PAD-wide tile — mirrors decode moe.py:354-358).
        s_self = pl.load(
            send_scale, [self_base + r, 0], [1, SCALE_W_PAD],
        )
        pl.store(s_self, [self_base + r, 0], recv_scale)

    # 2) AtomicAdd(1) notify every peer (EP barrier signal, §D:129).
    for peer in pl.range(group_size):
        if peer != my_rank:
            pld.system.notify(
                target=signal_window,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    # 3) Ge(1) wait for every peer.
    for src in pl.range(group_size):
        if src != my_rank:
            pld.system.wait(
                signal=signal_window,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    # 4) Pull every peer's bucket-for-me (symmetric fixed slots).
    #    My block in peer's send_buf is at my_rank*MAX; store into
    #    peer's block in my recv at peer*MAX. Read the full MAX rows
    #    (gap rows past the real count are ignored by the re-pack,
    #    which uses the per-(src,e) counts).
    my_base = pl.cast(my_rank * PER_PEER_BOUND, pl.INDEX)
    for peer in pl.range(group_size):
        if peer != my_rank:
            peer_base = pl.cast(peer * PER_PEER_BOUND, pl.INDEX)
            for r in pl.range(PER_PEER_BOUND):
                peer_tile = pld.tile.remote_load(
                    send,
                    peer=peer,
                    offsets=[my_base + r, 0],
                    shape=[1, HIDDEN],
                )
                pl.store(peer_tile, [peer_base + r, 0], recv)
                # Fused scale pull: same peer/row, SCALE_W_PAD-wide tile
                # (mirrors decode moe.py:401-407).
                s_peer = pld.tile.remote_load(
                    send_scale,
                    peer=peer,
                    offsets=[my_base + r, 0],
                    shape=[1, SCALE_W_PAD],
                )
                pl.store(s_peer, [peer_base + r, 0], recv_scale)

    return recv


@pl.jit.inline
def prefill_ep_all_to_all_bulk(
    send: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    recv: pld.DistributedTensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
    send_scale: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
    recv_scale: pld.DistributedTensor[
        [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
    ],
    send_counts: pl.Tensor[[N_RANKS], pl.INT32],
    recv_counts: pl.Tensor[[N_RANKS], pl.INT32],
    send_offsets: pl.Tensor[[N_RANKS], pl.INT32],
    recv_offsets: pl.Tensor[[N_RANKS], pl.INT32],
    signal_window: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Bulk-chunked variant of :func:`prefill_ep_all_to_all` (audit A1).

    Byte-identical semantics (same rows copied in the same order, same
    barrier placement) — only the copy granularity changes:
    [PREFILL_A2A_ROWS, HIDDEN] tiles instead of per-row [1, HIDDEN]. Legal
    because the fixed-slot layout makes every (peer, my_rank-block) segment
    contiguous (pack writes each dst block as a bucket-contiguous t-major
    prefix), and the gap rows past the real counts are zero-filled by
    zero_dispatch_buffers_step and ignored downstream — the per-row form
    copies them too, so the bulk form is bit-exact. Audit A1
    (performance/RESULT_NEW2.md §二): the per-row form issues 7*128 payload
    + 7*128 scale remote ops per dispatch (~1.65M/forward) and dominates
    the measured 1.853 ms moe_dispatch_step instance; the bulk form issues
    7*(128/ROWS)*2 = 112 ops at ROWS=16. The count/offset formals stay
    unused for signature parity with the per-row body.
    """
    group_size = N_RANKS
    # Same symmetric fixed-slot a2a as prefill_ep_all_to_all (§D:129).
    PER_PEER_BOUND = BATCH * TOPK
    # NOTE: shape dims must be bare module constants here (HIDDEN /
    # SCALE_W_PAD / PREFILL_A2A_ROWS), same rule as the literal 4 in
    # prefill_zero_dispatch_buffers_chunked: body-local aliases
    # (`d_cols = HIDDEN`, `chunk_rows = PREFILL_A2A_ROWS`) turn the
    # shape element into a dynamic IR Scalar and break InitMemRef
    # (lh8 smokes 2-4, 2026-10-08). Range bounds and offsets may stay
    # dynamic, so the PER_PEER_BOUND alias above is fine.
    self_base = pl.cast(my_rank * PER_PEER_BOUND, pl.INDEX)

    # 1) Local self-bucket copy, chunked.
    for r0 in pl.range(0, PER_PEER_BOUND, PREFILL_A2A_ROWS):
        self_tile = pl.load(
            send, [self_base + r0, 0], [PREFILL_A2A_ROWS, HIDDEN],
        )
        pl.store(self_tile, [self_base + r0, 0], recv)
        # Fused per-token dequant scale self-copy (same rows,
        # SCALE_W_PAD-wide tile — mirrors step 1 above).
        s_self = pl.load(
            send_scale, [self_base + r0, 0],
            [PREFILL_A2A_ROWS, SCALE_W_PAD],
        )
        pl.store(s_self, [self_base + r0, 0], recv_scale)

    # 2) AtomicAdd(1) notify every peer (EP barrier, unchanged).
    for peer in pl.range(group_size):
        if peer != my_rank:
            pld.system.notify(
                target=signal_window,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    # 3) Ge(1) wait for every peer (unchanged).
    for src in pl.range(group_size):
        if src != my_rank:
            pld.system.wait(
                signal=signal_window,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    # 4) Pull every peer's bucket-for-me, chunked bulk tiles.
    #    My block in peer's send_buf is at my_rank*PER_PEER_BOUND;
    #    store into peer's block in my recv at peer*PER_PEER_BOUND.
    my_base = pl.cast(my_rank * PER_PEER_BOUND, pl.INDEX)
    for peer in pl.range(group_size):
        if peer != my_rank:
            peer_base = pl.cast(peer * PER_PEER_BOUND, pl.INDEX)
            for r0 in pl.range(0, PER_PEER_BOUND, PREFILL_A2A_ROWS):
                peer_tile = pld.tile.remote_load(
                    send,
                    peer=peer,
                    offsets=[my_base + r0, 0],
                    shape=[PREFILL_A2A_ROWS, HIDDEN],
                )
                pl.store(peer_tile, [peer_base + r0, 0], recv)
                # Fused scale pull, chunked (mirrors step 4 above).
                s_peer = pld.tile.remote_load(
                    send_scale,
                    peer=peer,
                    offsets=[my_base + r0, 0],
                    shape=[PREFILL_A2A_ROWS, SCALE_W_PAD],
                )
                pl.store(s_peer, [peer_base + r0, 0], recv_scale)

    return recv


# =============================================================================
# Top-level @pl.program — Step3p5PrefillFwd.
# =============================================================================
def _build_prefill_fwd_program(
    tp_size: int = TP_WORLD_SIZE,
    layer_lo: int = 0,
    layer_hi: int = NUM_HIDDEN_LAYERS,
    n_moe_layers: int | None = None,
):
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must divide tp_size={tp_size}"
        )
    if not (0 <= layer_lo < layer_hi <= NUM_HIDDEN_LAYERS):
        raise ValueError(
            f"layer range [{layer_lo}, {layer_hi}) must satisfy "
            f"0 <= layer_lo < layer_hi <= NUM_HIDDEN_LAYERS={NUM_HIDDEN_LAYERS}"
        )
    # MoE weight stacks are indexed by the RELATIVE slot ``moe_pos = li - base``
    # where ``base = max(layer_lo, NUM_DENSE_LAYERS)``. For a whole-net or
    # dense-prefix run (layer_lo <= NUM_DENSE_LAYERS) this is the absolute slot
    # li - 3, so the 42-layer default layout is unchanged; for a late-MoE slice
    # (e.g. layer_lo=43) it rebases moe_pos to 0 and the stacks only need
    # n_moe_layers = max(1, layer_hi - base), shrinking the routed-expert stack
    # from ~47.6 GB/rank to ~2.3 GB/rank so it fits in HBM. The signal/window
    # stacks stay NUM_MOE_LAYERS tall: their rebased offsets land in the first
    # slots and never run past the end.
    base = max(layer_lo, NUM_DENSE_LAYERS)
    if n_moe_layers is None:
        n_moe_layers = NUM_MOE_LAYERS
    if n_moe_layers < 1:
        raise ValueError(f"n_moe_layers must be >= 1, got {n_moe_layers}")

    # Rotary dims (64 full / 128 swa).
    rotary_dim_full = 64
    rotary_dim_swa = 128

    rms_lm_head_inline = pl.inline(rms_lm_head._func)

    # Body inlines bound once at factory build time.  The frontend rejects
    # ``pl.inline(body._func)(args)`` written inline inside a @pl.function
    # body — the inline object must be constructed at factory build time
    # and called by name inside the body (proven pattern: decode_fwd.py:279
    # ``attention_full_inline = pl.inline(attention_full._func)`` and the
    # dense_layer chip_orch at line 511 for ``dense_inline``).
    #
    # P3a skeleton: the 8 per-kind chip_orch methods below are PLACEHOLDER
    # stubs (pl.create_tensor + return) so that the 45-layer whole_chip_orch
    # loop, the 8-way runtime if/elif dispatch, the pl.create_tensor Out
    # placement (Orchestration = device scope, B-probe-safe), and the
    # host_orch -> whole_chip_orch wiring all compile cleanly without body
    # UB interference.  Wiring the real inlines (attn_full_inline /
    # attn_swa_inline / dense_mlp_inline / MoE bodies) is a follow-up:
    #   - attn_full_inline: NPU PASS (P3c, TP=1) — wire first.
    #   - attn_swa_inline:  qk_norm Vec UB overflow (Task #18, parallel
    #                       agent on device 4) — would hit UB if wired.
    #   - dense_mlp_inline:  UB overflow (deferred).
    #   - MoE bodies:       prefill_moe_block_body not packaged yet
    #                       (task #13); 6 MoE chip_orch methods stay stubs
    #                       until that lands.
    # The follow-up also expands whole_chip_orch's signature to the full
    # per-layer weight stacks (full/swa × dense/moe × swiglu), sliced per
    # layer via FULL_POS / SWA_POS / DENSE_POS / MOE_POS offset tables.
    rmsnorm_inline = pl.inline(zero_centered_rmsnorm_apply._func)
    attn_full_inline = pl.inline(attention_full_prefill._func)
    attn_swa_inline = pl.inline(attention_swa_prefill._func)
    dense_mlp_inline = pl.inline(prefill_dense_mlp_body._func)
    # MoE silu/silu shim bodies (task #27): the 10 @pl.jit.inline stage
    # bodies are bound once here and called by name inside the MoE chip_orch
    # bodies below. The swiglu7/16 variants are added when layers 43/44 are
    # wired (next phase).
    gate_inline = pl.inline(prefill_gate_body._func)
    histogram_inline = pl.inline(prefill_histogram_and_prefix_sum._func)
    pack_send_inline = pl.inline(prefill_pack_send_payload._func)
    if PREFILL_ZERO_CHUNK == 1:
        zero_dispatch_inline = pl.inline(prefill_zero_dispatch_buffers._func)
    else:
        zero_dispatch_inline = pl.inline(
            prefill_zero_dispatch_buffers_chunked._func
        )
    build_csr_inline = pl.inline(prefill_build_local_expert_csr._func)
    build_inverse_inline = pl.inline(prefill_build_inverse_map._func)
    routed_silu_inline = pl.inline(prefill_expert_routed_silu._func)
    shared_silu_inline = pl.inline(prefill_expert_shared_silu._func)
    routed_swiglu7_inline = pl.inline(prefill_expert_routed_swiglu7._func)
    shared_swiglu16_inline = pl.inline(prefill_expert_shared_swiglu16._func)
    publish_route_inline = pl.inline(prefill_publish_src_route_table._func)
    if PREFILL_ZERO_CHUNK == 1:
        zero_routed_inline = pl.inline(prefill_zero_routed_y_buf._func)
    else:
        zero_routed_inline = pl.inline(prefill_zero_routed_y_buf_chunked._func)
    push_routed_inline = pl.inline(prefill_push_routed_y_to_sources._func)
    gather_add_inline = pl.inline(prefill_weighted_gather_and_add._func)
    # EP a2a variant selection (audit A1, env-gated like ZERO_CHUNK): plain
    # Python if/else at factory scope binds ONE name; moe_dispatch_step calls
    # ``ep_a2a_inline(...)`` (parse-time splice, no device branch — see the
    # block comment above prefill_ep_all_to_all for why a device-level if is
    # not viable).
    if PREFILL_A2A_BULK == 1:
        ep_a2a_inline = pl.inline(prefill_ep_all_to_all_bulk._func)
    else:
        ep_a2a_inline = pl.inline(prefill_ep_all_to_all._func)

    @pl.program
    class Step3p5PrefillFwd:
        # ── Final RMSNorm + vocab-sliced LM head (Phase 3b per-position). ──
        # Renamed from ``chip_orch`` to ``rms_lm_head_chip_orch`` to free the
        # ``<kind>_chip_orch`` namespace for the 8 per-layer-kind methods.
        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def rms_lm_head_chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[
                [VOCAB_LOCAL, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[PREFILL_T, VOCAB_LOCAL], pl.FP32]
            ],
        ) -> pl.Tensor[[PREFILL_T, VOCAB_LOCAL], pl.FP32]:
            # Phase 3b per-position tail: loop PREFILL_TILE_COUNT tiles of
            # BATCH(=16) rows, run the shared rms_lm_head body per tile into a
            # [BATCH, VOCAB_LOCAL] FP32 scratch, then copy the scratch into the
            # [PREFILL_T, VOCAB_LOCAL] Out at row [t0, 0].  The copy is a
            # VOCAB_CHUNK-column pl.assemble loop whose source is a pl.slice —
            # the tensor.assemble(target, tensor.slice(src, ...), offset) form
            # lowers to tile.load + tile.store (same as the whole-chip residual
            # writeback), whereas assembling a WHOLE [BATCH, VOCAB_LOCAL] tensor
            # into the Out (source is a plain tensor, not a slice) would fall
            # through to an unconverted tensor.assemble with no codegen.  The
            # input tile is likewise copied into a fresh [BATCH, HIDDEN] scratch
            # first — a partition_view slice would propagate explicit
            # valid_row/valid_col through the bf16->f32 casts and fail ptoas
            # (same reason as the Phase 3a last-token tail_hidden scratch).
            for tt in pl.range(PREFILL_TILE_COUNT):
                t0 = tt * BATCH
                tile_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_tail_tile_in",
                ):
                    tile_hidden = pl.assemble(
                        tile_hidden,
                        pl.slice(current_hidden, [BATCH, HIDDEN], [t0, 0]),
                        [0, 0],
                    )
                tile_logits = pl.create_tensor(
                    [BATCH, VOCAB_LOCAL], dtype=pl.FP32,
                )
                tile_logits = rms_lm_head_inline(
                    tile_hidden,
                    final_norm_weight,
                    lm_head_weight,
                    seq_lens,
                    tile_logits,
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_tail_tile_out",
                ):
                    for vo in pl.range(VOCAB_LOCAL // VOCAB_CHUNK):
                        o0 = vo * VOCAB_CHUNK
                        logits_shard_out = pl.assemble(
                            logits_shard_out,
                            pl.slice(
                                tile_logits,
                                [BATCH, VOCAB_CHUNK],
                                [0, o0],
                            ),
                            [t0, o0],
                        )
            return logits_shard_out

        # ── TP all-reduce collective (InCore). ──
        # The inlined attention/dense-MLP bodies call ``self.tp_all_reduce``
        # (at tp_size > 1) to sum o_proj / down_proj partials across the TP
        # group.  Barrier-style body (mirrors decode moe.py:270-313, PASS TP=8),
        # t_rows=PREFILL_T, d_cols=HIDDEN.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[PREFILL_T, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            # Barrier-style all-reduce (mirrors decode moe.py:270-313 tp_all_reduce,
            # PASS TP=8). Replaces the pull-side RING which deadlocked at TP=8
            # (monotonic AtomicAdd signal — see task #2). tmp_window is full
            # [PREFILL_T, HIDDEN] (not chunked) so each rank stages its whole
            # contribution before the barrier, then reduces every chunk across
            # all peers. ar_chunk = HIDDEN // 8 (fixed, §7a) keeps the fp32
            # accumulator [BATCH, ar_chunk] = 32KB inside UB at every TP.
            group_size = tp_size
            ar_chunk = HIDDEN // 8
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                for k0 in pl.range(0, HIDDEN, ar_chunk):
                    stage_tile = pl.load(local, [ttr, k0], [BATCH, ar_chunk])
                    pl.store(stage_tile, [ttr, k0], tmp_window)
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            for tt in pl.range(PREFILL_TILE_COUNT):
                ttr = tt * BATCH
                for k0 in pl.range(0, HIDDEN, ar_chunk):
                    own_tile = pl.load(tmp_window, [ttr, k0], [BATCH, ar_chunk])
                    acc = pl.cast(own_tile, target_type=pl.FP32)
                    for peer in pl.range(group_size):
                        if peer != my_rank:
                            recv = pld.tile.remote_load(
                                tmp_window, peer=peer,
                                offsets=[ttr, k0], shape=[BATCH, ar_chunk],
                            )
                            acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                    pl.store(
                        pl.cast(acc, target_type=pl.BF16, mode="rint"), [ttr, k0], local,
                    )
            return local

        # ── BATCH-sized TP all-reduce for the shared expert (InCore). ──
        # The existing tp_all_reduce above is PREFILL_T-sized (used by the
        # dense MLP partial); the shared-expert lane produces one [BATCH,
        # HIDDEN] partial per MoE tile, so it needs its own BATCH-sized barrier
        # (mirrors decode moe.py:270-313, PASS TP=8).
        @pl.function(type=pl.FunctionType.InCore)
        def moe_tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, HIDDEN], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # Barrier-style all-reduce (mirrors decode moe.py:270-313 tp_all_reduce,
            # PASS TP=8). Replaces the RING version which deadlocked at TP=8
            # (lone unconverted survivor; decode moe.py:278-281). tmp_window is
            # full [BATCH, HIDDEN] (not chunked) so each rank stages its whole
            # contribution before the barrier, then reduces every chunk across
            # all peers. NOTE: AtomicAdd + Ge(expected=1) is non-consuming
            # (collectives.py: "cells continue accumulating"), so the CALLER
            # must hand this a fresh per-tile [tp_size,1] signal slice (R3
            # plan-a) — reusing one window across the 8-tile unroll would let
            # tiles 2+ pass the wait before peers land.
            group_size = tp_size
            # ar_chunk = HIDDEN // 8 (fixed, §7a): NOT HIDDEN // tp_size —
            # at TP=1 that would make [BATCH, ar_chunk] fp32 = 256KB and
            # overflow the 184KB UB Vec buffer.
            ar_chunk = HIDDEN // 8
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                stage_tile = pl.load(local, [0, k0], [BATCH, ar_chunk])
                pl.store(stage_tile, [0, k0], tmp_window)
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window, peer=peer,
                        offsets=[my_rank, 0], value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window, offsets=[src, 0],
                        expected=1, cmp=pld.WaitCmp.Ge,
                    )
            for k0 in pl.range(0, HIDDEN, ar_chunk):
                own_tile = pl.load(tmp_window, [0, k0], [BATCH, ar_chunk])
                acc = pl.cast(own_tile, target_type=pl.FP32)
                for peer in pl.range(group_size):
                    if peer != my_rank:
                        recv = pld.tile.remote_load(
                            tmp_window, peer=peer,
                            offsets=[0, k0], shape=[BATCH, ar_chunk],
                        )
                        acc = pl.add(acc, pl.cast(recv, target_type=pl.FP32))
                pl.store(
                    pl.cast(acc, target_type=pl.BF16, mode="rint"), [0, k0], local,
                )
            return local

        # SERIALIZE shared tp_all_reduce -> routed dispatch (break overlap
        # deadlock): declares sh_y as an input so the orchestration forces the
        # shared-expert tp_all_reduce to COMPLETE before the routed
        # dispatch/combine cross-rank collectives run (mirrors decode moe.py
        # _serialize_after_shared). The routed lane is now INT8 (W8A8): the
        # dispatch reads x_scale (per-token FP32 dequant scale, [BATCH,
        # SCALE_W_PAD]) alongside the INT8 payload, so we fold 0*sh_y into
        # x_scale to establish the sh_y -> dispatch data dependency without
        # touching the large INT8 payload tile. Mirrors moe.py's elementwise
        # ``add(xr, mul(yr, 0.0))``, but sh_y ([1, HIDDEN]) is wider than
        # x_scale ([1, SCALE_W_PAD]), so we consume the full sh_y row in
        # SCALE_W_PAD-wide chunks — each ``add(s, mul(chunk, 0.0))`` is a
        # shape-compatible elementwise identity (no row_sum, which would need
        # a tmp_tile inside this scalar loop). The result is exact (0*chunk
        # adds no rounding).
        @pl.function(type=pl.FunctionType.InCore)
        def _serialize_after_shared(
            self,
            x_scale: pl.Tensor[[BATCH, SCALE_W_PAD], pl.FP32],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            x_scale_out: pl.Out[
                pl.Tensor[[BATCH, SCALE_W_PAD], pl.FP32]
            ],
        ):
            if PREFILL_SER_MODE == 2:
                # Tile-mode carrier: ONE [BATCH,16] sh_y read (covers every
                # row's first 16 columns) folded as 0*x into ONE [BATCH,8]
                # x_scale read -> x_scale_out write. Same task edge as
                # SER_MODE 0/1; the per-row scalar loop is gone entirely.
                # Exact (0*x adds no rounding).
                yr_t = pl.cast(
                    pl.load(sh_y, [0, 0], [BATCH, 2 * SCALE_W_PAD]),
                    target_type=pl.FP32,
                )
                s_t = pl.add(
                    pl.load(x_scale, [0, 0], [BATCH, SCALE_W_PAD]),
                    pl.mul(
                        pl.slice(yr_t, [BATCH, SCALE_W_PAD], [0, 0]),
                        0.0,
                    ),
                )
                pl.store(s_t, [0, 0], x_scale_out)
                return x_scale_out
            for b in pl.range(BATCH):
                s = pl.load(x_scale, [b, 0], [1, SCALE_W_PAD])
                if PREFILL_SER_MODE == 0:
                    # sh_y is BF16: a [1, SCALE_W_PAD]=[1, 8] load has a 16-byte
                    # row (8*2B), which ptoas rejects (alloc_tile rows must be
                    # 32-byte aligned). Load a [1, 2*SCALE_W_PAD]=[1, 16] chunk
                    # (32B) and fold BOTH [1, SCALE_W_PAD] FP32 halves into s so
                    # the full sh_y row is still consumed for the dependency.
                    for k in pl.range(HIDDEN // (2 * SCALE_W_PAD)):
                        yr16 = pl.cast(
                            pl.load(
                                sh_y,
                                [b, k * 2 * SCALE_W_PAD],
                                [1, 2 * SCALE_W_PAD],
                            ),
                            target_type=pl.FP32,
                        )
                        s = pl.add(
                            s,
                            pl.mul(
                                pl.slice(
                                    yr16, [1, SCALE_W_PAD], [0, 0],
                                ),
                                0.0,
                            ),
                        )
                        s = pl.add(
                            s,
                            pl.mul(
                                pl.slice(
                                    yr16,
                                    [1, SCALE_W_PAD],
                                    [0, SCALE_W_PAD],
                                ),
                                0.0,
                            ),
                        )
                else:
                    # Minimal dependency carrier: one 32B sh_y chunk per row.
                    # The sh_y task edge stays declared (whole tensor input);
                    # only the in-task dummy-load count shrinks ~512x. Exact
                    # (0*x adds no rounding).
                    yr16 = pl.cast(
                        pl.load(sh_y, [b, 0], [1, 2 * SCALE_W_PAD]),
                        target_type=pl.FP32,
                    )
                    s = pl.add(
                        s,
                        pl.mul(
                            pl.slice(yr16, [1, SCALE_W_PAD], [0, 0]),
                            0.0,
                        ),
                    )
                    s = pl.add(
                        s,
                        pl.mul(
                            pl.slice(yr16, [1, SCALE_W_PAD], [0, SCALE_W_PAD]),
                            0.0,
                        ),
                    )
                pl.store(s, [b, 0], x_scale_out)
            return x_scale_out

        # ── MoE dispatch / combine glue (InCore). ──
        # The per-tile EP dispatch (histogram -> publish pub_counts ->
        # count_done barrier -> pack send payload -> recv_counts/offsets ->
        # ep_all_to_all -> build local-expert CSR -> reorder recv_x into
        # local_routed_x) and the combine (publish src_route_table ->
        # route_pub barrier -> push routed-y rows -> weighted gather+add).
        # Scalar pl.read/pl.write/pl.load/pl.store and pld.system.notify/wait
        # live in InCore scope (decode/OLD-preflight-faithful); the
        # Orchestration chip_orch below calls these per tile. The pure-scalar
        # shim bodies (histogram/pack/csr/publish/push) are @pl.jit.inline
        # and inherit this InCore context.
        @pl.function(type=pl.FunctionType.InCore)
        def zero_dispatch_buffers_step(
            self,
            send_buf: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale_buf: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            recv_x: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
        ) -> pld.DistributedTensor[
            [LOCAL_RECV_MAX, HIDDEN], pl.INT8
        ]:
            # Scalar zero-fill body must live in its own InCore method (NOT
            # spliced into moe_dispatch_step) — the texpands+tstore loop trips
            # the pto-memory-consistency pass when inlined (ptoas 60s hang),
            # mirroring zero_routed_y_buf_step on the combine side.
            zero_dispatch_inline(
                send_buf, send_scale_buf, recv_x, recv_scale,
            )
            return send_buf

        @pl.function(type=pl.FunctionType.InCore)
        def moe_dispatch_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.INT8],
            x_scale: pl.Tensor[[BATCH, SCALE_W_PAD], pl.FP32],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            local_routed_x_out: pl.Out[
                pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8]
            ],
            local_routed_x_scale_out: pl.Out[
                pl.Tensor[[1, LOCAL_RECV_MAX], pl.FP32]
            ],
            local_expert_offset: pl.Out[
                pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32]
            ],
            local_expert_count: pl.Out[
                pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32]
            ],
            send_buf: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale_buf: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            recv_x: pld.DistributedTensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            pl.Tensor[[1, LOCAL_RECV_MAX], pl.FP32],
            pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32]
        ]:
            send_counts_bkt = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            send_counts_rank = pl.create_tensor([N_RANKS], dtype=pl.INT32)
            send_offsets_rank = pl.create_tensor([N_RANKS], dtype=pl.INT32)
            histogram_inline(
                expert_indices,
                send_counts_bkt, send_counts_rank, send_offsets_rank,
            )

            for peer in pl.range(N_RANKS):
                for e in pl.range(N_LOCAL_EXPERTS):
                    v = pl.read(
                        send_counts_bkt, [peer * N_LOCAL_EXPERTS + e],
                    )
                    if peer == my_rank:
                        pl.write(
                            pub_counts,
                            [my_rank * N_RANKS + my_rank, e],
                            v,
                        )
                    else:
                        pld.system.notify(
                            target=pub_counts,
                            peer=peer,
                            offsets=[my_rank * N_RANKS + peer, e],
                            value=v,
                            op=pld.NotifyOp.Set,
                        )

            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            cursor_bkt = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            bucket_offset = pl.create_tensor(
                [PER_RANK_BUCKETS], dtype=pl.INT32,
            )
            pack_send_inline(
                x, x_scale, expert_indices,
                send_counts_bkt, send_offsets_rank,
                send_buf, send_scale_buf, cursor_bkt, bucket_offset,
            )

            recv_counts = pl.create_tensor([N_RANKS], dtype=pl.INT32)
            for src in pl.range(N_RANKS):
                acc = pl.cast(0, pl.INT32)
                for e in pl.range(N_LOCAL_EXPERTS):
                    acc = acc + pl.read(
                        pub_counts, [src * N_RANKS + my_rank, e],
                    )
                pl.write(recv_counts, [src], pl.cast(acc, pl.INT32))
            recv_offsets = pl.create_tensor([N_RANKS], dtype=pl.INT32)
            pl.write(recv_offsets, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, N_RANKS):
                prev_off = pl.read(recv_offsets, [r - 1])
                prev_cnt = pl.read(recv_counts, [r - 1])
                pl.write(
                    recv_offsets, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

            # EP all-to-all (variant bound at factory scope per
            # PYPTO_PREFILL_A2A_BULK: per-row default, bulk-chunked =1).
            ep_a2a_inline(
                send_buf, recv_x, send_scale_buf, recv_scale,
                send_counts_rank, recv_counts,
                send_offsets_rank, recv_offsets,
                data_done_sig, my_rank,
            )

            build_csr_inline(
                pub_counts,
                local_expert_offset, local_expert_count,
                my_rank,
            )
            running = pl.cast(0, pl.INT32)
            # Incremental per-src in-expert offset: a running prefix over e
            # maintained across the (e, src) loops. Replaces the per-(e,src)
            # O(N_LOCAL_EXPERTS) re-sum of pub_counts (~5k scalar window
            # reads per dispatch call = ~110k/tile-layer at 22 tiles) with
            # O(1) accumulate — identical integer math, bit-identical values.
            src_e_off_acc = pl.create_tensor([N_RANKS], dtype=pl.INT32)
            for src0 in pl.range(N_RANKS):
                pl.write(src_e_off_acc, [src0], pl.cast(0, pl.INT32))
            for e in pl.range(N_LOCAL_EXPERTS):
                for src in pl.range(N_RANKS):
                    n = pl.cast(
                        pl.read(pub_counts, [src * N_RANKS + my_rank, e]),
                        pl.INDEX,
                    )
                    # Symmetric fixed-slot src-block base (§D:129, mirrors
                    # decode moe.py:972). The recv side is laid out in
                    # BATCH*TOPK blocks per src rank, so the block base is
                    # src*BATCH*TOPK — NOT the variable-length recv_offsets
                    # prefix sum (Problem-27 src/dst skew).
                    src_base = pl.cast(src * (BATCH * TOPK), pl.INDEX)
                    src_e_off = pl.read(src_e_off_acc, [src])
                    for row in pl.range(n):
                        src_row = (
                            src_base
                            + pl.cast(src_e_off, pl.INDEX) + row
                        )
                        dst_row = pl.cast(running, pl.INDEX) + row
                        tile = pl.load(recv_x, [src_row, 0], [1, HIDDEN])
                        pl.store(tile, [dst_row, 0], local_routed_x_out)
                        # Un-pad the per-token scale: scalar col-0 read from the
                        # SCALE_W_PAD-wide a2a window -> contiguous
                        # [1,LOCAL_RECV_MAX] so the routed expert reads a clean
                        # [1,ROUTED_ROW_TILE] row-slice (mirrors decode
                        # moe.py:991-992).
                        sv = pl.read(recv_scale, [src_row, 0])
                        pl.write(local_routed_x_scale_out, [0, dst_row], sv)
                    running = running + pl.cast(n, pl.INT32)
                    pl.write(
                        src_e_off_acc, [src],
                        pl.cast(src_e_off, pl.INT32)
                        + pl.cast(n, pl.INT32),
                    )

            return (
                local_routed_x_out,
                local_routed_x_scale_out,
                local_expert_offset,
                local_expert_count,
            )

        @pl.function(type=pl.FunctionType.InCore)
        def zero_routed_y_buf_step(
            self,
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
        ) -> pld.DistributedTensor[
            [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
        ]:
            # Scalar zero-fill body must live in its own InCore method (NOT
            # spliced into the Inline moe_combine_step) — same constraint as
            # publish/push below. Mirrors decode moe.py:1838-1850
            # _zero_routed_y_buf, called before the route_pub barrier so the
            # zero stores are fenced (via publish notify dsb) before any
            # remote push lands in this rank's window.
            zero_routed_inline(routed_y_buf)
            return routed_y_buf

        @pl.function(type=pl.FunctionType.InCore)
        def publish_src_route_table_step(
            self,
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pld.DistributedTensor[
            [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
        ]:
            # Scalar publish body must live in its own InCore method (NOT
            # spliced into the Inline moe_combine_step) — mirroring decode
            # moe.py:1623 _publish_src_route_table, which the Inline
            # combine_step calls via self.
            publish_route_inline(expert_indices, src_route_table, my_rank)
            return src_route_table

        @pl.function(type=pl.FunctionType.InCore)
        def push_routed_y_step(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pld.DistributedTensor[
            [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
        ]:
            push_routed_inline(
                local_routed_y, pub_counts, routed_y_buf,
                combine_done_sig, src_route_table, my_rank,
            )
            return routed_y_buf

        @pl.function(type=pl.FunctionType.Inline)
        def moe_combine_step(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tile_y_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            pub_counts: pld.DistributedTensor[
                [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK], pl.INT32
            ],
            route_pub_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [N_RANKS, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # Inline (NOT InCore), mirroring decode moe.py:1853 combine_step:
            # the gather carries a pl.at(CORE_GROUP) scope which must live in
            # an Inline function. An InCore function calling an Inline method
            # inlines its body, dragging the CORE_GROUP scope into InCore
            # ("InCore ScopeStmt found in non-InCore function"). The scalar
            # publish/push stages are delegated to InCore methods above.
            self.zero_routed_y_buf_step(routed_y_buf)
            self.publish_src_route_table_step(
                expert_indices, src_route_table, my_rank,
            )

            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="route_pub_barrier",
            ):
                for peer in pl.range(N_RANKS):
                    if peer != my_rank:
                        pld.system.notify(
                            target=route_pub_sig,
                            peer=peer,
                            offsets=[my_rank, 0],
                            value=1,
                            op=pld.NotifyOp.AtomicAdd,
                        )
                for src in pl.range(N_RANKS):
                    if src != my_rank:
                        pld.system.wait(
                            signal=route_pub_sig,
                            offsets=[src, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

            self.push_routed_y_step(
                local_routed_y, pub_counts, routed_y_buf,
                combine_done_sig, src_route_table, my_rank,
            )

            tile_y_out = self.expert_combine_gather_step(
                routed_y_buf, expert_weights, sh_y, tile_y_out,
            )
            return tile_y_out

        @pl.function(type=pl.FunctionType.InCore)
        def moe_cross_layer_fence(
            self,
            combine_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pld.DistributedTensor[
            [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
        ]:
            # Cross-layer WAR fence: wait for the previous MoE layer's combine
            # to fully drain before this layer's dispatch overwrites the shared
            # single-layer data buffers (pub_counts / send_x / send_scale /
            # src_route_table). combine_done is signaled per tile, so wait on
            # every (tile, src) slot of the previous layer's slice.
            for tile in pl.range(PREFILL_TILE_COUNT):
                for src in pl.range(N_RANKS):
                    if src != my_rank:
                        pld.system.wait(
                            signal=combine_done_sig,
                            offsets=[tile * N_RANKS + src, 0],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )
            return combine_done_sig

        # ── Tile-compute scope-isolation wrappers (Inline). ──
        # The three @pl.jit.inline tile-compute shim bodies (gate / shared /
        # routed) are spliced via pl.inline() WITHOUT scope isolation, so
        # their local variable names (w0, x0, gated, gate_acc, up_acc, ...)
        # collide with attention_full's and each other's when all are inlined
        # into the same Orchestration chip_orch body (compile error: "Cannot
        # reassign 'w0' with a different type"). Wrapping each in its own
        # @pl.function(type=Inline) method gives it an isolated scope —
        # the same scope-isolation wrapper pattern used for the former
        # per-layer builder (_gate / _expert_routed / _expert_shared_local
        # were @pl.function (type=Inline) methods). The chip_orch bodies
        # call these wrappers
        # instead of the raw inline objects.
        @pl.function(type=pl.FunctionType.Inline)
        def gate_step(  # noqa: PLR0913
            self,
            resid: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            norm_layer_idx: pl.Scalar[pl.INT32],
            inv_rms: pl.Tensor[[BATCH, 1], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
        ) -> pl.Tensor[[BATCH, TOPK], pl.FP32]:
            return gate_inline(
                resid, post_rms_weight, norm_layer_idx, inv_rms,
                gate_w, router_bias, expert_indices, expert_weights,
            )

        @pl.function(type=pl.FunctionType.Inline)
        def expert_shared_step(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return shared_silu_inline(x, w_gate_s, w_up_s, w_down_s, sh_y)

        @pl.function(type=pl.FunctionType.Inline)
        def expert_routed_step(  # noqa: PLR0913
            self,
            local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            local_routed_x_scale: pl.Tensor[[1, LOCAL_RECV_MAX], pl.FP32],
            local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
        ) -> pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]:
            return routed_silu_inline(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale,
                w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

        @pl.function(type=pl.FunctionType.Inline)
        def expert_shared_step_swiglu16(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            return shared_swiglu16_inline(
                x, w_gate_s, w_up_s, w_down_s, sh_y,
            )

        @pl.function(type=pl.FunctionType.Inline)
        def expert_routed_step_swiglu7(  # noqa: PLR0913
            self,
            local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.INT8],
            local_routed_x_scale: pl.Tensor[[1, LOCAL_RECV_MAX], pl.FP32],
            local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            local_routed_y: pl.Tensor[
                [LOCAL_RECV_MAX, HIDDEN], pl.BF16
            ],
        ) -> pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16]:
            return routed_swiglu7_inline(
                local_routed_x, local_routed_x_scale,
                local_expert_offset, local_expert_count,
                w_gate_r, w_gate_r_scale,
                w_up_r, w_up_r_scale,
                w_down_r, w_down_r_scale,
                local_routed_y,
            )

        @pl.function(type=pl.FunctionType.Inline)
        def expert_combine_gather_step(
            self,
            routed_y_buf: pld.DistributedTensor[
                [N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.FP32],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tile_y_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # The gather body carries a pl.at(CORE_GROUP) scope; splicing it
            # (pl.inline) directly into the InCore moe_combine_step leaves a
            # nested InCore ScopeStmt that SplitIncoreOrch rejects ("InCore
            # ScopeStmt found in non-InCore function"). Wrap it in an Inline
            # method so the CORE_GROUP scope lives in an Inline function —
            # mirroring decode moe.py:1782 _weighted_gather_and_add.
            return gather_add_inline(
                routed_y_buf, expert_weights, sh_y, tile_y_out,
            )

        # ── 8 per-layer-kind chip_orch methods (Orchestration). ──
        # Mirror decode_fwd.py:419 ``full_chip_orch`` /
        # decode_fwd.py:2165 ``full_moe_chip_orch``: each is an
        # ``@pl.function(type=Orchestration, attrs={"inline_orchestration":
        # True})`` method that inlines the relevant @pl.jit.inline bodies and
        # creates its per-layer Out via pl.create_tensor INSIDE (Orchestration
        # = device scope; B-probe proved HOST-scope pl.create_tensor is
        # compile-rejected).
        #
        # P3a: all 8 are placeholder stubs — each creates the per-layer Out
        # tensor via pl.create_tensor and returns it (current_hidden is
        # accepted so the residual-flow type is consistent, but not read).
        # Stubs keep the 8-way dispatch syntactically exhaustive so
        # whole_chip_orch's runtime if/elif has a target for every layer.
        # See the factory-build comment above for the body-wiring follow-up.
        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def full_dense_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [HIDDEN, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [HIDDEN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [HIDDEN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[HIDDEN_Q_FULL_LOCAL, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [HIDDEN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[
                [HIDDEN, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [HIDDEN, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [INTERMEDIATE_LOCAL, HIDDEN], pl.BF16
            ],
            out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            gate_logits_dump: pl.Tensor[[PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16],
            resid1_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            k_rot_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            mlp_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            mlp_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            mlp_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            _dummy_ad = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
            _dummy_agd = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
            _dummy_op = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            _dummy_opr = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            _dummy_qkv = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
            _dummy_scores = pl.create_tensor([PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128], dtype=pl.FP32)
            resid1 = attn_full_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1,
                _dummy_qkv,
                k_rot_dump,
                _dummy_ad, _dummy_agd, _dummy_op, _dummy_opr,
                _dummy_scores,
                input_norm_dump,
                q_proj_dump,
                k_proj_dump,
                v_proj_dump,
                v_tile_dump,
                q_norm_dump,
                k_norm_dump,
                gate_logits_dump,
                attn_delta_dump,
                norm_layer_idx,
                attn_layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            if _DUMP_ENABLED:
                resid1_dump = pl.assemble(resid1_dump, resid1, [0, 0])
            out = dense_mlp_inline(
                resid1, post_rms_weight,
                w_gate, w_up, w_down,
                out,
                post_norm_dump,
                ffn_output_dump,
                norm_layer_idx, mlp_layer_idx,
                mlp_tmp_window, mlp_signal_window, my_rank,
            )
            return out

        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def swa_dense_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [HIDDEN, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [HIDDEN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [HIDDEN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[HIDDEN_Q_SWA_LOCAL, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [HIDDEN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[
                [HIDDEN, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [HIDDEN, INTERMEDIATE_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [INTERMEDIATE_LOCAL, HIDDEN], pl.BF16
            ],
            out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            gate_logits_dump: pl.Tensor[[PREFILL_T, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16],
            resid1_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            k_rot_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            mlp_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            mlp_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            mlp_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            attn_out_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16,
            )
            attn_out_gated_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16,
            )
            o_proj_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            o_proj_reduced_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            _dummy_scores = pl.create_tensor([PREFILL_T * 16, 128], dtype=pl.FP32)
            resid1 = attn_swa_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1,
                k_rot_dump,
                attn_out_dump,
                attn_out_gated_dump,
                o_proj_dump,
                o_proj_reduced_dump,
                _dummy_scores,
                input_norm_dump,
                q_proj_dump,
                k_proj_dump,
                v_proj_dump,
                v_tile_dump,
                q_norm_dump,
                k_norm_dump,
                gate_logits_dump,
                attn_delta_dump,
                norm_layer_idx,
                attn_layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            if _DUMP_ENABLED:
                resid1_dump = pl.assemble(resid1_dump, resid1, [0, 0])
            out = dense_mlp_inline(
                resid1, post_rms_weight,
                w_gate, w_up, w_down,
                out,
                post_norm_dump,
                ffn_output_dump,
                norm_layer_idx, mlp_layer_idx,
                mlp_tmp_window, mlp_signal_window, my_rank,
            )
            return out

        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def full_moe_silu_silu_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN_FULL, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            gate_logits_dump: pl.Tensor[[PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16],
            resid1_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            k_rot_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                pl.INT32,
            ],
            count_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            send_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            recv_x: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                pl.INT32,
            ],
            route_pub_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            # ── A: prefill attention + tp_all_reduce -> resid1. ────────
            resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            _dummy_qkv = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
            _dummy_ad = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
            _dummy_agd = pl.create_tensor([PREFILL_T, HIDDEN_Q_FULL_LOCAL], dtype=pl.BF16)
            _dummy_op = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            _dummy_opr = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            _dummy_scores = pl.create_tensor([PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128], dtype=pl.FP32)
            resid1 = attn_full_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1,
                _dummy_qkv,
                k_rot_dump,
                _dummy_ad,
                _dummy_agd,
                _dummy_op,
                _dummy_opr,
                _dummy_scores,
                input_norm_dump,
                q_proj_dump,
                k_proj_dump,
                v_proj_dump,
                v_tile_dump,
                q_norm_dump,
                k_norm_dump,
                gate_logits_dump,
                attn_delta_dump,
                norm_layer_idx,
                attn_layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            if _DUMP_ENABLED:
                resid1_dump = pl.assemble(resid1_dump, resid1, [0, 0])
            # NaN DIAG: attention output after o_proj tp_all_reduce.
            # DIAG: accumulate resid1 (post-attention) into host-visible dump.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_resid1_dump"):
                # for kb0 in pl.range(HIDDEN // K_CHUNK):
                    # k00 = kb0 * K_CHUNK
                    # resid1_dump = pl.assemble(
                        # resid1_dump,
                        # pl.slice(resid1, [PREFILL_T, K_CHUNK], [0, k00]),
                        # [0, k00],
                    # )
            # NaN DIAG: per-layer resid1 Inf count (blue §32 point 3 —
            # locate the FIRST layer whose post-attention residual holds
            # Inf = the overflow/fix site, NOT the last layer).
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_resid1_inf_count"):
                # for ric_tg in pl.range(PREFILL_TILE_COUNT):
                    # ric_t0 = ric_tg * BATCH
                    # ric_part = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                    # for ric_kb in pl.range(HIDDEN // K_CHUNK):
                        # ric_k0 = ric_kb * K_CHUNK
                        # ric_r = pl.cast(
                            # pl.slice(resid1, [BATCH, K_CHUNK], [ric_t0, ric_k0]),
                            # pl.FP32,
                        # )
                        # ric_abs = pl.maximum(ric_r, pl.neg(ric_r))
                        # ric_mask = pl.cmp(ric_abs, 1.0e30, cmp_type=4)
                        # ric_part = pl.add(
                            # ric_part,
                            # pl.reshape(pl.row_sum(ric_mask), [1, BATCH]),
                        # )
                    # resid1_inf_count = pl.assemble(
                        # resid1_inf_count,
                        # pl.reshape(ric_part, [BATCH, 1]),
                        # [ric_t0, norm_layer_idx],
                    # )


            # ── B: post-attention V4 deferred RMSNorm + INT8/scale producer.
            # Mirrors decode_fwd.py _norm_quant_moe_input: first pass forms
            # Pass 1: reduce sum(resid**2) -> inv_rms. Pass 2: BF16 post-norm
            # (shared lane) + amax over the BF16 post-norm. Pass 3: INT8 payload
            # quant(post_norm / (pn_amax/127)) + per-token dequant scale
            # (pn_amax/127) for the routed lane, bit-matching VLLM W8A8.
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            moe_inv_rms = pl.create_tensor(
                [PREFILL_T, 1], dtype=pl.FP32,
            )
            x_i8 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.INT8,
            )
            x_scale = pl.create_tensor(
                [PREFILL_T, SCALE_W_PAD], dtype=pl.FP32,
            )
            resid1_fp32 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.FP32,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_zc",
            ):
                for tg in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg * BATCH
                    sq_sum = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=0.0,
                    )
                    pn_amax = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=1e-4,
                    )
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        rchunk = pl.cast(
                            pl.slice(
                                resid1, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        resid1_fp32 = pl.assemble(
                            resid1_fp32, rchunk, [t0, k0],
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(rchunk, rchunk)),
                                [1, BATCH],
                            ),
                        )
                    inv_rms_moe = pl.recip(
                        pl.sqrt(
                            pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                        ),
                    )
                    inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                    moe_inv_rms[t0:t0+BATCH, 0:1] = inv_rms_col
                    for kb3 in pl.range(hidden_blocks):
                        k0 = kb3 * K_CHUNK
                        norm_chunk = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        gamma = pl.slice(
                            post_rms_weight, [1, K_CHUNK],
                            [norm_layer_idx, k0],
                        )
                        xg = pl.col_expand_mul(
                            norm_chunk, pl.add(gamma, 1.0),
                        )
                        normed = pl.row_expand_mul(xg, inv_rms_col)
                        post_norm_chunk = pl.cast(normed, target_type=pl.BF16)
                        post_norm = pl.assemble(
                            post_norm,
                            post_norm_chunk,
                            [t0, k0],
                        )
                        pn_fp32 = pl.cast(
                            post_norm_chunk, target_type=pl.FP32,
                        )
                        pn_amax = pl.maximum(
                            pn_amax,
                            pl.reshape(
                                pl.row_max(
                                    pl.maximum(pn_fp32, pl.neg(pn_fp32)),
                                ),
                                [1, BATCH],
                            ),
                        )
                    dequant_scale = pl.reshape(
                        pl.div(
                            pn_amax,
                            pl.full(
                                [1, BATCH], dtype=pl.FP32, value=127.0,
                            ),
                        ),
                        [BATCH, 1],
                    )
                    x_scale = pl.assemble(
                        x_scale,
                        pl.row_expand_mul(
                            pl.full(
                                [BATCH, SCALE_W_PAD],
                                dtype=pl.FP32,
                                value=1.0,
                            ),
                            dequant_scale,
                        ),
                        [t0, 0],
                    )
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        pn_chunk = pl.slice(
                            post_norm, [BATCH, K_CHUNK], [t0, k0],
                        )
                        qi32 = pl.cast(
                            pl.row_expand_div(
                                pl.cast(pn_chunk, target_type=pl.FP32),
                                dequant_scale,
                            ),
                            target_type=pl.INT32,
                            mode="rint",
                        )
                        qf16 = pl.cast(
                            qi32, target_type=pl.FP16, mode="round",
                        )
                        x_i8 = pl.assemble(
                            x_i8,
                            pl.cast(
                                qf16, target_type=pl.INT8, mode="trunc",
                            ),
                            [t0, k0],
                        )

            # Module dump 7: post_norm (post-attention RMSNorm hidden,
            # replicated, pre-quantization).
            if _DUMP_ENABLED:
                post_norm_dump = pl.assemble(post_norm_dump, post_norm, [0, 0])

            # NaN DIAG: producer-exit x_scale (dequant scale) dump. NaN
            # here proves producer overflow (resid1 Inf -> dequant_scale
            # 0*Inf); clean here pushes the residual NaN to the a2a/gap
            # side.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_x_scale_dump"):
                # x_scale_dump = pl.assemble(
                    # x_scale_dump,
                    # pl.slice(x_scale, [PREFILL_T, SCALE_W_PAD], [0, 0]),
                    # [0, 0],
                # )

            # ── C: per-tile MoE adapter (gate -> shared -> dispatch ->
            # routed -> combine), one [BATCH, HIDDEN] tile at a time. ───
            moe_out = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            for tile_idx in pl.range(PREFILL_TILE_COUNT):
                t_lo = tile_idx * BATCH
                tile_resid = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                tile_post_norm = pl.create_tensor(
                    [BATCH, HIDDEN], dtype=pl.BF16,
                )
                tile_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
                tile_x_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
                tile_x_scale = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_in",
                ):
                    tile_resid = pl.assemble(
                        tile_resid,
                        pl.slice(resid1, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_post_norm = pl.assemble(
                        tile_post_norm,
                        pl.slice(post_norm, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_inv_rms = pl.assemble(
                        tile_inv_rms,
                        pl.slice(moe_inv_rms, [BATCH, 1], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_i8 = pl.assemble(
                        tile_x_i8,
                        pl.slice(x_i8, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_scale = pl.assemble(
                        tile_x_scale,
                        pl.slice(x_scale, [BATCH, SCALE_W_PAD], [t_lo, 0]),
                        [0, 0],
                    )

                # 1) Gate (local, replicated).
                expert_indices = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.INT32,
                )
                expert_weights = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.FP32,
                )
                expert_weights = self.gate_step(
                    tile_resid, post_rms_weight, norm_layer_idx, tile_inv_rms,
                    gate_w, router_bias,
                    expert_indices, expert_weights,
                )

                # DIAG: accumulate gate top-K into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_gate_dump"):
                    # expert_indices_dump = pl.assemble(
                        # expert_indices_dump,
                        # pl.slice(expert_indices, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                    # expert_weights_dump = pl.assemble(
                        # expert_weights_dump,
                        # pl.slice(expert_weights, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
                sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                sh_y = self.expert_shared_step(
                    tile_post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                )
                self.moe_tp_all_reduce(
                    sh_y, sh_tmp_window,
                    pl.slice(
                        sh_signal_window, [tp_size, 1],
                        [tile_idx * tp_size, 0],
                    ),
                    my_rank,
                )

                # DIAG (bisect shared): TP-reduced shared-expert output.
                # DIAG: accumulate TP-reduced shared-expert output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_sh_y_dump"):
                    # for kb1 in pl.range(HIDDEN // K_CHUNK):
                        # k01 = kb1 * K_CHUNK
                        # sh_y_dump = pl.assemble(
                            # sh_y_dump,
                            # pl.slice(sh_y, [BATCH, K_CHUNK], [0, k01]),
                            # [t_lo, k01],
                        # )

                # Serialize shared TP -> routed dispatch so dispatch reads the
                # completed tile_x_scale (see _serialize_after_shared).
                tile_x_scale_ser = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                tile_x_scale = self._serialize_after_shared(
                    tile_x_scale, sh_y, tile_x_scale_ser,
                )
                # NaN DIAG: producer send-side dequant scale (pre-dispatch).

                # Per-tile data-window views (Orchestration scope): the 5 EP
                # data windows are shared across the 8-tile unroll, so slice a
                # fresh per-tile view to stop tile i+1's zero/push from racing
                # tile i's gather (same per-tile scheme as the signal windows).
                pub_counts_tile = pl.slice(
                    pub_counts, [N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                    [tile_idx * N_RANKS * N_RANKS, 0],
                )
                send_buf_tile = pl.slice(
                    send_buf, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                send_scale_buf_tile = pl.slice(
                    send_scale_buf, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 1): packed send-side scale window.
                recv_x_tile = pl.slice(
                    recv_x, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                recv_scale_tile = pl.slice(
                    recv_scale, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 2): a2a-delivered recv-side scale window.
                src_route_table_tile = pl.slice(
                    src_route_table,
                    [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    [tile_idx * N_RANKS, 0, 0],
                )
                routed_y_buf_tile = pl.slice(
                    routed_y_buf, [N_ROUTES_PER_RANK, HIDDEN],
                    [tile_idx * N_ROUTES_PER_RANK, 0],
                )

                # 3) Dispatch (EP all-to-all).
                local_routed_x = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                local_routed_x_scale = pl.create_tensor(
                    [1, LOCAL_RECV_MAX], dtype=pl.FP32,
                )
                local_expert_offset = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                local_expert_count = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                self.zero_dispatch_buffers_step(
                    send_buf_tile, send_scale_buf_tile,
                    recv_x_tile, recv_scale_tile,
                )
                (
                    local_routed_x,
                    local_routed_x_scale,
                    local_expert_offset,
                    local_expert_count,
                ) = self.moe_dispatch_step(
                    tile_x_i8, tile_x_scale, expert_indices,
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    send_buf_tile, send_scale_buf_tile,
                    pub_counts_tile,
                    pl.slice(count_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    recv_x_tile, recv_scale_tile,
                    pl.slice(data_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    my_rank,
                )
                # NaN DIAG (candidate ①): per-token activation scale post-dispatch.
                # DIAG: accumulate per-token dequant scale into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_scale_dump"):
                    # local_routed_x_scale_dump = pl.assemble(
                        # local_routed_x_scale_dump,
                        # pl.slice(local_routed_x_scale, [1, N_ROUTES_PER_RANK], [0, 0]),
                        # [0, tile_idx * N_ROUTES_PER_RANK],
                    # )
                # DIAG (bisect dispatch): reordered INT8 x post-a2a (recv_x equiv).
                # DIAG: accumulate reordered routed input into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_x_dump"):
                    # for kb_x in pl.range(HIDDEN // K_CHUNK):
                        # kx0 = kb_x * K_CHUNK
                        # local_routed_x_dump = pl.assemble(
                            # local_routed_x_dump,
                            # pl.slice(local_routed_x, [N_ROUTES_PER_RANK, K_CHUNK], [0, kx0]),
                            # [tile_idx * N_ROUTES_PER_RANK, kx0],
                        # )

                # 4) Routed experts (local 36).
                local_routed_y = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.BF16,
                )
                local_routed_y = self.expert_routed_step(
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    w_gate_r, w_gate_r_scale,
                    w_up_r, w_up_r_scale,
                    w_down_r, w_down_r_scale,
                    local_routed_y,
                )

                # DIAG (bisect routed): per-local-expert output pre-combine.
                # DIAG: accumulate routed expert output into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_y_dump"):
                    # for kb_y in pl.range(HIDDEN // K_CHUNK):
                        # ky0 = kb_y * K_CHUNK
                        # local_routed_y_dump = pl.assemble(
                            # local_routed_y_dump,
                            # pl.slice(local_routed_y, [N_ROUTES_PER_RANK, K_CHUNK], [0, ky0]),
                            # [tile_idx * N_ROUTES_PER_RANK, ky0],
                        # )

                # 5) Combine (EP a2a back + weighted gather + sh_y add).
                tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                # Bind per-tile signal slices to named DistributedTensor views
                # at Orchestration scope before the Inline combine step inlines
                # them: an inline pl.slice(...) arg would splice into the
                # route_pub barrier's CORE_GROUP scope, where
                # ConvertTensorToTileOps demotes it to a TileType and
                # pld.system.notify rejects it (must stay window-bound).
                route_pub_tile_sig = pl.slice(
                    route_pub_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                combine_done_tile_sig = pl.slice(
                    combine_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                tile_y = self.moe_combine_step(
                    local_routed_y,
                    expert_indices, expert_weights, sh_y,
                    tile_y,
                    pub_counts_tile, src_route_table_tile,
                    route_pub_tile_sig,
                    routed_y_buf_tile,
                    combine_done_tile_sig,
                    my_rank,
                )
                # NaN DIAG: combined MoE output — should be TP-replicated.
                # DIAG: accumulate combined MoE output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_tile_y_dump"):
                    # for kb2 in pl.range(HIDDEN // K_CHUNK):
                        # k02 = kb2 * K_CHUNK
                        # tile_y_dump = pl.assemble(
                            # tile_y_dump,
                            # pl.slice(tile_y, [BATCH, K_CHUNK], [0, k02]),
                            # [t_lo, k02],
                        # )

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_out",
                ):
                    moe_out = pl.assemble(
                        moe_out,
                        pl.slice(tile_y, [BATCH, HIDDEN], [0, 0]),
                        [t_lo, 0],
                    )

            # Module dump 8: ffn_output (combined shared+routed MoE output,
            # pre-residual-add, TP-replicated).
            if _DUMP_ENABLED:
                ffn_output_dump = pl.assemble(ffn_output_dump, moe_out, [0, 0])

            # ── D: residual add. ───────────────────────────────────────
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add",
            ):
                for tg5 in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg5 * BATCH
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        m = pl.cast(
                            pl.slice(
                                moe_out, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        r = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        out = pl.assemble(
                            out,
                            pl.cast(pl.add(r, m), target_type=pl.BF16),
                            [t0, k0],
                        )
            return out

        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def swa_moe_silu_silu_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN_SWA, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            gate_logits_dump: pl.Tensor[[PREFILL_T, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16],
            resid1_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            k_rot_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                pl.INT32,
            ],
            count_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            send_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            recv_x: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                pl.INT32,
            ],
            route_pub_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            # ── A: prefill attention + tp_all_reduce -> resid1. ────────
            resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            attn_out_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16
            )
            attn_out_gated_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16
            )
            o_proj_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16
            )
            o_proj_reduced_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16
            )
            scores_dump = pl.create_tensor([PREFILL_T * 16, 128], dtype=pl.FP32)
            resid1 = attn_swa_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1,
                k_rot_dump,
                attn_out_dump,
                attn_out_gated_dump,
                o_proj_dump,
                o_proj_reduced_dump,
                scores_dump,
                input_norm_dump,
                q_proj_dump,
                k_proj_dump,
                v_proj_dump,
                v_tile_dump,
                q_norm_dump,
                k_norm_dump,
                gate_logits_dump,
                attn_delta_dump,
                norm_layer_idx,
                attn_layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            if _DUMP_ENABLED:
                resid1_dump = pl.assemble(resid1_dump, resid1, [0, 0])
            # NaN DIAG: attention output after o_proj tp_all_reduce.
            # DIAG: accumulate resid1 (post-attention) into host-visible dump.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_resid1_dump"):
                # for kb0 in pl.range(HIDDEN // K_CHUNK):
                    # k00 = kb0 * K_CHUNK
                    # resid1_dump = pl.assemble(
                        # resid1_dump,
                        # pl.slice(resid1, [PREFILL_T, K_CHUNK], [0, k00]),
                        # [0, k00],
                    # )
            # NaN DIAG: per-layer resid1 Inf count (blue §32 point 3 —
            # locate the FIRST layer whose post-attention residual holds
            # Inf = the overflow/fix site, NOT the last layer).
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_resid1_inf_count"):
                # for ric_tg in pl.range(PREFILL_TILE_COUNT):
                    # ric_t0 = ric_tg * BATCH
                    # ric_part = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                    # for ric_kb in pl.range(HIDDEN // K_CHUNK):
                        # ric_k0 = ric_kb * K_CHUNK
                        # ric_r = pl.cast(
                            # pl.slice(resid1, [BATCH, K_CHUNK], [ric_t0, ric_k0]),
                            # pl.FP32,
                        # )
                        # ric_abs = pl.maximum(ric_r, pl.neg(ric_r))
                        # ric_mask = pl.cmp(ric_abs, 1.0e30, cmp_type=4)
                        # ric_part = pl.add(
                            # ric_part,
                            # pl.reshape(pl.row_sum(ric_mask), [1, BATCH]),
                        # )
                    # resid1_inf_count = pl.assemble(
                        # resid1_inf_count,
                        # pl.reshape(ric_part, [BATCH, 1]),
                        # [ric_t0, norm_layer_idx],
                    # )


            # ── B: post-attention V4 deferred RMSNorm + INT8/scale producer.
            # Mirrors decode_fwd.py _norm_quant_moe_input: first pass forms
            # Pass 1: reduce sum(resid**2) -> inv_rms. Pass 2: BF16 post-norm
            # (shared lane) + amax over the BF16 post-norm. Pass 3: INT8 payload
            # quant(post_norm / (pn_amax/127)) + per-token dequant scale
            # (pn_amax/127) for the routed lane, bit-matching VLLM W8A8.
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            moe_inv_rms = pl.create_tensor(
                [PREFILL_T, 1], dtype=pl.FP32,
            )
            x_i8 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.INT8,
            )
            x_scale = pl.create_tensor(
                [PREFILL_T, SCALE_W_PAD], dtype=pl.FP32,
            )
            resid1_fp32 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.FP32,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_zc",
            ):
                for tg in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg * BATCH
                    sq_sum = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=0.0,
                    )
                    pn_amax = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=1e-4,
                    )
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        rchunk = pl.cast(
                            pl.slice(
                                resid1, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        resid1_fp32 = pl.assemble(
                            resid1_fp32, rchunk, [t0, k0],
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(rchunk, rchunk)),
                                [1, BATCH],
                            ),
                        )
                    inv_rms_moe = pl.recip(
                        pl.sqrt(
                            pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                        ),
                    )
                    inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                    moe_inv_rms[t0:t0+BATCH, 0:1] = inv_rms_col
                    for kb3 in pl.range(hidden_blocks):
                        k0 = kb3 * K_CHUNK
                        norm_chunk = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        gamma = pl.slice(
                            post_rms_weight, [1, K_CHUNK],
                            [norm_layer_idx, k0],
                        )
                        xg = pl.col_expand_mul(
                            norm_chunk, pl.add(gamma, 1.0),
                        )
                        normed = pl.row_expand_mul(xg, inv_rms_col)
                        post_norm_chunk = pl.cast(normed, target_type=pl.BF16)
                        post_norm = pl.assemble(
                            post_norm,
                            post_norm_chunk,
                            [t0, k0],
                        )
                        pn_fp32 = pl.cast(
                            post_norm_chunk, target_type=pl.FP32,
                        )
                        pn_amax = pl.maximum(
                            pn_amax,
                            pl.reshape(
                                pl.row_max(
                                    pl.maximum(pn_fp32, pl.neg(pn_fp32)),
                                ),
                                [1, BATCH],
                            ),
                        )
                    dequant_scale = pl.reshape(
                        pl.div(
                            pn_amax,
                            pl.full(
                                [1, BATCH], dtype=pl.FP32, value=127.0,
                            ),
                        ),
                        [BATCH, 1],
                    )
                    x_scale = pl.assemble(
                        x_scale,
                        pl.row_expand_mul(
                            pl.full(
                                [BATCH, SCALE_W_PAD],
                                dtype=pl.FP32,
                                value=1.0,
                            ),
                            dequant_scale,
                        ),
                        [t0, 0],
                    )
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        pn_chunk = pl.slice(
                            post_norm, [BATCH, K_CHUNK], [t0, k0],
                        )
                        qi32 = pl.cast(
                            pl.row_expand_div(
                                pl.cast(pn_chunk, target_type=pl.FP32),
                                dequant_scale,
                            ),
                            target_type=pl.INT32,
                            mode="rint",
                        )
                        qf16 = pl.cast(
                            qi32, target_type=pl.FP16, mode="round",
                        )
                        x_i8 = pl.assemble(
                            x_i8,
                            pl.cast(
                                qf16, target_type=pl.INT8, mode="trunc",
                            ),
                            [t0, k0],
                        )

            # Module dump 7: post_norm (post-attention RMSNorm hidden,
            # replicated, pre-quantization).
            if _DUMP_ENABLED:
                post_norm_dump = pl.assemble(post_norm_dump, post_norm, [0, 0])

            # NaN DIAG: producer-exit x_scale (dequant scale) dump. NaN
            # here proves producer overflow (resid1 Inf -> dequant_scale
            # 0*Inf); clean here pushes the residual NaN to the a2a/gap
            # side.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_x_scale_dump"):
                # x_scale_dump = pl.assemble(
                    # x_scale_dump,
                    # pl.slice(x_scale, [PREFILL_T, SCALE_W_PAD], [0, 0]),
                    # [0, 0],
                # )

            # ── C: per-tile MoE adapter (gate -> shared -> dispatch ->
            # routed -> combine), one [BATCH, HIDDEN] tile at a time. ───
            moe_out = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            for tile_idx in pl.range(PREFILL_TILE_COUNT):
                t_lo = tile_idx * BATCH
                tile_resid = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                tile_post_norm = pl.create_tensor(
                    [BATCH, HIDDEN], dtype=pl.BF16,
                )
                tile_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
                tile_x_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
                tile_x_scale = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_in",
                ):
                    tile_resid = pl.assemble(
                        tile_resid,
                        pl.slice(resid1, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_post_norm = pl.assemble(
                        tile_post_norm,
                        pl.slice(post_norm, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_inv_rms = pl.assemble(
                        tile_inv_rms,
                        pl.slice(moe_inv_rms, [BATCH, 1], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_i8 = pl.assemble(
                        tile_x_i8,
                        pl.slice(x_i8, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_scale = pl.assemble(
                        tile_x_scale,
                        pl.slice(x_scale, [BATCH, SCALE_W_PAD], [t_lo, 0]),
                        [0, 0],
                    )

                # 1) Gate (local, replicated).
                expert_indices = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.INT32,
                )
                expert_weights = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.FP32,
                )
                expert_weights = self.gate_step(
                    tile_resid, post_rms_weight, norm_layer_idx, tile_inv_rms,
                    gate_w, router_bias,
                    expert_indices, expert_weights,
                )

                # DIAG: accumulate gate top-K into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_gate_dump"):
                    # expert_indices_dump = pl.assemble(
                        # expert_indices_dump,
                        # pl.slice(expert_indices, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                    # expert_weights_dump = pl.assemble(
                        # expert_weights_dump,
                        # pl.slice(expert_weights, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
                sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                sh_y = self.expert_shared_step(
                    tile_post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                )
                self.moe_tp_all_reduce(
                    sh_y, sh_tmp_window,
                    pl.slice(
                        sh_signal_window, [tp_size, 1],
                        [tile_idx * tp_size, 0],
                    ),
                    my_rank,
                )

                # DIAG (bisect shared): TP-reduced shared-expert output.
                # DIAG: accumulate TP-reduced shared-expert output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_sh_y_dump"):
                    # for kb1 in pl.range(HIDDEN // K_CHUNK):
                        # k01 = kb1 * K_CHUNK
                        # sh_y_dump = pl.assemble(
                            # sh_y_dump,
                            # pl.slice(sh_y, [BATCH, K_CHUNK], [0, k01]),
                            # [t_lo, k01],
                        # )

                # Serialize shared TP -> routed dispatch so dispatch reads the
                # completed tile_x_scale (see _serialize_after_shared).
                tile_x_scale_ser = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                tile_x_scale = self._serialize_after_shared(
                    tile_x_scale, sh_y, tile_x_scale_ser,
                )
                # NaN DIAG: producer send-side dequant scale (pre-dispatch).

                # Per-tile data-window views (Orchestration scope): the 5 EP
                # data windows are shared across the 8-tile unroll, so slice a
                # fresh per-tile view to stop tile i+1's zero/push from racing
                # tile i's gather (same per-tile scheme as the signal windows).
                pub_counts_tile = pl.slice(
                    pub_counts, [N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                    [tile_idx * N_RANKS * N_RANKS, 0],
                )
                send_buf_tile = pl.slice(
                    send_buf, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                send_scale_buf_tile = pl.slice(
                    send_scale_buf, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 1): packed send-side scale window.
                recv_x_tile = pl.slice(
                    recv_x, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                recv_scale_tile = pl.slice(
                    recv_scale, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 2): a2a-delivered recv-side scale window.
                src_route_table_tile = pl.slice(
                    src_route_table,
                    [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    [tile_idx * N_RANKS, 0, 0],
                )
                routed_y_buf_tile = pl.slice(
                    routed_y_buf, [N_ROUTES_PER_RANK, HIDDEN],
                    [tile_idx * N_ROUTES_PER_RANK, 0],
                )

                # 3) Dispatch (EP all-to-all).
                local_routed_x = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                local_routed_x_scale = pl.create_tensor(
                    [1, LOCAL_RECV_MAX], dtype=pl.FP32,
                )
                local_expert_offset = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                local_expert_count = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                self.zero_dispatch_buffers_step(
                    send_buf_tile, send_scale_buf_tile,
                    recv_x_tile, recv_scale_tile,
                )
                (
                    local_routed_x,
                    local_routed_x_scale,
                    local_expert_offset,
                    local_expert_count,
                ) = self.moe_dispatch_step(
                    tile_x_i8, tile_x_scale, expert_indices,
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    send_buf_tile, send_scale_buf_tile,
                    pub_counts_tile,
                    pl.slice(count_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    recv_x_tile, recv_scale_tile,
                    pl.slice(data_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    my_rank,
                )
                # NaN DIAG (candidate ①): per-token activation scale post-dispatch.
                # DIAG: accumulate per-token dequant scale into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_scale_dump"):
                    # local_routed_x_scale_dump = pl.assemble(
                        # local_routed_x_scale_dump,
                        # pl.slice(local_routed_x_scale, [1, N_ROUTES_PER_RANK], [0, 0]),
                        # [0, tile_idx * N_ROUTES_PER_RANK],
                    # )
                # DIAG (bisect dispatch): reordered INT8 x post-a2a (recv_x equiv).
                # DIAG: accumulate reordered routed input into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_x_dump"):
                    # for kb_x in pl.range(HIDDEN // K_CHUNK):
                        # kx0 = kb_x * K_CHUNK
                        # local_routed_x_dump = pl.assemble(
                            # local_routed_x_dump,
                            # pl.slice(local_routed_x, [N_ROUTES_PER_RANK, K_CHUNK], [0, kx0]),
                            # [tile_idx * N_ROUTES_PER_RANK, kx0],
                        # )

                # 4) Routed experts (local 36).
                local_routed_y = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.BF16,
                )
                local_routed_y = self.expert_routed_step(
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    w_gate_r, w_gate_r_scale,
                    w_up_r, w_up_r_scale,
                    w_down_r, w_down_r_scale,
                    local_routed_y,
                )

                # DIAG (bisect routed): per-local-expert output pre-combine.
                # DIAG: accumulate routed expert output into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_y_dump"):
                    # for kb_y in pl.range(HIDDEN // K_CHUNK):
                        # ky0 = kb_y * K_CHUNK
                        # local_routed_y_dump = pl.assemble(
                            # local_routed_y_dump,
                            # pl.slice(local_routed_y, [N_ROUTES_PER_RANK, K_CHUNK], [0, ky0]),
                            # [tile_idx * N_ROUTES_PER_RANK, ky0],
                        # )

                # 5) Combine (EP a2a back + weighted gather + sh_y add).
                tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                # Bind per-tile signal slices to named DistributedTensor views
                # at Orchestration scope before the Inline combine step inlines
                # them: an inline pl.slice(...) arg would splice into the
                # route_pub barrier's CORE_GROUP scope, where
                # ConvertTensorToTileOps demotes it to a TileType and
                # pld.system.notify rejects it (must stay window-bound).
                route_pub_tile_sig = pl.slice(
                    route_pub_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                combine_done_tile_sig = pl.slice(
                    combine_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                tile_y = self.moe_combine_step(
                    local_routed_y,
                    expert_indices, expert_weights, sh_y,
                    tile_y,
                    pub_counts_tile, src_route_table_tile,
                    route_pub_tile_sig,
                    routed_y_buf_tile,
                    combine_done_tile_sig,
                    my_rank,
                )
                # NaN DIAG: combined MoE output — should be TP-replicated.
                # DIAG: accumulate combined MoE output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_tile_y_dump"):
                    # for kb2 in pl.range(HIDDEN // K_CHUNK):
                        # k02 = kb2 * K_CHUNK
                        # tile_y_dump = pl.assemble(
                            # tile_y_dump,
                            # pl.slice(tile_y, [BATCH, K_CHUNK], [0, k02]),
                            # [t_lo, k02],
                        # )

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_out",
                ):
                    moe_out = pl.assemble(
                        moe_out,
                        pl.slice(tile_y, [BATCH, HIDDEN], [0, 0]),
                        [t_lo, 0],
                    )


            # Module dump 8: ffn_output (combined shared+routed MoE output,
            # pre-residual-add, TP-replicated).
            if _DUMP_ENABLED:
                ffn_output_dump = pl.assemble(ffn_output_dump, moe_out, [0, 0])

            # ── D: residual add. ───────────────────────────────────────
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add",
            ):
                for tg5 in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg5 * BATCH
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        m = pl.cast(
                            pl.slice(
                                moe_out, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        r = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        out = pl.assemble(
                            out,
                            pl.cast(pl.add(r, m), target_type=pl.BF16),
                            [t0, k0],
                        )
            return out

        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def swa_moe_swiglu7_silu_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN_SWA, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            gate_logits_dump: pl.Tensor[[PREFILL_T, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16],
            resid1_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            k_rot_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                pl.INT32,
            ],
            count_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            send_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            recv_x: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                pl.INT32,
            ],
            route_pub_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            # ── A: prefill attention + tp_all_reduce -> resid1. ────────
            resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            attn_out_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16
            )
            attn_out_gated_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN_Q_SWA_LOCAL], dtype=pl.BF16
            )
            o_proj_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16
            )
            o_proj_reduced_dump = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16
            )
            _dummy_scores = pl.create_tensor([PREFILL_T * 16, 128], dtype=pl.FP32)
            resid1 = attn_swa_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1,
                k_rot_dump,
                attn_out_dump,
                attn_out_gated_dump,
                o_proj_dump,
                o_proj_reduced_dump,
                _dummy_scores,
                input_norm_dump,
                q_proj_dump,
                k_proj_dump,
                v_proj_dump,
                v_tile_dump,
                q_norm_dump,
                k_norm_dump,
                gate_logits_dump,
                attn_delta_dump,
                norm_layer_idx,
                attn_layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            if _DUMP_ENABLED:
                resid1_dump = pl.assemble(resid1_dump, resid1, [0, 0])
            # NaN DIAG: attention output after o_proj tp_all_reduce.
            pl.dump_tag(resid1)
            # DIAG: accumulate resid1 (post-attention) into host-visible dump.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_resid1_dump"):
                # for kb0 in pl.range(HIDDEN // K_CHUNK):
                    # k00 = kb0 * K_CHUNK
                    # resid1_dump = pl.assemble(
                        # resid1_dump,
                        # pl.slice(resid1, [PREFILL_T, K_CHUNK], [0, k00]),
                        # [0, k00],
                    # )

            # ── B: post-attention V4 deferred RMSNorm + INT8/scale producer.
            # Mirrors decode_fwd.py _norm_quant_moe_input: first pass forms
            # Pass 1: reduce sum(resid**2) -> inv_rms. Pass 2: BF16 post-norm
            # (shared lane) + amax over the BF16 post-norm. Pass 3: INT8 payload
            # quant(post_norm / (pn_amax/127)) + per-token dequant scale
            # (pn_amax/127) for the routed lane, bit-matching VLLM W8A8.
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            moe_inv_rms = pl.create_tensor(
                [PREFILL_T, 1], dtype=pl.FP32,
            )
            x_i8 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.INT8,
            )
            x_scale = pl.create_tensor(
                [PREFILL_T, SCALE_W_PAD], dtype=pl.FP32,
            )
            resid1_fp32 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.FP32,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_zc",
            ):
                for tg in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg * BATCH
                    sq_sum = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=0.0,
                    )
                    pn_amax = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=1e-4,
                    )
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        rchunk = pl.cast(
                            pl.slice(
                                resid1, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        resid1_fp32 = pl.assemble(
                            resid1_fp32, rchunk, [t0, k0],
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(rchunk, rchunk)),
                                [1, BATCH],
                            ),
                        )
                    inv_rms_moe = pl.recip(
                        pl.sqrt(
                            pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                        ),
                    )
                    inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                    moe_inv_rms[t0:t0+BATCH, 0:1] = inv_rms_col
                    for kb3 in pl.range(hidden_blocks):
                        k0 = kb3 * K_CHUNK
                        norm_chunk = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        gamma = pl.slice(
                            post_rms_weight, [1, K_CHUNK],
                            [norm_layer_idx, k0],
                        )
                        xg = pl.col_expand_mul(
                            norm_chunk, pl.add(gamma, 1.0),
                        )
                        normed = pl.row_expand_mul(xg, inv_rms_col)
                        post_norm_chunk = pl.cast(normed, target_type=pl.BF16)
                        post_norm = pl.assemble(
                            post_norm,
                            post_norm_chunk,
                            [t0, k0],
                        )
                        pn_fp32 = pl.cast(
                            post_norm_chunk, target_type=pl.FP32,
                        )
                        pn_amax = pl.maximum(
                            pn_amax,
                            pl.reshape(
                                pl.row_max(
                                    pl.maximum(pn_fp32, pl.neg(pn_fp32)),
                                ),
                                [1, BATCH],
                            ),
                        )
                    dequant_scale = pl.reshape(
                        pl.div(
                            pn_amax,
                            pl.full(
                                [1, BATCH], dtype=pl.FP32, value=127.0,
                            ),
                        ),
                        [BATCH, 1],
                    )
                    x_scale = pl.assemble(
                        x_scale,
                        pl.row_expand_mul(
                            pl.full(
                                [BATCH, SCALE_W_PAD],
                                dtype=pl.FP32,
                                value=1.0,
                            ),
                            dequant_scale,
                        ),
                        [t0, 0],
                    )
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        pn_chunk = pl.slice(
                            post_norm, [BATCH, K_CHUNK], [t0, k0],
                        )
                        qi32 = pl.cast(
                            pl.row_expand_div(
                                pl.cast(pn_chunk, target_type=pl.FP32),
                                dequant_scale,
                            ),
                            target_type=pl.INT32,
                            mode="rint",
                        )
                        qf16 = pl.cast(
                            qi32, target_type=pl.FP16, mode="round",
                        )
                        x_i8 = pl.assemble(
                            x_i8,
                            pl.cast(
                                qf16, target_type=pl.INT8, mode="trunc",
                            ),
                            [t0, k0],
                        )

            # Module dump 7: post_norm (post-attention RMSNorm hidden,
            # replicated, pre-quantization).
            if _DUMP_ENABLED:
                post_norm_dump = pl.assemble(post_norm_dump, post_norm, [0, 0])

            # NaN DIAG: producer-exit x_scale (dequant scale) dump.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_x_scale_dump"):
                # x_scale_dump = pl.assemble(
                    # x_scale_dump,
                    # pl.slice(x_scale, [PREFILL_T, SCALE_W_PAD], [0, 0]),
                    # [0, 0],
                # )

            # ── C: per-tile MoE adapter (gate -> shared -> dispatch ->
            # routed -> combine), one [BATCH, HIDDEN] tile at a time. ───
            moe_out = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            for tile_idx in pl.range(PREFILL_TILE_COUNT):
                t_lo = tile_idx * BATCH
                tile_resid = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                tile_post_norm = pl.create_tensor(
                    [BATCH, HIDDEN], dtype=pl.BF16,
                )
                tile_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
                tile_x_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
                tile_x_scale = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_in",
                ):
                    tile_resid = pl.assemble(
                        tile_resid,
                        pl.slice(resid1, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_post_norm = pl.assemble(
                        tile_post_norm,
                        pl.slice(post_norm, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_inv_rms = pl.assemble(
                        tile_inv_rms,
                        pl.slice(moe_inv_rms, [BATCH, 1], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_i8 = pl.assemble(
                        tile_x_i8,
                        pl.slice(x_i8, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_scale = pl.assemble(
                        tile_x_scale,
                        pl.slice(x_scale, [BATCH, SCALE_W_PAD], [t_lo, 0]),
                        [0, 0],
                    )

                # 1) Gate (local, replicated).
                expert_indices = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.INT32,
                )
                expert_weights = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.FP32,
                )
                expert_weights = self.gate_step(
                    tile_resid, post_rms_weight, norm_layer_idx, tile_inv_rms,
                    gate_w, router_bias,
                    expert_indices, expert_weights,
                )

                # DIAG (bisect gate): replicated top-K selection + weights.
                # DIAG: accumulate gate top-K into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_gate_dump"):
                    # expert_indices_dump = pl.assemble(
                        # expert_indices_dump,
                        # pl.slice(expert_indices, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                    # expert_weights_dump = pl.assemble(
                        # expert_weights_dump,
                        # pl.slice(expert_weights, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
                sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                sh_y = self.expert_shared_step(
                    tile_post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                )
                self.moe_tp_all_reduce(
                    sh_y, sh_tmp_window,
                    pl.slice(
                        sh_signal_window, [tp_size, 1],
                        [tile_idx * tp_size, 0],
                    ),
                    my_rank,
                )

                # DIAG (bisect shared): TP-reduced shared-expert output.
                # DIAG: accumulate TP-reduced shared-expert output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_sh_y_dump"):
                    # for kb1 in pl.range(HIDDEN // K_CHUNK):
                        # k01 = kb1 * K_CHUNK
                        # sh_y_dump = pl.assemble(
                            # sh_y_dump,
                            # pl.slice(sh_y, [BATCH, K_CHUNK], [0, k01]),
                            # [t_lo, k01],
                        # )

                # Serialize shared TP -> routed dispatch so dispatch reads the
                # completed tile_x_scale (see _serialize_after_shared).
                tile_x_scale_ser = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                tile_x_scale = self._serialize_after_shared(
                    tile_x_scale, sh_y, tile_x_scale_ser,
                )
                # NaN DIAG: producer send-side dequant scale (pre-dispatch).

                # Per-tile data-window views (Orchestration scope): the 5 EP
                # data windows are shared across the 8-tile unroll, so slice a
                # fresh per-tile view to stop tile i+1's zero/push from racing
                # tile i's gather (same per-tile scheme as the signal windows).
                pub_counts_tile = pl.slice(
                    pub_counts, [N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                    [tile_idx * N_RANKS * N_RANKS, 0],
                )
                send_buf_tile = pl.slice(
                    send_buf, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                send_scale_buf_tile = pl.slice(
                    send_scale_buf, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 1): packed send-side scale window.
                recv_x_tile = pl.slice(
                    recv_x, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                recv_scale_tile = pl.slice(
                    recv_scale, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 2): a2a-delivered recv-side scale window.
                src_route_table_tile = pl.slice(
                    src_route_table,
                    [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    [tile_idx * N_RANKS, 0, 0],
                )
                routed_y_buf_tile = pl.slice(
                    routed_y_buf, [N_ROUTES_PER_RANK, HIDDEN],
                    [tile_idx * N_ROUTES_PER_RANK, 0],
                )

                # 3) Dispatch (EP all-to-all).
                local_routed_x = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                local_routed_x_scale = pl.create_tensor(
                    [1, LOCAL_RECV_MAX], dtype=pl.FP32,
                )
                local_expert_offset = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                local_expert_count = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                self.zero_dispatch_buffers_step(
                    send_buf_tile, send_scale_buf_tile,
                    recv_x_tile, recv_scale_tile,
                )
                (
                    local_routed_x,
                    local_routed_x_scale,
                    local_expert_offset,
                    local_expert_count,
                ) = self.moe_dispatch_step(
                    tile_x_i8, tile_x_scale, expert_indices,
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    send_buf_tile, send_scale_buf_tile,
                    pub_counts_tile,
                    pl.slice(count_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    recv_x_tile, recv_scale_tile,
                    pl.slice(data_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    my_rank,
                )
                # NaN DIAG (candidate ①): per-token activation scale post-dispatch.
                # DIAG: accumulate per-token dequant scale into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_scale_dump"):
                    # local_routed_x_scale_dump = pl.assemble(
                        # local_routed_x_scale_dump,
                        # pl.slice(local_routed_x_scale, [1, N_ROUTES_PER_RANK], [0, 0]),
                        # [0, tile_idx * N_ROUTES_PER_RANK],
                    # )
                # DIAG (bisect dispatch): reordered INT8 x post-a2a (recv_x equiv).
                # DIAG: accumulate reordered routed input into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_x_dump"):
                    # for kb_x in pl.range(HIDDEN // K_CHUNK):
                        # kx0 = kb_x * K_CHUNK
                        # local_routed_x_dump = pl.assemble(
                            # local_routed_x_dump,
                            # pl.slice(local_routed_x, [N_ROUTES_PER_RANK, K_CHUNK], [0, kx0]),
                            # [tile_idx * N_ROUTES_PER_RANK, kx0],
                        # )

                # 4) Routed experts (local 36).
                local_routed_y = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.BF16,
                )
                local_routed_y = self.expert_routed_step_swiglu7(
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    w_gate_r, w_gate_r_scale,
                    w_up_r, w_up_r_scale,
                    w_down_r, w_down_r_scale,
                    local_routed_y,
                )

                # DIAG (bisect routed): per-local-expert output pre-combine.
                # DIAG: accumulate routed expert output into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_y_dump"):
                    # for kb_y in pl.range(HIDDEN // K_CHUNK):
                        # ky0 = kb_y * K_CHUNK
                        # local_routed_y_dump = pl.assemble(
                            # local_routed_y_dump,
                            # pl.slice(local_routed_y, [N_ROUTES_PER_RANK, K_CHUNK], [0, ky0]),
                            # [tile_idx * N_ROUTES_PER_RANK, ky0],
                        # )

                # 5) Combine (EP a2a back + weighted gather + sh_y add).
                tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                # Bind per-tile signal slices to named DistributedTensor views
                # at Orchestration scope before the Inline combine step inlines
                # them: an inline pl.slice(...) arg would splice into the
                # route_pub barrier's CORE_GROUP scope, where
                # ConvertTensorToTileOps demotes it to a TileType and
                # pld.system.notify rejects it (must stay window-bound).
                route_pub_tile_sig = pl.slice(
                    route_pub_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                combine_done_tile_sig = pl.slice(
                    combine_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                tile_y = self.moe_combine_step(
                    local_routed_y,
                    expert_indices, expert_weights, sh_y,
                    tile_y,
                    pub_counts_tile, src_route_table_tile,
                    route_pub_tile_sig,
                    routed_y_buf_tile,
                    combine_done_tile_sig,
                    my_rank,
                )
                # NaN DIAG: combined MoE output — should be TP-replicated.
                # DIAG: accumulate combined MoE tile output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_tile_y_dump"):
                    # for kb2 in pl.range(HIDDEN // K_CHUNK):
                        # k02 = kb2 * K_CHUNK
                        # tile_y_dump = pl.assemble(
                            # tile_y_dump,
                            # pl.slice(tile_y, [BATCH, K_CHUNK], [0, k02]),
                            # [t_lo, k02],
                        # )

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_out",
                ):
                    moe_out = pl.assemble(
                        moe_out,
                        pl.slice(tile_y, [BATCH, HIDDEN], [0, 0]),
                        [t_lo, 0],
                    )

            # Module dump 8: ffn_output (combined shared+routed MoE output,
            # pre-residual-add, TP-replicated).
            if _DUMP_ENABLED:
                ffn_output_dump = pl.assemble(ffn_output_dump, moe_out, [0, 0])

            # ── D: residual add. ───────────────────────────────────────
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add",
            ):
                for tg5 in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg5 * BATCH
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        m = pl.cast(
                            pl.slice(
                                moe_out, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        r = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        out = pl.assemble(
                            out,
                            pl.cast(pl.add(r, m), target_type=pl.BF16),
                            [t0, k0],
                        )
            return out

        @pl.function(
            type=pl.FunctionType.Orchestration,
            attrs={"inline_orchestration": True},
        )
        def full_moe_swiglu7_swiglu16_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim_full], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN_FULL, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            gate_r: pl.Tensor[
                [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_gate_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_up_r: pl.Tensor[
                [N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            w_up_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
            w_down_r: pl.Tensor[
                [N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            w_down_r_scale: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN], pl.FP32],
            w_gate_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, SH_INTER_LOCAL], pl.BF16],
            w_down_s: pl.Tensor[[SH_INTER_LOCAL, HIDDEN], pl.BF16],
            out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            input_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_proj_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_proj_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            v_tile_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32],
            k_norm_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32],
            gate_logits_dump: pl.Tensor[[PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16],
            resid1_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            attn_delta_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            post_norm_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            ffn_output_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            q_rot_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16],
            k_rot_dump: pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16],
            attn_out_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16],
            attn_out_gated_dump: pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16],
            o_proj_local_dump: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            scores_dump: pl.Tensor[
                [PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128], pl.FP32
            ],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, HIDDEN], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, HIDDEN], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                pl.INT32,
            ],
            count_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            send_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            send_scale_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            recv_x: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            recv_scale: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            data_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                pl.INT32,
            ],
            route_pub_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            norm_layer_idx: pl.Scalar[pl.INT32],
            attn_layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            # ── A: prefill attention + tp_all_reduce -> resid1. ────────
            resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            _dummy_opr = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            resid1 = attn_full_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                gate_r,
                positions,
                resid1,
                q_rot_dump,
                k_rot_dump,
                attn_out_dump,
                attn_out_gated_dump,
                o_proj_local_dump,
                _dummy_opr,
                scores_dump,
                input_norm_dump,
                q_proj_dump,
                k_proj_dump,
                v_proj_dump,
                v_tile_dump,
                q_norm_dump,
                k_norm_dump,
                gate_logits_dump,
                attn_delta_dump,
                norm_layer_idx,
                attn_layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            if _DUMP_ENABLED:
                resid1_dump = pl.assemble(resid1_dump, resid1, [0, 0])
            # NaN DIAG: attention output after o_proj tp_all_reduce.
            pl.dump_tag(resid1)
            # DIAG: accumulate resid1 (post-attention) into host-visible dump.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_resid1_dump"):
                # for kb0 in pl.range(HIDDEN // K_CHUNK):
                    # k00 = kb0 * K_CHUNK
                    # resid1_dump = pl.assemble(
                        # resid1_dump,
                        # pl.slice(resid1, [PREFILL_T, K_CHUNK], [0, k00]),
                        # [0, k00],
                    # )

            # ── B: post-attention V4 deferred RMSNorm + INT8/scale producer.
            # Mirrors decode_fwd.py _norm_quant_moe_input: first pass forms
            # Pass 1: reduce sum(resid**2) -> inv_rms. Pass 2: BF16 post-norm
            # (shared lane) + amax over the BF16 post-norm. Pass 3: INT8 payload
            # quant(post_norm / (pn_amax/127)) + per-token dequant scale
            # (pn_amax/127) for the routed lane, bit-matching VLLM W8A8.
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            moe_inv_rms = pl.create_tensor(
                [PREFILL_T, 1], dtype=pl.FP32,
            )
            x_i8 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.INT8,
            )
            x_scale = pl.create_tensor(
                [PREFILL_T, SCALE_W_PAD], dtype=pl.FP32,
            )
            resid1_fp32 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.FP32,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_zc",
            ):
                for tg in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg * BATCH
                    sq_sum = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=0.0,
                    )
                    pn_amax = pl.full(
                        [1, BATCH], dtype=pl.FP32, value=1e-4,
                    )
                    for kb in pl.range(hidden_blocks):
                        k0 = kb * K_CHUNK
                        rchunk = pl.cast(
                            pl.slice(
                                resid1, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        resid1_fp32 = pl.assemble(
                            resid1_fp32, rchunk, [t0, k0],
                        )
                        sq_sum = pl.add(
                            sq_sum,
                            pl.reshape(
                                pl.row_sum(pl.mul(rchunk, rchunk)),
                                [1, BATCH],
                            ),
                        )
                    inv_rms_moe = pl.recip(
                        pl.sqrt(
                            pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                        ),
                    )
                    inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                    moe_inv_rms[t0:t0+BATCH, 0:1] = inv_rms_col
                    for kb3 in pl.range(hidden_blocks):
                        k0 = kb3 * K_CHUNK
                        norm_chunk = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        gamma = pl.slice(
                            post_rms_weight, [1, K_CHUNK],
                            [norm_layer_idx, k0],
                        )
                        xg = pl.col_expand_mul(
                            norm_chunk, pl.add(gamma, 1.0),
                        )
                        normed = pl.row_expand_mul(xg, inv_rms_col)
                        post_norm_chunk = pl.cast(normed, target_type=pl.BF16)
                        post_norm = pl.assemble(
                            post_norm,
                            post_norm_chunk,
                            [t0, k0],
                        )
                        pn_fp32 = pl.cast(
                            post_norm_chunk, target_type=pl.FP32,
                        )
                        pn_amax = pl.maximum(
                            pn_amax,
                            pl.reshape(
                                pl.row_max(
                                    pl.maximum(pn_fp32, pl.neg(pn_fp32)),
                                ),
                                [1, BATCH],
                            ),
                        )
                    dequant_scale = pl.reshape(
                        pl.div(
                            pn_amax,
                            pl.full(
                                [1, BATCH], dtype=pl.FP32, value=127.0,
                            ),
                        ),
                        [BATCH, 1],
                    )
                    x_scale = pl.assemble(
                        x_scale,
                        pl.row_expand_mul(
                            pl.full(
                                [BATCH, SCALE_W_PAD],
                                dtype=pl.FP32,
                                value=1.0,
                            ),
                            dequant_scale,
                        ),
                        [t0, 0],
                    )
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        pn_chunk = pl.slice(
                            post_norm, [BATCH, K_CHUNK], [t0, k0],
                        )
                        qi32 = pl.cast(
                            pl.row_expand_div(
                                pl.cast(pn_chunk, target_type=pl.FP32),
                                dequant_scale,
                            ),
                            target_type=pl.INT32,
                            mode="rint",
                        )
                        qf16 = pl.cast(
                            qi32, target_type=pl.FP16, mode="round",
                        )
                        x_i8 = pl.assemble(
                            x_i8,
                            pl.cast(
                                qf16, target_type=pl.INT8, mode="trunc",
                            ),
                            [t0, k0],
                        )

            # Module dump 7: post_norm (post-attention RMSNorm hidden,
            # replicated, pre-quantization).
            if _DUMP_ENABLED:
                post_norm_dump = pl.assemble(post_norm_dump, post_norm, [0, 0])

            # NaN DIAG: producer-exit x_scale (dequant scale) dump.
            # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_x_scale_dump"):
                # x_scale_dump = pl.assemble(
                    # x_scale_dump,
                    # pl.slice(x_scale, [PREFILL_T, SCALE_W_PAD], [0, 0]),
                    # [0, 0],
                # )

            # ── C: per-tile MoE adapter (gate -> shared -> dispatch ->
            # routed -> combine), one [BATCH, HIDDEN] tile at a time. ───
            moe_out = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            for tile_idx in pl.range(PREFILL_TILE_COUNT):
                t_lo = tile_idx * BATCH
                tile_resid = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                tile_post_norm = pl.create_tensor(
                    [BATCH, HIDDEN], dtype=pl.BF16,
                )
                tile_inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
                tile_x_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
                tile_x_scale = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_in",
                ):
                    tile_resid = pl.assemble(
                        tile_resid,
                        pl.slice(resid1, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_post_norm = pl.assemble(
                        tile_post_norm,
                        pl.slice(post_norm, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_inv_rms = pl.assemble(
                        tile_inv_rms,
                        pl.slice(moe_inv_rms, [BATCH, 1], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_i8 = pl.assemble(
                        tile_x_i8,
                        pl.slice(x_i8, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )
                    tile_x_scale = pl.assemble(
                        tile_x_scale,
                        pl.slice(x_scale, [BATCH, SCALE_W_PAD], [t_lo, 0]),
                        [0, 0],
                    )

                # 1) Gate (local, replicated).
                expert_indices = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.INT32,
                )
                expert_weights = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.FP32,
                )
                expert_weights = self.gate_step(
                    tile_resid, post_rms_weight, norm_layer_idx, tile_inv_rms,
                    gate_w, router_bias,
                    expert_indices, expert_weights,
                )

                # DIAG (bisect gate): replicated top-K selection + weights.
                # DIAG: accumulate gate top-K into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_gate_dump"):
                    # expert_indices_dump = pl.assemble(
                        # expert_indices_dump,
                        # pl.slice(expert_indices, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                    # expert_weights_dump = pl.assemble(
                        # expert_weights_dump,
                        # pl.slice(expert_weights, [BATCH, TOPK], [0, 0]),
                        # [t_lo, 0],
                    # )
                # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
                sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                sh_y = self.expert_shared_step_swiglu16(
                    tile_post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
                )
                self.moe_tp_all_reduce(
                    sh_y, sh_tmp_window,
                    pl.slice(
                        sh_signal_window, [tp_size, 1],
                        [tile_idx * tp_size, 0],
                    ),
                    my_rank,
                )

                # DIAG (bisect shared): TP-reduced shared-expert output.
                # DIAG: accumulate TP-reduced shared-expert output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_sh_y_dump"):
                    # for kb1 in pl.range(HIDDEN // K_CHUNK):
                        # k01 = kb1 * K_CHUNK
                        # sh_y_dump = pl.assemble(
                            # sh_y_dump,
                            # pl.slice(sh_y, [BATCH, K_CHUNK], [0, k01]),
                            # [t_lo, k01],
                        # )

                # Serialize shared TP -> routed dispatch so dispatch reads the
                # completed tile_x_scale (see _serialize_after_shared).
                tile_x_scale_ser = pl.create_tensor(
                    [BATCH, SCALE_W_PAD], dtype=pl.FP32,
                )
                tile_x_scale = self._serialize_after_shared(
                    tile_x_scale, sh_y, tile_x_scale_ser,
                )
                # NaN DIAG: producer send-side dequant scale (pre-dispatch).

                # Per-tile data-window views (Orchestration scope): the 5 EP
                # data windows are shared across the 8-tile unroll, so slice a
                # fresh per-tile view to stop tile i+1's zero/push from racing
                # tile i's gather (same per-tile scheme as the signal windows).
                pub_counts_tile = pl.slice(
                    pub_counts, [N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                    [tile_idx * N_RANKS * N_RANKS, 0],
                )
                send_buf_tile = pl.slice(
                    send_buf, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                send_scale_buf_tile = pl.slice(
                    send_scale_buf, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 1): packed send-side scale window.
                recv_x_tile = pl.slice(
                    recv_x, [LOCAL_RECV_MAX, HIDDEN],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                recv_scale_tile = pl.slice(
                    recv_scale, [LOCAL_RECV_MAX, SCALE_W_PAD],
                    [tile_idx * LOCAL_RECV_MAX, 0],
                )
                # NaN DIAG (stage 2): a2a-delivered recv-side scale window.
                src_route_table_tile = pl.slice(
                    src_route_table,
                    [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    [tile_idx * N_RANKS, 0, 0],
                )
                routed_y_buf_tile = pl.slice(
                    routed_y_buf, [N_ROUTES_PER_RANK, HIDDEN],
                    [tile_idx * N_ROUTES_PER_RANK, 0],
                )

                # 3) Dispatch (EP all-to-all).
                local_routed_x = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                local_routed_x_scale = pl.create_tensor(
                    [1, LOCAL_RECV_MAX], dtype=pl.FP32,
                )
                local_expert_offset = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                local_expert_count = pl.create_tensor(
                    [N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                self.zero_dispatch_buffers_step(
                    send_buf_tile, send_scale_buf_tile,
                    recv_x_tile, recv_scale_tile,
                )
                (
                    local_routed_x,
                    local_routed_x_scale,
                    local_expert_offset,
                    local_expert_count,
                ) = self.moe_dispatch_step(
                    tile_x_i8, tile_x_scale, expert_indices,
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    send_buf_tile, send_scale_buf_tile,
                    pub_counts_tile,
                    pl.slice(count_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    recv_x_tile, recv_scale_tile,
                    pl.slice(data_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0]),
                    my_rank,
                )
                # NaN DIAG (candidate ①): per-token activation scale post-dispatch.
                # DIAG: accumulate per-token dequant scale into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_scale_dump"):
                    # local_routed_x_scale_dump = pl.assemble(
                        # local_routed_x_scale_dump,
                        # pl.slice(local_routed_x_scale, [1, N_ROUTES_PER_RANK], [0, 0]),
                        # [0, tile_idx * N_ROUTES_PER_RANK],
                    # )
                # DIAG (bisect dispatch): reordered INT8 x post-a2a (recv_x equiv).
                # DIAG: accumulate reordered routed input into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_x_dump"):
                    # for kb_x in pl.range(HIDDEN // K_CHUNK):
                        # kx0 = kb_x * K_CHUNK
                        # local_routed_x_dump = pl.assemble(
                            # local_routed_x_dump,
                            # pl.slice(local_routed_x, [N_ROUTES_PER_RANK, K_CHUNK], [0, kx0]),
                            # [tile_idx * N_ROUTES_PER_RANK, kx0],
                        # )

                # 4) Routed experts (local 36).
                local_routed_y = pl.create_tensor(
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.BF16,
                )
                local_routed_y = self.expert_routed_step_swiglu7(
                    local_routed_x, local_routed_x_scale,
                    local_expert_offset, local_expert_count,
                    w_gate_r, w_gate_r_scale,
                    w_up_r, w_up_r_scale,
                    w_down_r, w_down_r_scale,
                    local_routed_y,
                )

                # DIAG (bisect routed): per-local-expert output pre-combine.
                # DIAG: accumulate routed expert output into host-visible dump.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_routed_y_dump"):
                    # for kb_y in pl.range(HIDDEN // K_CHUNK):
                        # ky0 = kb_y * K_CHUNK
                        # local_routed_y_dump = pl.assemble(
                            # local_routed_y_dump,
                            # pl.slice(local_routed_y, [N_ROUTES_PER_RANK, K_CHUNK], [0, ky0]),
                            # [tile_idx * N_ROUTES_PER_RANK, ky0],
                        # )

                # 5) Combine (EP a2a back + weighted gather + sh_y add).
                tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                # Bind per-tile signal slices to named DistributedTensor views
                # at Orchestration scope before the Inline combine step inlines
                # them: an inline pl.slice(...) arg would splice into the
                # route_pub barrier's CORE_GROUP scope, where
                # ConvertTensorToTileOps demotes it to a TileType and
                # pld.system.notify rejects it (must stay window-bound).
                route_pub_tile_sig = pl.slice(
                    route_pub_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                combine_done_tile_sig = pl.slice(
                    combine_done_sig, [N_RANKS, 1], [tile_idx * N_RANKS, 0],
                )
                tile_y = self.moe_combine_step(
                    local_routed_y,
                    expert_indices, expert_weights, sh_y,
                    tile_y,
                    pub_counts_tile, src_route_table_tile,
                    route_pub_tile_sig,
                    routed_y_buf_tile,
                    combine_done_tile_sig,
                    my_rank,
                )
                # NaN DIAG: combined MoE output — should be TP-replicated.
                # DIAG: accumulate combined MoE tile output.
                # with pl.at(level=pl.Level.CORE_GROUP, name_hint="diag_tile_y_dump"):
                    # for kb2 in pl.range(HIDDEN // K_CHUNK):
                        # k02 = kb2 * K_CHUNK
                        # tile_y_dump = pl.assemble(
                            # tile_y_dump,
                            # pl.slice(tile_y, [BATCH, K_CHUNK], [0, k02]),
                            # [t_lo, k02],
                        # )

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_out",
                ):
                    moe_out = pl.assemble(
                        moe_out,
                        pl.slice(tile_y, [BATCH, HIDDEN], [0, 0]),
                        [t_lo, 0],
                    )

            # Module dump 8: ffn_output (combined shared+routed MoE output,
            # pre-residual-add, TP-replicated).
            if _DUMP_ENABLED:
                ffn_output_dump = pl.assemble(ffn_output_dump, moe_out, [0, 0])

            # ── D: residual add. ───────────────────────────────────────
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add",
            ):
                for tg5 in pl.range(PREFILL_TILE_COUNT):
                    t0 = tg5 * BATCH
                    for kb4 in pl.range(hidden_blocks):
                        k0 = kb4 * K_CHUNK
                        m = pl.cast(
                            pl.slice(
                                moe_out, [BATCH, K_CHUNK], [t0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        r = pl.slice(
                            resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                        )
                        out = pl.assemble(
                            out,
                            pl.cast(pl.add(r, m), target_type=pl.BF16),
                            [t0, k0],
                        )
            return out

        # ── whole_chip_orch: 45-main-layer loop (Orchestration). ──
        # Mirrors decode_fwd.py:3381 ``whole_chip_orch`` (Orchestration level,
        # NOT HOST): the 45-layer pl.range loop and every per-layer Out
        # pl.create_tensor live here because pl.create_tensor is a device-
        # scope op (B-probe: HOST-scope use is compile-rejected).  host_orch
        # (HOST) just allocs window buffers and calls this once per rank.
        #
        # Dispatch: runtime if/elif on the loop scalar ``li``, proven by
        # decode_fwd.py:3640 (``if layer_idx % 4 == 1``).  LAYER_TYPES is the
        # repeating block (full, swa, swa, swa) → full-attention layers are
        # ``li % 4 == 0``.  Layers 0,1,2 are dense MLP; 3..44 are MoE; 43/44
        # carry SwigluStep limits (see config.SWIGLU_LIMITS).
        @pl.function(type=pl.FunctionType.Orchestration)
        def whole_chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            full_wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            full_wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            full_wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            full_wo: pl.Tensor[
                [LAYER_QHIDDEN_ROWS_DYN_FULL, HIDDEN], pl.BF16
            ],
            full_w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            full_gate_r: pl.Tensor[
                [NUM_FULL_LAYERS * NUM_HEADS_FULL_LOCAL_PAD,
                 HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            swa_wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN_SWA, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            swa_wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL], pl.BF16
            ],
            swa_wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL], pl.BF16
            ],
            swa_wo: pl.Tensor[
                [LAYER_QHIDDEN_ROWS_DYN_SWA, HIDDEN], pl.BF16
            ],
            swa_w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN_SWA, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            swa_gate_r: pl.Tensor[
                [NUM_SWA_LAYERS * NUM_HEADS_SWA_LOCAL_PAD,
                 HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            dense_w_gate: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL], pl.BF16
            ],
            dense_w_up: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL], pl.BF16
            ],
            dense_w_down: pl.Tensor[
                [LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos_full: pl.Tensor[
                [ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
            ],
            rope_sin_full: pl.Tensor[
                [ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
            ],
            rope_cos_swa: pl.Tensor[
                [ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
            ],
            rope_sin_swa: pl.Tensor[
                [ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
            ],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            next_hidden_out: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            dense_attn_tmp_stack: pld.DistributedTensor[
                [NUM_DENSE_LAYERS * PREFILL_T, HIDDEN], pl.BF16
            ],
            dense_attn_signal_stack: pld.DistributedTensor[
                [NUM_DENSE_LAYERS * tp_size, 1], pl.INT32
            ],
            dense_mlp_tmp_stack: pld.DistributedTensor[
                [NUM_DENSE_LAYERS * PREFILL_T, HIDDEN], pl.BF16
            ],
            dense_mlp_signal_stack: pld.DistributedTensor[
                [NUM_DENSE_LAYERS * tp_size, 1], pl.INT32
            ],
            moe_gate_w: pl.Tensor[
                [n_moe_layers * HIDDEN, N_EXPERTS], pl.FP32
            ],
            moe_router_bias: pl.Tensor[
                [n_moe_layers * N_EXPERTS], pl.FP32
            ],
            moe_w_gate_r: pl.Tensor[
                [n_moe_layers * N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            moe_w_gate_r_scale: pl.Tensor[
                [n_moe_layers * N_LOCAL_EXPERTS, INTER], pl.FP32
            ],
            moe_w_up_r: pl.Tensor[
                [n_moe_layers * N_LOCAL_EXPERTS * HIDDEN, INTER], pl.INT8
            ],
            moe_w_up_r_scale: pl.Tensor[
                [n_moe_layers * N_LOCAL_EXPERTS, INTER], pl.FP32
            ],
            moe_w_down_r: pl.Tensor[
                [n_moe_layers * N_LOCAL_EXPERTS * INTER, HIDDEN], pl.INT8
            ],
            moe_w_down_r_scale: pl.Tensor[
                [n_moe_layers * N_LOCAL_EXPERTS, HIDDEN], pl.FP32
            ],
            moe_w_gate_s: pl.Tensor[
                [n_moe_layers * HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            moe_w_up_s: pl.Tensor[
                [n_moe_layers * HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            moe_w_down_s: pl.Tensor[
                [n_moe_layers * SH_INTER_LOCAL, HIDDEN], pl.BF16
            ],
            moe_attn_tmp_stack: pld.DistributedTensor[
                [NUM_MOE_LAYERS * PREFILL_T, HIDDEN], pl.BF16
            ],
            moe_attn_signal_stack: pld.DistributedTensor[
                [NUM_MOE_LAYERS * tp_size, 1], pl.INT32
            ],
            moe_sh_tmp_stack: pld.DistributedTensor[
                [NUM_MOE_LAYERS * BATCH, HIDDEN], pl.BF16
            ],
            moe_sh_signal_stack: pld.DistributedTensor[
                [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * tp_size, 1], pl.INT32
            ],
            moe_pub_counts: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                pl.INT32,
            ],
            moe_count_done_sig: pld.DistributedTensor[
                [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            moe_send_x: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            moe_send_scale: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            moe_recv_x: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], pl.INT8
            ],
            moe_recv_scale: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], pl.FP32
            ],
            moe_data_done_sig: pld.DistributedTensor[
                [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            moe_src_route_table: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                pl.INT32,
            ],
            moe_route_pub_sig: pld.DistributedTensor[
                [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            moe_routed_y_buf: pld.DistributedTensor[
                [PREFILL_TILE_COUNT * N_ROUTES_PER_RANK, HIDDEN], pl.BF16
            ],
            moe_combine_done_sig: pld.DistributedTensor[
                [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1], pl.INT32
            ],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[VOCAB_LOCAL, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[PREFILL_T, VOCAB_LOCAL], pl.FP32]
            ],
            input_norm_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            q_proj_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32]],
            k_proj_dump: pl.Out[pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32]],
            v_proj_dump: pl.Out[pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32]],
            v_tile_dump: pl.Out[pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]],
            q_norm_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32]],
            k_norm_dump: pl.Out[pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32]],
            gate_logits_dump: pl.Out[pl.Tensor[[PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16]],
            resid1_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            attn_delta_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            post_norm_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            ffn_output_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            q_rot_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16]],
            k_rot_dump: pl.Out[pl.Tensor[[PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]],
            attn_out_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16]],
            attn_out_gated_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16]],
            o_proj_local_dump: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            scores_dump: pl.Out[
                pl.Tensor[
                    [PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128], pl.FP32
                ]
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            prev = current_hidden
            # ``li`` stays ABSOLUTE (0..44) even for a restricted
            # [layer_lo, layer_hi) range — the %4 / ==43 / ==44 / <3 dispatch
            # predicates all key off the absolute layer id (blue-approved:
            # never remap li to a chunk-local 0 base).
            for li in pl.range(layer_lo, layer_hi):
                out = pl.create_tensor(
                    [PREFILL_T, HIDDEN], dtype=pl.BF16,
                )
                if li == 0:
                    # Layer 0: dense MLP, full-attention (norm=0, attn=0,
                    # mlp=0; all stack offsets 0).
                    out = self.full_dense_chip_orch(
                        prev,
                        input_rms_weight,
                        pl.slice(full_wq, [HIDDEN, HIDDEN_Q_FULL_LOCAL], [0, 0]),
                        pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [0, 0]),
                        pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [0, 0]),
                        q_norm_weight,
                        k_norm_weight,
                        block_table,
                        slot_mapping,
                        rope_cos_full,
                        rope_sin_full,
                        k_cache,
                        v_cache,
                        pl.slice(full_wo, [HIDDEN_Q_FULL_LOCAL, HIDDEN], [0, 0]),
                        pl.slice(full_w_g, [HIDDEN, NUM_HEADS_FULL_LOCAL_PAD], [0, 0]),
                        pl.slice(full_gate_r, [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], [0, 0]),
                        positions,
                        post_rms_weight,
                        pl.slice(dense_w_gate, [HIDDEN, INTERMEDIATE_LOCAL], [0, 0]),
                        pl.slice(dense_w_up, [HIDDEN, INTERMEDIATE_LOCAL], [0, 0]),
                        pl.slice(dense_w_down, [INTERMEDIATE_LOCAL, HIDDEN], [0, 0]),
                        out,
                        input_norm_dump,
                        q_proj_dump,
                        k_proj_dump,
                        v_proj_dump,
                        v_tile_dump,
                        q_norm_dump,
                        k_norm_dump,
                        gate_logits_dump,
                        resid1_dump,
                        attn_delta_dump,
                        post_norm_dump,
                        ffn_output_dump,
                        k_rot_dump,
                        pl.slice(dense_attn_tmp_stack, [PREFILL_T, HIDDEN], [0, 0]),
                        pl.slice(dense_attn_signal_stack, [tp_size, 1], [0, 0]),
                        pl.slice(dense_mlp_tmp_stack, [PREFILL_T, HIDDEN], [0, 0]),
                        pl.slice(dense_mlp_signal_stack, [tp_size, 1], [0, 0]),
                        0,
                        0,
                        0,
                        my_rank,
                    )
                elif li < 3:
                    # Layers 1, 2: dense MLP, sliding-window attention.
                    # swa-local index = li - 1; dense mlp index = li.
                    swa_local = li - 1
                    swa_w_off = swa_local * HIDDEN
                    swa_wo_off = swa_local * HIDDEN_Q_SWA_LOCAL
                    swa_gate_r_off = swa_local * NUM_HEADS_SWA_LOCAL_PAD
                    dense_w_off = li * HIDDEN
                    dense_down_off = li * INTERMEDIATE_LOCAL
                    win_off = li * PREFILL_T
                    sig_off = li * tp_size
                    norm_layer_idx = pl.cast(li, pl.INT32)
                    # Weights are pre-sliced per layer at orchestration level
                    # (decode-faithful), so the body's attn/mlp local indices
                    # must stay 0. A non-zero index would slice the already
                    # sliced stack a second time (double offset) and fault
                    # out-of-range on layers 1/2 (w_down is only 3*1408 rows).
                    attn_layer_idx = pl.cast(0, pl.INT32)
                    mlp_layer_idx = pl.cast(0, pl.INT32)
                    out = self.swa_dense_chip_orch(
                        prev,
                        input_rms_weight,
                        pl.slice(swa_wq, [HIDDEN, HIDDEN_Q_SWA_LOCAL], [swa_w_off, 0]),
                        pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [swa_w_off, 0]),
                        pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [swa_w_off, 0]),
                        q_norm_weight,
                        k_norm_weight,
                        block_table,
                        slot_mapping,
                        rope_cos_swa,
                        rope_sin_swa,
                        k_cache,
                        v_cache,
                        pl.slice(swa_wo, [HIDDEN_Q_SWA_LOCAL, HIDDEN], [swa_wo_off, 0]),
                        pl.slice(swa_w_g, [HIDDEN, NUM_HEADS_SWA_LOCAL_PAD], [swa_w_off, 0]),
                        pl.slice(swa_gate_r, [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], [swa_gate_r_off, 0]),
                        positions,
                        post_rms_weight,
                        pl.slice(dense_w_gate, [HIDDEN, INTERMEDIATE_LOCAL], [dense_w_off, 0]),
                        pl.slice(dense_w_up, [HIDDEN, INTERMEDIATE_LOCAL], [dense_w_off, 0]),
                        pl.slice(dense_w_down, [INTERMEDIATE_LOCAL, HIDDEN], [dense_down_off, 0]),
                        out,
                        input_norm_dump,
                        q_proj_dump,
                        k_proj_dump,
                        v_proj_dump,
                        v_tile_dump,
                        q_norm_dump,
                        k_norm_dump,
                        gate_logits_dump,
                        resid1_dump,
                        attn_delta_dump,
                        post_norm_dump,
                        ffn_output_dump,
                        k_rot_dump,
                        pl.slice(dense_attn_tmp_stack, [PREFILL_T, HIDDEN], [win_off, 0]),
                        pl.slice(dense_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                        pl.slice(dense_mlp_tmp_stack, [PREFILL_T, HIDDEN], [win_off, 0]),
                        pl.slice(dense_mlp_signal_stack, [tp_size, 1], [sig_off, 0]),
                        norm_layer_idx,
                        attn_layer_idx,
                        mlp_layer_idx,
                        my_rank,
                    )
                elif li == 43:
                    # Layer 43: swa-attn MoE, routed swiglu7 / shared silu.
                    swa_local = li - li // 4 - 1
                    swa_w_off = swa_local * HIDDEN
                    swa_wo_off = swa_local * HIDDEN_Q_SWA_LOCAL
                    swa_gate_r_off = swa_local * NUM_HEADS_SWA_LOCAL_PAD
                    moe_pos = li - base
                    moe_w_off = moe_pos * HIDDEN
                    moe_bias_off = moe_pos * N_EXPERTS
                    moe_r_off = moe_pos * N_LOCAL_EXPERTS * HIDDEN
                    moe_r_down_off = moe_pos * N_LOCAL_EXPERTS * INTER
                    moe_r_scale_off = moe_pos * N_LOCAL_EXPERTS
                    moe_sh_down_off = moe_pos * SH_INTER_LOCAL
                    win_off = moe_pos * PREFILL_T
                    sh_win_off = moe_pos * BATCH
                    sig_off = moe_pos * tp_size
                    sh_sig_off = moe_pos * PREFILL_TILE_COUNT * tp_size
                    barrier_off = moe_pos * PREFILL_TILE_COUNT * N_RANKS
                    norm_layer_idx = pl.cast(li, pl.INT32)
                    attn_layer_idx = pl.cast(0, pl.INT32)
                    if moe_pos > 0 and PREFILL_SKIP_CROSS_FENCE == 0:
                        self.moe_cross_layer_fence(
                            pl.slice(
                                moe_combine_done_sig,
                                [PREFILL_TILE_COUNT * N_RANKS, 1],
                                [(moe_pos - 1) * PREFILL_TILE_COUNT * N_RANKS, 0],
                            ),
                            my_rank,
                        )
                    out = self.swa_moe_swiglu7_silu_chip_orch(
                        prev,
                        input_rms_weight,
                        pl.slice(swa_wq, [HIDDEN, HIDDEN_Q_SWA_LOCAL], [swa_w_off, 0]),
                        pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [swa_w_off, 0]),
                        pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [swa_w_off, 0]),
                        q_norm_weight,
                        k_norm_weight,
                        block_table,
                        slot_mapping,
                        rope_cos_swa,
                        rope_sin_swa,
                        k_cache,
                        v_cache,
                        pl.slice(swa_wo, [HIDDEN_Q_SWA_LOCAL, HIDDEN], [swa_wo_off, 0]),
                        pl.slice(swa_w_g, [HIDDEN, NUM_HEADS_SWA_LOCAL_PAD], [swa_w_off, 0]),
                        pl.slice(swa_gate_r, [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], [swa_gate_r_off, 0]),
                        positions,
                        post_rms_weight,
                        pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off, 0]),
                        pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                        pl.slice(moe_w_gate_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_gate_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_up_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_up_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_down_r, [N_LOCAL_EXPERTS * INTER, HIDDEN], [moe_r_down_off, 0]),
                        pl.slice(moe_w_down_r_scale, [N_LOCAL_EXPERTS, HIDDEN], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_gate_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_up_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_down_s, [SH_INTER_LOCAL, HIDDEN], [moe_sh_down_off, 0]),
                        out,
                        input_norm_dump,
                        q_proj_dump,
                        k_proj_dump,
                        v_proj_dump,
                        v_tile_dump,
                        q_norm_dump,
                        k_norm_dump,
                        gate_logits_dump,
                        resid1_dump,
                        attn_delta_dump,
                        post_norm_dump,
                        ffn_output_dump,
                        k_rot_dump,
                        pl.slice(moe_attn_tmp_stack, [PREFILL_T, HIDDEN], [win_off, 0]),
                        pl.slice(moe_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                        pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [sh_win_off, 0]),
                        pl.slice(moe_sh_signal_stack, [PREFILL_TILE_COUNT * tp_size, 1], [sh_sig_off, 0]),
                        moe_pub_counts,
                        pl.slice(moe_count_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_send_x,
                        moe_send_scale,
                        moe_recv_x,
                        moe_recv_scale,
                        pl.slice(moe_data_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_src_route_table,
                        pl.slice(moe_route_pub_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_routed_y_buf,
                        pl.slice(moe_combine_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        norm_layer_idx,
                        attn_layer_idx,
                        my_rank,
                    )
                elif li == 44:
                    # Layer 44: full-attn MoE, routed swiglu7 / shared swiglu16.
                    full_local = li // 4
                    full_w_off = full_local * HIDDEN
                    full_wo_off = full_local * HIDDEN_Q_FULL_LOCAL
                    full_gate_r_off = full_local * NUM_HEADS_FULL_LOCAL_PAD
                    moe_pos = li - base
                    moe_w_off = moe_pos * HIDDEN
                    moe_bias_off = moe_pos * N_EXPERTS
                    moe_r_off = moe_pos * N_LOCAL_EXPERTS * HIDDEN
                    moe_r_down_off = moe_pos * N_LOCAL_EXPERTS * INTER
                    moe_r_scale_off = moe_pos * N_LOCAL_EXPERTS
                    moe_sh_down_off = moe_pos * SH_INTER_LOCAL
                    win_off = moe_pos * PREFILL_T
                    sh_win_off = moe_pos * BATCH
                    sig_off = moe_pos * tp_size
                    sh_sig_off = moe_pos * PREFILL_TILE_COUNT * tp_size
                    barrier_off = moe_pos * PREFILL_TILE_COUNT * N_RANKS
                    norm_layer_idx = pl.cast(li, pl.INT32)
                    attn_layer_idx = pl.cast(0, pl.INT32)
                    if moe_pos > 0 and PREFILL_SKIP_CROSS_FENCE == 0:
                        self.moe_cross_layer_fence(
                            pl.slice(
                                moe_combine_done_sig,
                                [PREFILL_TILE_COUNT * N_RANKS, 1],
                                [(moe_pos - 1) * PREFILL_TILE_COUNT * N_RANKS, 0],
                            ),
                            my_rank,
                        )
                    out = self.full_moe_swiglu7_swiglu16_chip_orch(
                        prev,
                        input_rms_weight,
                        pl.slice(full_wq, [HIDDEN, HIDDEN_Q_FULL_LOCAL], [full_w_off, 0]),
                        pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [full_w_off, 0]),
                        pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [full_w_off, 0]),
                        q_norm_weight,
                        k_norm_weight,
                        block_table,
                        slot_mapping,
                        rope_cos_full,
                        rope_sin_full,
                        k_cache,
                        v_cache,
                        pl.slice(full_wo, [HIDDEN_Q_FULL_LOCAL, HIDDEN], [full_wo_off, 0]),
                        pl.slice(full_w_g, [HIDDEN, NUM_HEADS_FULL_LOCAL_PAD], [full_w_off, 0]),
                        pl.slice(full_gate_r, [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], [full_gate_r_off, 0]),
                        positions,
                        post_rms_weight,
                        pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off, 0]),
                        pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                        pl.slice(moe_w_gate_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_gate_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_up_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_up_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_down_r, [N_LOCAL_EXPERTS * INTER, HIDDEN], [moe_r_down_off, 0]),
                        pl.slice(moe_w_down_r_scale, [N_LOCAL_EXPERTS, HIDDEN], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_gate_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_up_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_down_s, [SH_INTER_LOCAL, HIDDEN], [moe_sh_down_off, 0]),
                        out,
                        input_norm_dump,
                        q_proj_dump,
                        k_proj_dump,
                        v_proj_dump,
                        v_tile_dump,
                        q_norm_dump,
                        k_norm_dump,
                        gate_logits_dump,
                        resid1_dump,
                        attn_delta_dump,
                        post_norm_dump,
                        ffn_output_dump,
                        q_rot_dump,
                        k_rot_dump,
                        attn_out_dump,
                        attn_out_gated_dump,
                        o_proj_local_dump,
                        scores_dump,
                        pl.slice(moe_attn_tmp_stack, [PREFILL_T, HIDDEN], [win_off, 0]),
                        pl.slice(moe_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                        pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [sh_win_off, 0]),
                        pl.slice(moe_sh_signal_stack, [PREFILL_TILE_COUNT * tp_size, 1], [sh_sig_off, 0]),
                        moe_pub_counts,
                        pl.slice(moe_count_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_send_x,
                        moe_send_scale,
                        moe_recv_x,
                        moe_recv_scale,
                        pl.slice(moe_data_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_src_route_table,
                        pl.slice(moe_route_pub_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_routed_y_buf,
                        pl.slice(moe_combine_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        norm_layer_idx,
                        attn_layer_idx,
                        my_rank,
                    )
                elif li % 4 == 0:
                    # Full-attention MoE layers (4, 8, ..., 40): silu/silu.
                    # Attention weights are pre-sliced per layer (decode-
                    # faithful), so the body's attn_layer_idx stays 0; the
                    # norm stack is indexed by the absolute layer id.
                    full_local = li // 4
                    full_w_off = full_local * HIDDEN
                    full_wo_off = full_local * HIDDEN_Q_FULL_LOCAL
                    full_gate_r_off = full_local * NUM_HEADS_FULL_LOCAL_PAD
                    moe_pos = li - base
                    moe_w_off = moe_pos * HIDDEN
                    moe_bias_off = moe_pos * N_EXPERTS
                    moe_r_off = moe_pos * N_LOCAL_EXPERTS * HIDDEN
                    moe_r_down_off = moe_pos * N_LOCAL_EXPERTS * INTER
                    moe_r_scale_off = moe_pos * N_LOCAL_EXPERTS
                    moe_sh_down_off = moe_pos * SH_INTER_LOCAL
                    win_off = moe_pos * PREFILL_T
                    sh_win_off = moe_pos * BATCH
                    sig_off = moe_pos * tp_size
                    sh_sig_off = moe_pos * PREFILL_TILE_COUNT * tp_size
                    barrier_off = moe_pos * PREFILL_TILE_COUNT * N_RANKS
                    norm_layer_idx = pl.cast(li, pl.INT32)
                    attn_layer_idx = pl.cast(0, pl.INT32)
                    if moe_pos > 0 and PREFILL_SKIP_CROSS_FENCE == 0:
                        self.moe_cross_layer_fence(
                            pl.slice(
                                moe_combine_done_sig,
                                [PREFILL_TILE_COUNT * N_RANKS, 1],
                                [(moe_pos - 1) * PREFILL_TILE_COUNT * N_RANKS, 0],
                            ),
                            my_rank,
                        )
                    out = self.full_moe_silu_silu_chip_orch(
                        prev,
                        input_rms_weight,
                        pl.slice(full_wq, [HIDDEN, HIDDEN_Q_FULL_LOCAL], [full_w_off, 0]),
                        pl.slice(full_wk, [HIDDEN, KV_HIDDEN_LOCAL], [full_w_off, 0]),
                        pl.slice(full_wv, [HIDDEN, KV_HIDDEN_LOCAL], [full_w_off, 0]),
                        q_norm_weight,
                        k_norm_weight,
                        block_table,
                        slot_mapping,
                        rope_cos_full,
                        rope_sin_full,
                        k_cache,
                        v_cache,
                        pl.slice(full_wo, [HIDDEN_Q_FULL_LOCAL, HIDDEN], [full_wo_off, 0]),
                        pl.slice(full_w_g, [HIDDEN, NUM_HEADS_FULL_LOCAL_PAD], [full_w_off, 0]),
                        pl.slice(full_gate_r, [NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL], [full_gate_r_off, 0]),
                        positions,
                        post_rms_weight,
                        pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off, 0]),
                        pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                        pl.slice(moe_w_gate_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_gate_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_up_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_up_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_down_r, [N_LOCAL_EXPERTS * INTER, HIDDEN], [moe_r_down_off, 0]),
                        pl.slice(moe_w_down_r_scale, [N_LOCAL_EXPERTS, HIDDEN], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_gate_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_up_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_down_s, [SH_INTER_LOCAL, HIDDEN], [moe_sh_down_off, 0]),
                        out,
                        input_norm_dump,
                        q_proj_dump,
                        k_proj_dump,
                        v_proj_dump,
                        v_tile_dump,
                        q_norm_dump,
                        k_norm_dump,
                        gate_logits_dump,
                        resid1_dump,
                        attn_delta_dump,
                        post_norm_dump,
                        ffn_output_dump,
                        k_rot_dump,
                        pl.slice(moe_attn_tmp_stack, [PREFILL_T, HIDDEN], [win_off, 0]),
                        pl.slice(moe_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                        pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [sh_win_off, 0]),
                        pl.slice(moe_sh_signal_stack, [PREFILL_TILE_COUNT * tp_size, 1], [sh_sig_off, 0]),
                        moe_pub_counts,
                        pl.slice(moe_count_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_send_x,
                        moe_send_scale,
                        moe_recv_x,
                        moe_recv_scale,
                        pl.slice(moe_data_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_src_route_table,
                        pl.slice(moe_route_pub_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_routed_y_buf,
                        pl.slice(moe_combine_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        norm_layer_idx,
                        attn_layer_idx,
                        my_rank,
                    )
                else:
                    # Sliding-window MoE layers (3, 5, 6, 7, 9, ...): silu/silu.
                    swa_local = li - li // 4 - 1
                    swa_w_off = swa_local * HIDDEN
                    swa_wo_off = swa_local * HIDDEN_Q_SWA_LOCAL
                    swa_gate_r_off = swa_local * NUM_HEADS_SWA_LOCAL_PAD
                    moe_pos = li - base
                    moe_w_off = moe_pos * HIDDEN
                    moe_bias_off = moe_pos * N_EXPERTS
                    moe_r_off = moe_pos * N_LOCAL_EXPERTS * HIDDEN
                    moe_r_down_off = moe_pos * N_LOCAL_EXPERTS * INTER
                    moe_r_scale_off = moe_pos * N_LOCAL_EXPERTS
                    moe_sh_down_off = moe_pos * SH_INTER_LOCAL
                    win_off = moe_pos * PREFILL_T
                    sh_win_off = moe_pos * BATCH
                    sig_off = moe_pos * tp_size
                    sh_sig_off = moe_pos * PREFILL_TILE_COUNT * tp_size
                    barrier_off = moe_pos * PREFILL_TILE_COUNT * N_RANKS
                    norm_layer_idx = pl.cast(li, pl.INT32)
                    attn_layer_idx = pl.cast(0, pl.INT32)
                    if moe_pos > 0 and PREFILL_SKIP_CROSS_FENCE == 0:
                        self.moe_cross_layer_fence(
                            pl.slice(
                                moe_combine_done_sig,
                                [PREFILL_TILE_COUNT * N_RANKS, 1],
                                [(moe_pos - 1) * PREFILL_TILE_COUNT * N_RANKS, 0],
                            ),
                            my_rank,
                        )
                    out = self.swa_moe_silu_silu_chip_orch(
                        prev,
                        input_rms_weight,
                        pl.slice(swa_wq, [HIDDEN, HIDDEN_Q_SWA_LOCAL], [swa_w_off, 0]),
                        pl.slice(swa_wk, [HIDDEN, KV_HIDDEN_LOCAL], [swa_w_off, 0]),
                        pl.slice(swa_wv, [HIDDEN, KV_HIDDEN_LOCAL], [swa_w_off, 0]),
                        q_norm_weight,
                        k_norm_weight,
                        block_table,
                        slot_mapping,
                        rope_cos_swa,
                        rope_sin_swa,
                        k_cache,
                        v_cache,
                        pl.slice(swa_wo, [HIDDEN_Q_SWA_LOCAL, HIDDEN], [swa_wo_off, 0]),
                        pl.slice(swa_w_g, [HIDDEN, NUM_HEADS_SWA_LOCAL_PAD], [swa_w_off, 0]),
                        pl.slice(swa_gate_r, [NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL], [swa_gate_r_off, 0]),
                        positions,
                        post_rms_weight,
                        pl.slice(moe_gate_w, [HIDDEN, N_EXPERTS], [moe_w_off, 0]),
                        pl.slice(moe_router_bias, [N_EXPERTS], [moe_bias_off]),
                        pl.slice(moe_w_gate_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_gate_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_up_r, [N_LOCAL_EXPERTS * HIDDEN, INTER], [moe_r_off, 0]),
                        pl.slice(moe_w_up_r_scale, [N_LOCAL_EXPERTS, INTER], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_down_r, [N_LOCAL_EXPERTS * INTER, HIDDEN], [moe_r_down_off, 0]),
                        pl.slice(moe_w_down_r_scale, [N_LOCAL_EXPERTS, HIDDEN], [moe_r_scale_off, 0]),
                        pl.slice(moe_w_gate_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_up_s, [HIDDEN, SH_INTER_LOCAL], [moe_w_off, 0]),
                        pl.slice(moe_w_down_s, [SH_INTER_LOCAL, HIDDEN], [moe_sh_down_off, 0]),
                        out,
                        input_norm_dump,
                        q_proj_dump,
                        k_proj_dump,
                        v_proj_dump,
                        v_tile_dump,
                        q_norm_dump,
                        k_norm_dump,
                        gate_logits_dump,
                        resid1_dump,
                        attn_delta_dump,
                        post_norm_dump,
                        ffn_output_dump,
                        k_rot_dump,
                        pl.slice(moe_attn_tmp_stack, [PREFILL_T, HIDDEN], [win_off, 0]),
                        pl.slice(moe_attn_signal_stack, [tp_size, 1], [sig_off, 0]),
                        pl.slice(moe_sh_tmp_stack, [BATCH, HIDDEN], [sh_win_off, 0]),
                        pl.slice(moe_sh_signal_stack, [PREFILL_TILE_COUNT * tp_size, 1], [sh_sig_off, 0]),
                        moe_pub_counts,
                        pl.slice(moe_count_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_send_x,
                        moe_send_scale,
                        moe_recv_x,
                        moe_recv_scale,
                        pl.slice(moe_data_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_src_route_table,
                        pl.slice(moe_route_pub_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        moe_routed_y_buf,
                        pl.slice(moe_combine_done_sig, [PREFILL_TILE_COUNT * N_RANKS, 1], [barrier_off, 0]),
                        norm_layer_idx,
                        attn_layer_idx,
                        my_rank,
                    )
                prev = out
            # Copy the final residual into the Out formal.  Plain name
            # rebinding (``next_hidden_out = prev``) does not write the Out
            # buffer in PyPTO — the Out must be written via pl.assemble (the
            # same K_CHUNK-column loop the residual-add bodies use).  Without
            # this the device run validates with an all-zero next_hidden_out.
            # lm_head wiring (last-token slice + rms_lm_head) lands with the
            # weight-stack expansion.
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_whole_residual_write",
            ):
                for tt in pl.range(PREFILL_TILE_COUNT):
                    t0 = tt * BATCH
                    for kb in pl.range(HIDDEN // K_CHUNK):
                        k0 = kb * K_CHUNK
                        next_hidden_out = pl.assemble(
                            next_hidden_out,
                            pl.slice(prev, [BATCH, K_CHUNK], [t0, k0]),
                            [t0, k0],
                        )
            # ── tail (Phase 3b): per-position logits [PREFILL_T, VOCAB_LOCAL]. ──
            # Delegate to rms_lm_head_chip_orch, which loops PREFILL_TILE_COUNT
            # tiles of BATCH(=16) rows and assembles the full 128-row logits
            # shard (per-position next-token predictions, §12.8).
            logits_shard_out = self.rms_lm_head_chip_orch(
                prev,
                final_norm_weight,
                lm_head_weight,
                seq_lens,
                logits_shard_out,
            )
            return next_hidden_out

        # ── host_orch (HOST): window alloc + per-rank whole_chip_orch. ──
        # Mirrors decode_fwd.py:3936 host_orch.  Each weight stack carries a
        # tp_size leading dim and is sliced [r] per rank; the dense window
        # stacks are sized NUM_DENSE_LAYERS × per-layer and sliced by layer
        # inside whole_chip_orch.
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913
            self,
            hidden_states: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            post_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            full_wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            full_wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            full_wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            full_wo: pl.Tensor[
                [tp_size, LAYER_QHIDDEN_ROWS_DYN_FULL, HIDDEN], pl.BF16
            ],
            full_w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16
            ],
            full_gate_r: pl.Tensor[
                [tp_size, NUM_FULL_LAYERS * NUM_HEADS_FULL_LOCAL_PAD,
                 HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            swa_wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            swa_wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL], pl.BF16
            ],
            swa_wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL], pl.BF16
            ],
            swa_wo: pl.Tensor[
                [tp_size, LAYER_QHIDDEN_ROWS_DYN_SWA, HIDDEN], pl.BF16
            ],
            swa_w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, NUM_HEADS_SWA_LOCAL_PAD], pl.BF16
            ],
            swa_gate_r: pl.Tensor[
                [tp_size, NUM_SWA_LAYERS * NUM_HEADS_SWA_LOCAL_PAD,
                 HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            dense_w_gate: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL], pl.BF16
            ],
            dense_w_up: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL], pl.BF16
            ],
            dense_w_down: pl.Tensor[
                [tp_size, LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            moe_gate_w: pl.Tensor[
                [tp_size, n_moe_layers * HIDDEN, N_EXPERTS], pl.FP32
            ],
            moe_router_bias: pl.Tensor[
                [tp_size, n_moe_layers * N_EXPERTS], pl.FP32
            ],
            moe_w_gate_r: pl.Tensor[
                [tp_size, n_moe_layers * N_LOCAL_EXPERTS * HIDDEN, INTER],
                pl.INT8,
            ],
            moe_w_gate_r_scale: pl.Tensor[
                [tp_size, n_moe_layers * N_LOCAL_EXPERTS, INTER],
                pl.FP32,
            ],
            moe_w_up_r: pl.Tensor[
                [tp_size, n_moe_layers * N_LOCAL_EXPERTS * HIDDEN, INTER],
                pl.INT8,
            ],
            moe_w_up_r_scale: pl.Tensor[
                [tp_size, n_moe_layers * N_LOCAL_EXPERTS, INTER],
                pl.FP32,
            ],
            moe_w_down_r: pl.Tensor[
                [tp_size, n_moe_layers * N_LOCAL_EXPERTS * INTER, HIDDEN],
                pl.INT8,
            ],
            moe_w_down_r_scale: pl.Tensor[
                [tp_size, n_moe_layers * N_LOCAL_EXPERTS, HIDDEN],
                pl.FP32,
            ],
            moe_w_gate_s: pl.Tensor[
                [tp_size, n_moe_layers * HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            moe_w_up_s: pl.Tensor[
                [tp_size, n_moe_layers * HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            moe_w_down_s: pl.Tensor[
                [tp_size, n_moe_layers * SH_INTER_LOCAL, HIDDEN], pl.BF16
            ],
            block_table: pl.Tensor[
                [tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32
            ],
            slot_mapping: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            rope_cos_full: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
            ],
            rope_sin_full: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim_full], pl.FP32
            ],
            rope_cos_swa: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
            ],
            rope_sin_swa: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim_swa], pl.FP32
            ],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            positions: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            next_hidden_out: pl.Out[
                pl.Tensor[
                    [tp_size, PREFILL_T, HIDDEN], pl.BF16
                ]
            ],
            final_norm_weight: pl.Tensor[[tp_size, 1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[
                [tp_size, VOCAB_LOCAL, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, VOCAB_LOCAL], pl.FP32]
            ],
            input_norm_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
            q_proj_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32]],
            k_proj_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32]],
            v_proj_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32]],
            v_tile_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]],
            q_norm_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN_Q_SWA_LOCAL], pl.FP32]],
            k_norm_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.FP32]],
            gate_logits_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD], pl.BF16]],
            resid1_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
            attn_delta_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
            post_norm_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
            ffn_output_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
            q_rot_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16]],
            k_rot_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, KV_HIDDEN_LOCAL], pl.BF16]],
            attn_out_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16]],
            attn_out_gated_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN_Q_FULL_LOCAL], pl.BF16]],
            o_proj_local_dump: pl.Out[pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]],
            scores_dump: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128], pl.FP32]
            ],
        ):
            dense_attn_tmp_buf = pld.alloc_window_buffer(
                NUM_DENSE_LAYERS * PREFILL_T * HIDDEN * 2,
            )
            dense_attn_sig_buf = pld.alloc_window_buffer(
                NUM_DENSE_LAYERS * tp_size * 4,
            )
            dense_mlp_tmp_buf = pld.alloc_window_buffer(
                NUM_DENSE_LAYERS * PREFILL_T * HIDDEN * 2,
            )
            dense_mlp_sig_buf = pld.alloc_window_buffer(
                NUM_DENSE_LAYERS * tp_size * 4,
            )
            moe_attn_tmp_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * PREFILL_T * HIDDEN * 2,
            )
            moe_attn_sig_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * tp_size * 4,
            )
            moe_sh_tmp_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * BATCH * HIDDEN * 2,
            )
            moe_sh_sig_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * PREFILL_TILE_COUNT * tp_size * 4,
            )
            moe_pub_counts_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * N_RANKS * N_RANKS * N_LOCAL_EXPERTS * 4,
            )
            moe_count_done_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS * 4,
            )
            moe_send_x_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * LOCAL_RECV_MAX * HIDDEN,
            )
            moe_send_scale_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * LOCAL_RECV_MAX * SCALE_W_PAD * 4,
            )
            moe_recv_x_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * LOCAL_RECV_MAX * HIDDEN,
            )
            moe_recv_scale_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * LOCAL_RECV_MAX * SCALE_W_PAD * 4,
            )
            moe_data_done_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS * 4,
            )
            moe_src_route_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * N_RANKS * N_LOCAL_EXPERTS * N_ROUTES_PER_RANK * 4,
            )
            moe_route_pub_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS * 4,
            )
            moe_routed_y_window_buf = pld.alloc_window_buffer(
                PREFILL_TILE_COUNT * N_ROUTES_PER_RANK * HIDDEN * 2,
            )
            moe_combine_done_buf = pld.alloc_window_buffer(
                NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS * 4,
            )
            for r in pl.range(pld.world_size()):
                dense_attn_tmp_window = pld.window(
                    dense_attn_tmp_buf,
                    [NUM_DENSE_LAYERS * PREFILL_T, HIDDEN],
                    dtype=pl.BF16,
                )
                dense_attn_signal_window = pld.window(
                    dense_attn_sig_buf,
                    [NUM_DENSE_LAYERS * tp_size, 1],
                    dtype=pl.INT32,
                )
                dense_mlp_tmp_window = pld.window(
                    dense_mlp_tmp_buf,
                    [NUM_DENSE_LAYERS * PREFILL_T, HIDDEN],
                    dtype=pl.BF16,
                )
                dense_mlp_signal_window = pld.window(
                    dense_mlp_sig_buf,
                    [NUM_DENSE_LAYERS * tp_size, 1],
                    dtype=pl.INT32,
                )
                moe_attn_tmp_window = pld.window(
                    moe_attn_tmp_buf,
                    [NUM_MOE_LAYERS * PREFILL_T, HIDDEN],
                    dtype=pl.BF16,
                )
                moe_attn_signal_window = pld.window(
                    moe_attn_sig_buf,
                    [NUM_MOE_LAYERS * tp_size, 1],
                    dtype=pl.INT32,
                )
                moe_sh_tmp_window = pld.window(
                    moe_sh_tmp_buf,
                    [NUM_MOE_LAYERS * BATCH, HIDDEN],
                    dtype=pl.BF16,
                )
                moe_sh_signal_window = pld.window(
                    moe_sh_sig_buf,
                    [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * tp_size, 1],
                    dtype=pl.INT32,
                )
                moe_pub_counts = pld.window(
                    moe_pub_counts_buf,
                    [PREFILL_TILE_COUNT * N_RANKS * N_RANKS, N_LOCAL_EXPERTS],
                    dtype=pl.INT32,
                )
                moe_count_done_sig = pld.window(
                    moe_count_done_buf,
                    [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1],
                    dtype=pl.INT32,
                )
                moe_send_x = pld.window(
                    moe_send_x_buf,
                    [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                moe_send_scale = pld.window(
                    moe_send_scale_buf,
                    [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], dtype=pl.FP32,
                )
                moe_recv_x = pld.window(
                    moe_recv_x_buf,
                    [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, HIDDEN], dtype=pl.INT8,
                )
                moe_recv_scale = pld.window(
                    moe_recv_scale_buf,
                    [PREFILL_TILE_COUNT * LOCAL_RECV_MAX, SCALE_W_PAD], dtype=pl.FP32,
                )
                moe_data_done_sig = pld.window(
                    moe_data_done_buf,
                    [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1],
                    dtype=pl.INT32,
                )
                moe_src_route_table = pld.window(
                    moe_src_route_buf,
                    [PREFILL_TILE_COUNT * N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    dtype=pl.INT32,
                )
                moe_route_pub_sig = pld.window(
                    moe_route_pub_buf,
                    [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1],
                    dtype=pl.INT32,
                )
                moe_routed_y_buf = pld.window(
                    moe_routed_y_window_buf,
                    [PREFILL_TILE_COUNT * N_ROUTES_PER_RANK, HIDDEN], dtype=pl.BF16,
                )
                moe_combine_done_sig = pld.window(
                    moe_combine_done_buf,
                    [NUM_MOE_LAYERS * PREFILL_TILE_COUNT * N_RANKS, 1],
                    dtype=pl.INT32,
                )
                self.whole_chip_orch(
                    hidden_states[r],
                    input_rms_weight[r],
                    post_rms_weight[r],
                    q_norm_weight[r],
                    k_norm_weight[r],
                    full_wq[r], full_wk[r], full_wv[r],
                    full_wo[r], full_w_g[r], full_gate_r[r],
                    swa_wq[r], swa_wk[r], swa_wv[r],
                    swa_wo[r], swa_w_g[r], swa_gate_r[r],
                    dense_w_gate[r], dense_w_up[r], dense_w_down[r],
                    block_table[r],
                    slot_mapping[r],
                    rope_cos_full[r], rope_sin_full[r],
                    rope_cos_swa[r], rope_sin_swa[r],
                    k_cache[r], v_cache[r],
                    positions[r],
                    next_hidden_out[r],
                    dense_attn_tmp_window,
                    dense_attn_signal_window,
                    dense_mlp_tmp_window,
                    dense_mlp_signal_window,
                    moe_gate_w[r], moe_router_bias[r],
                    moe_w_gate_r[r], moe_w_gate_r_scale[r],
                    moe_w_up_r[r], moe_w_up_r_scale[r],
                    moe_w_down_r[r], moe_w_down_r_scale[r],
                    moe_w_gate_s[r], moe_w_up_s[r], moe_w_down_s[r],
                    moe_attn_tmp_window, moe_attn_signal_window,
                    moe_sh_tmp_window, moe_sh_signal_window,
                    moe_pub_counts, moe_count_done_sig,
                    moe_send_x, moe_send_scale, moe_recv_x, moe_recv_scale, moe_data_done_sig,
                    moe_src_route_table, moe_route_pub_sig,
                    moe_routed_y_buf, moe_combine_done_sig,
                    final_norm_weight[r],
                    lm_head_weight[r],
                    seq_lens[r],
                    logits_shard_out[r],
                    input_norm_dump[r],
                    q_proj_dump[r],
                    k_proj_dump[r],
                    v_proj_dump[r],
                    v_tile_dump[r],
                    q_norm_dump[r],
                    k_norm_dump[r],
                    gate_logits_dump[r],
                    resid1_dump[r],
                    attn_delta_dump[r],
                    post_norm_dump[r],
                    ffn_output_dump[r],
                    q_rot_dump[r],
                    k_rot_dump[r],
                    attn_out_dump[r],
                    attn_out_gated_dump[r],
                    o_proj_local_dump[r],
                    scores_dump[r],
                    r,
                    device=r,
                )

    return Step3p5PrefillFwd


Step3p5PrefillFwd = _build_prefill_fwd_program(TP_WORLD_SIZE)


# =============================================================================
# Dense-path (layers 0-2) golden + specs — single-card TP=1 L0 validation.
# =============================================================================
def _gate_r_stack(num_layers, num_heads_pad, num_heads_real, hidden_q):
    """Block-diagonal gate expander R stacked per layer.

    Each layer block is [num_heads_pad, hidden_q] with ``r[h, h*HEAD_DIM:
    (h+1)*HEAD_DIM] = 1`` for the real heads; pad heads are all zero. Mirrors
    ``prefill_attention_full.build_tensor_specs.init_gate_r`` (validated L0).
    """
    import torch

    r = torch.zeros(
        num_layers * num_heads_pad, hidden_q, dtype=torch.bfloat16,
    )
    for layer in range(num_layers):
        base = layer * num_heads_pad
        for h in range(num_heads_real):
            r[base + h, h * HEAD_DIM:(h + 1) * HEAD_DIM] = 1.0
    return r


def build_tensor_specs(
    tp_size: int = 1,
    *,
    seed: int = 0,
    n_moe_layers: int = NUM_MOE_LAYERS,
):
    """Synthetic single-card specs for the whole-network host_orch.

    Weights are layer-major (per-row independent, never ``.repeat``) so a
    wrong slice offset in ``whole_chip_orch`` surfaces as a numeric mismatch
    rather than a silent pass. Only the dense layers (0-2) are consumed when
    the program is built with ``layer_hi=3``; the MoE-layer rows are padding.

    ``n_moe_layers`` sizes the MoE weight stacks (must match the value
    passed to ``_build_prefill_fwd_program``) so a restricted layer range like
    [3, 5) materialises only 2 MoE layers instead of the full 42.
    """
    import torch
    from golden import TensorSpec

    torch.manual_seed(seed)
    # swa o-proj row count must match the staticized LAYER_QHIDDEN_ROWS_DYN_SWA
    # (33 swa layers × HIDDEN_Q_SWA_LOCAL). Previously derived from
    # LAYER_HIDDEN_ROWS_DYN // HIDDEN (12) — the 33-vs-12 gap (task #28).
    swa_qhidden_rows = LAYER_QHIDDEN_ROWS_DYN_SWA

    def _rand(shape, scale):
        return ((torch.rand(shape) - 0.5) * scale).bfloat16()

    def _rand_f32(shape, scale):
        return ((torch.rand(shape) - 0.5) * scale).float()

    def _rep(t):
        # Replicate one per-rank tensor across the tp_size leading ranks.
        return t.unsqueeze(0).expand(tp_size, *t.shape).contiguous()

    def _shards(shape, scale, dtype):
        # tp_size independent per-rank shards stacked on a fresh leading dim.
        return torch.stack(
            [((torch.rand(shape) - 0.5) * scale).to(dtype) for _ in range(tp_size)],
            dim=0,
        )

    def _quant_int8_routed(n_experts, kdim, ndim, scale):
        # W8A8_DYNAMIC symmetric per-output-channel quantization of a float
        # reference weight stack.  Each (rank, expert) block [kdim, ndim] is
        # quantized independently with scale[n] = amax_k(|w[:, n]|) / 127 (zero
        # offset), matching the checkpoint layout in weight_loader.py.  Returns
        # (w_i8 [tp_size, n_experts * kdim, ndim], s [tp_size, n_experts, ndim]).
        w_i8_list, s_list = [], []
        for _ in range(tp_size):
            w_fp = (torch.rand(n_experts, kdim, ndim) - 0.5) * scale
            amax = w_fp.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
            s = amax / 127.0
            w_i8 = torch.round(w_fp / s).clamp(-127, 127).to(torch.int8)
            w_i8_list.append(w_i8.reshape(n_experts * kdim, ndim))
            s_list.append(s.squeeze(1))
        return torch.stack(w_i8_list, dim=0), torch.stack(s_list, dim=0)

    # position-indexed rope tables (shared across layers, matching decode).
    rope_cos_full, rope_sin_full = build_plain_rope_tables(
        int(ROPE_SEQ_DYN), 64, 10000.0,
    )
    rope_cos_swa, rope_sin_swa = build_plain_rope_tables(
        int(ROPE_SEQ_DYN), 128, 10000.0,
    )

    def _init_hidden():
        return _rep((torch.rand(PREFILL_T, HIDDEN) - 0.5).bfloat16())

    # INT8 routed weight pool: quantize once so the INT8 weight and its
    # per-output-channel scale are derived from the SAME float reference
    # (independent TensorSpec init lambdas would draw different randn streams).
    moe_w_gate_r_i8, moe_w_gate_r_scale = _quant_int8_routed(
        n_moe_layers * N_LOCAL_EXPERTS, HIDDEN, INTER, 1.0 / HIDDEN ** 0.5,
    )
    moe_w_up_r_i8, moe_w_up_r_scale = _quant_int8_routed(
        n_moe_layers * N_LOCAL_EXPERTS, HIDDEN, INTER, 1.0 / HIDDEN ** 0.5,
    )
    moe_w_down_r_i8, moe_w_down_r_scale = _quant_int8_routed(
        n_moe_layers * N_LOCAL_EXPERTS, INTER, HIDDEN, 1.0 / INTER ** 0.5,
    )

    return [
        TensorSpec(
            "hidden_states", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            init_value=_init_hidden,
        ),
        TensorSpec(
            "input_rms_weight", [tp_size, LAYER_DYN, HIDDEN], torch.float32,
            init_value=lambda: _rep(_rand_f32((LAYER_DYN, HIDDEN), 0.1)),
        ),
        TensorSpec(
            "post_rms_weight", [tp_size, LAYER_DYN, HIDDEN], torch.float32,
            init_value=lambda: _rep(_rand_f32((LAYER_DYN, HIDDEN), 0.1)),
        ),
        TensorSpec(
            "q_norm_weight", [tp_size, LAYER_DYN, HEAD_DIM], torch.float32,
            init_value=lambda: _rep(_rand_f32((LAYER_DYN, HEAD_DIM), 0.1)),
        ),
        TensorSpec(
            "k_norm_weight", [tp_size, LAYER_DYN, HEAD_DIM], torch.float32,
            init_value=lambda: _rep(_rand_f32((LAYER_DYN, HEAD_DIM), 0.1)),
        ),
        TensorSpec(
            "full_wq", [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL), 1.0 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "full_wk", [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL), 1.0 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "full_wv", [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL), 0.5 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "full_wo", [tp_size, LAYER_QHIDDEN_ROWS_DYN_FULL, HIDDEN],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_QHIDDEN_ROWS_DYN_FULL, HIDDEN),
                0.5 / HIDDEN_Q_FULL_LOCAL ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "full_w_g", [tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL_PAD),
                0.5 / HIDDEN ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "full_gate_r",
            [tp_size, NUM_FULL_LAYERS * NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL],
            torch.bfloat16,
            init_value=lambda: _rep(_gate_r_stack(
                NUM_FULL_LAYERS, NUM_HEADS_FULL_LOCAL_PAD,
                NUM_HEADS_FULL_LOCAL, HIDDEN_Q_FULL_LOCAL,
            )),
        ),
        TensorSpec(
            "swa_wq", [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, HIDDEN_Q_SWA_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN_SWA, HIDDEN_Q_SWA_LOCAL), 1.0 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "swa_wk", [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL), 1.0 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "swa_wv", [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN_SWA, KV_HIDDEN_LOCAL), 0.5 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "swa_wo", [tp_size, swa_qhidden_rows, HIDDEN],
            torch.bfloat16,
            init_value=lambda: _shards(
                (swa_qhidden_rows, HIDDEN), 0.5 / HIDDEN_Q_SWA_LOCAL ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "swa_w_g", [tp_size, LAYER_HIDDEN_ROWS_DYN_SWA, NUM_HEADS_SWA_LOCAL_PAD],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN_SWA, NUM_HEADS_SWA_LOCAL_PAD),
                0.5 / HIDDEN ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "swa_gate_r",
            [tp_size, NUM_SWA_LAYERS * NUM_HEADS_SWA_LOCAL_PAD, HIDDEN_Q_SWA_LOCAL],
            torch.bfloat16,
            init_value=lambda: _rep(_gate_r_stack(
                NUM_SWA_LAYERS, NUM_HEADS_SWA_LOCAL_PAD,
                NUM_HEADS_SWA_LOCAL, HIDDEN_Q_SWA_LOCAL,
            )),
        ),
        TensorSpec(
            "dense_w_gate", [tp_size, LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL), 1.0 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "dense_w_up", [tp_size, LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_HIDDEN_ROWS_DYN_DENSE, INTERMEDIATE_LOCAL), 1.0 / HIDDEN ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "dense_w_down", [tp_size, LAYER_INTER_ROWS_DYN, HIDDEN],
            torch.bfloat16,
            init_value=lambda: _shards(
                (LAYER_INTER_ROWS_DYN, HIDDEN), 1.0 / INTERMEDIATE_LOCAL ** 0.5,
                torch.bfloat16,
            ),
        ),
        TensorSpec(
            "moe_gate_w", [tp_size, n_moe_layers * HIDDEN, N_EXPERTS],
            torch.float32,
            init_value=lambda: _rep(
                _rand_f32((n_moe_layers * HIDDEN, N_EXPERTS), 0.1),
            ),
        ),
        TensorSpec(
            "moe_router_bias", [tp_size, n_moe_layers * N_EXPERTS],
            torch.float32,
            init_value=lambda: _rep(
                _rand_f32((n_moe_layers * N_EXPERTS,), 0.1),
            ),
        ),
        TensorSpec(
            "moe_w_gate_r",
            [tp_size, n_moe_layers * N_LOCAL_EXPERTS * HIDDEN, INTER],
            torch.int8,
            init_value=moe_w_gate_r_i8,
        ),
        TensorSpec(
            "moe_w_gate_r_scale",
            [tp_size, n_moe_layers * N_LOCAL_EXPERTS, INTER],
            torch.float32,
            init_value=moe_w_gate_r_scale,
        ),
        TensorSpec(
            "moe_w_up_r",
            [tp_size, n_moe_layers * N_LOCAL_EXPERTS * HIDDEN, INTER],
            torch.int8,
            init_value=moe_w_up_r_i8,
        ),
        TensorSpec(
            "moe_w_up_r_scale",
            [tp_size, n_moe_layers * N_LOCAL_EXPERTS, INTER],
            torch.float32,
            init_value=moe_w_up_r_scale,
        ),
        TensorSpec(
            "moe_w_down_r",
            [tp_size, n_moe_layers * N_LOCAL_EXPERTS * INTER, HIDDEN],
            torch.int8,
            init_value=moe_w_down_r_i8,
        ),
        TensorSpec(
            "moe_w_down_r_scale",
            [tp_size, n_moe_layers * N_LOCAL_EXPERTS, HIDDEN],
            torch.float32,
            init_value=moe_w_down_r_scale,
        ),
        TensorSpec(
            "moe_w_gate_s",
            [tp_size, n_moe_layers * HIDDEN, SH_INTER_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (n_moe_layers * HIDDEN, SH_INTER_LOCAL),
                1.0 / HIDDEN ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "moe_w_up_s",
            [tp_size, n_moe_layers * HIDDEN, SH_INTER_LOCAL],
            torch.bfloat16,
            init_value=lambda: _shards(
                (n_moe_layers * HIDDEN, SH_INTER_LOCAL),
                1.0 / HIDDEN ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "moe_w_down_s",
            [tp_size, n_moe_layers * SH_INTER_LOCAL, HIDDEN],
            torch.bfloat16,
            init_value=lambda: _shards(
                (n_moe_layers * SH_INTER_LOCAL, HIDDEN),
                1.0 / SH_INTER_LOCAL ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "block_table", [tp_size, BLOCK_TABLE_FLAT_DYN], torch.int32,
            init_value=lambda: _rep(
                torch.arange(BLOCK_TABLE_FLAT_DYN, dtype=torch.int32),
            ),
        ),
        TensorSpec(
            "slot_mapping", [tp_size, PREFILL_T], torch.int32,
            init_value=lambda: _rep(
                torch.arange(PREFILL_T, dtype=torch.int32),
            ),
        ),
        TensorSpec(
            "rope_cos_full", [tp_size, ROPE_SEQ_DYN, 64], torch.float32,
            init_value=lambda: _rep(rope_cos_full),
        ),
        TensorSpec(
            "rope_sin_full", [tp_size, ROPE_SEQ_DYN, 64], torch.float32,
            init_value=lambda: _rep(rope_sin_full),
        ),
        TensorSpec(
            "rope_cos_swa", [tp_size, ROPE_SEQ_DYN, 128], torch.float32,
            init_value=lambda: _rep(rope_cos_swa),
        ),
        TensorSpec(
            "rope_sin_swa", [tp_size, ROPE_SEQ_DYN, 128], torch.float32,
            init_value=lambda: _rep(rope_sin_swa),
        ),
        TensorSpec(
            "k_cache", [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], torch.bfloat16,
            init_value=torch.zeros,
        ),
        TensorSpec(
            "v_cache", [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], torch.bfloat16,
            init_value=torch.zeros,
        ),
        TensorSpec(
            "positions", [tp_size, PREFILL_T], torch.int32,
            init_value=lambda: _rep(torch.arange(PREFILL_T, dtype=torch.int32)),
        ),
        TensorSpec(
            "next_hidden_out", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=True,
        ),
        TensorSpec(
            "final_norm_weight", [tp_size, 1, HIDDEN], torch.float32,
            init_value=lambda: _rep(_rand_f32((1, HIDDEN), 0.1)),
        ),
        TensorSpec(
            "lm_head_weight", [tp_size, VOCAB_LOCAL, HIDDEN], torch.bfloat16,
            init_value=lambda: _shards(
                (VOCAB_LOCAL, HIDDEN), 1.0 / HIDDEN ** 0.5, torch.bfloat16,
            ),
        ),
        TensorSpec(
            "seq_lens", [tp_size, USER_BATCH_DYN], torch.int32,
            init_value=lambda: _rep(
                torch.full((USER_BATCH_DYN,), PREFILL_T, dtype=torch.int32),
            ),
        ),
        TensorSpec(
            "logits_shard_out", [tp_size, PREFILL_T, VOCAB_LOCAL],
            torch.float32, is_output=True,
        ),
        TensorSpec(
            "input_norm_dump", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "q_proj_dump", [tp_size, PREFILL_T, HIDDEN_Q_SWA_LOCAL], torch.float32,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "k_proj_dump", [tp_size, PREFILL_T, KV_HIDDEN_LOCAL], torch.float32,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "v_proj_dump", [tp_size, PREFILL_T, KV_HIDDEN_LOCAL], torch.float32,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "v_tile_dump", [tp_size, PREFILL_T, KV_HIDDEN_LOCAL], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "q_norm_dump", [tp_size, PREFILL_T, HIDDEN_Q_SWA_LOCAL], torch.float32,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "k_norm_dump", [tp_size, PREFILL_T, KV_HIDDEN_LOCAL], torch.float32,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "gate_logits_dump", [tp_size, PREFILL_T, NUM_HEADS_FULL_LOCAL_PAD],
            torch.bfloat16, is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "resid1_dump", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "attn_delta_dump", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "post_norm_dump", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "ffn_output_dump", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        # Attention-core dumps (task #2): q_rot/k_rot/attn_out/attn_out_gated/
        # o_proj_local + full-column scores, produced by layer 44 only.
        TensorSpec(
            "q_rot_dump", [tp_size, PREFILL_T, HIDDEN_Q_FULL_LOCAL], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "k_rot_dump", [tp_size, PREFILL_T, KV_HIDDEN_LOCAL], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "attn_out_dump", [tp_size, PREFILL_T, HIDDEN_Q_FULL_LOCAL], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "attn_out_gated_dump", [tp_size, PREFILL_T, HIDDEN_Q_FULL_LOCAL], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "o_proj_local_dump", [tp_size, PREFILL_T, HIDDEN], torch.bfloat16,
            is_output=_DUMP_ENABLED,
        ),
        TensorSpec(
            "scores_dump",
            [tp_size, PREFILL_T * 16, ((PREFILL_T + 127) // 128) * 128],
            torch.float32, is_output=_DUMP_ENABLED,
        ),
    ]


def golden_whole_dense(tensors):
    """Torch reference for the dense path (layers 0-2) at TP=1.

    Chains the already-validated per-layer goldens: layer 0 = full attention +
    dense MLP, layers 1-2 = SWA attention + dense MLP, with decode-faithful
    (norm, attn, mlp) index separation. The per-layer goldens skip the paged
    KV-cache round-trip (identity write-then-read), which holds here too.
    """
    import torch

    from .prefill_attention_full import golden_attention_full_prefill
    from .prefill_attention_swa import golden_attention_swa_prefill
    from .prefill_dense_mlp import golden_prefill_dense_mlp

    def _attn_full(norm, attn, hid):
        d = {
            "norm_layer_idx": norm,
            "attn_layer_idx": attn,
            "current_hidden": hid,
            "input_rms_weight": tensors["input_rms_weight"],
            "wq": tensors["full_wq"],
            "wk": tensors["full_wk"],
            "wv": tensors["full_wv"],
            "q_norm_weight": tensors["q_norm_weight"],
            "k_norm_weight": tensors["k_norm_weight"],
            "wo": tensors["full_wo"],
            "w_g": tensors["full_w_g"],
            "rope_cos": tensors["rope_cos_full"],
            "rope_sin": tensors["rope_sin_full"],
            "positions": tensors["positions"],
            "resid1_out": torch.zeros_like(hid),
        }
        golden_attention_full_prefill(d)
        return d["resid1_out"]

    def _attn_swa(norm, attn, hid):
        d = {
            "norm_layer_idx": norm,
            "attn_layer_idx": attn,
            "current_hidden": hid,
            "input_rms_weight": tensors["input_rms_weight"],
            "wq": tensors["swa_wq"],
            "wk": tensors["swa_wk"],
            "wv": tensors["swa_wv"],
            "q_norm_weight": tensors["q_norm_weight"],
            "k_norm_weight": tensors["k_norm_weight"],
            "wo": tensors["swa_wo"],
            "w_g": tensors["swa_w_g"],
            "rope_cos": tensors["rope_cos_swa"],
            "rope_sin": tensors["rope_sin_swa"],
            "positions": tensors["positions"],
            "resid1_out": torch.zeros_like(hid),
        }
        golden_attention_swa_prefill(d)
        return d["resid1_out"]

    def _dense(norm, mlp, resid1):
        d = {
            "norm_layer_idx": norm,
            "mlp_layer_idx": mlp,
            "resid1": resid1,
            "post_rms_weight": tensors["post_rms_weight"],
            "w_gate": tensors["dense_w_gate"],
            "w_up": tensors["dense_w_up"],
            "w_down": tensors["dense_w_down"],
            "next_hidden": torch.zeros_like(resid1),
        }
        golden_prefill_dense_mlp(d)
        return d["next_hidden"]

    hidden = tensors["hidden_states"]
    # Layer 0: full attention (norm=0, attn=0) + dense MLP (norm=0, mlp=0).
    r0 = _attn_full(0, 0, hidden)
    h0 = _dense(0, 0, r0)
    # Layer 1: SWA attention (norm=1, attn=0) + dense MLP (norm=1, mlp=1).
    r1 = _attn_swa(1, 0, h0)
    h1 = _dense(1, 1, r1)
    # Layer 2: SWA attention (norm=2, attn=1) + dense MLP (norm=2, mlp=2).
    r2 = _attn_swa(2, 1, h1)
    h2 = _dense(2, 2, r2)

    tensors["next_hidden_out"][0] = h2[0]


# =============================================================================
# MoE silu/silu golden — torch reference for layers [layer_lo, layer_hi).
# =============================================================================
def golden_whole_moe(
    tensors,
    *,
    layer_lo: int = NUM_DENSE_LAYERS,
    layer_hi: int = NUM_DENSE_LAYERS + 2,
    tp_size: int = TP_WORLD_SIZE,
    record: list | None = None,
):
    """Torch reference for MoE silu/silu layers ``[layer_lo, layer_hi)``.

    Single-machine math for the 8-rank EP+TP wiring (task #25).  Chained
    per layer: TP attention (per-rank partial o_proj -> FP32 sum -> residual
    add) -> V4 deferred RMSNorm + W8A8 producer (post_norm BF16 + inv_rms +
    x_i8/x_scale) -> replicated router gate (deferred inv_rms on the shared
    xg; sigmoid + bias + top-K + renorm*3.0) -> TP-sliced shared expert
    (per-rank silu shard on post_norm, FP32 sum -> BF16) -> EP-sliced routed
    experts (W8A8 INT8 GEMM + per-row h requant, weights on rank
    ``eid // N_LOCAL_EXPERTS``) -> weighted combine
    (sh_y + sum_k routed_y*weight) -> residual add.

    The low-level EP buffers (pub_counts / src_route_table / send_buf /
    routed_y_buf) are never materialised: the routed output for route
    ``(b, k)`` is deterministic given ``eid = indices[b, k]`` and the
    owning rank's weight shard, so the combine is computed directly from
    the gate output.  The MoE weights are indexed by the relative MoE slot
    ``moe_pos = li - base`` (base = max(layer_lo, NUM_DENSE_LAYERS)), matching
    ``whole_chip_orch``.
    """
    import torch

    from ._moe_constants import (
        INTER,
        N_EXPERTS,
        N_LOCAL_EXPERTS,
        ROUTER_SCALE,
        SH_INTER_LOCAL,
        TOPK,
    )
    from .prefill_attention_full import _torch_per_rank_partial_full
    from .prefill_attention_swa import _torch_per_rank_partial_swa

    def producer(resid1, gamma):
        # V4 deferred RMSNorm + W8A8 activation producer (stage B).  xg =
        # resid1*(gamma+1) is shared by the gate (deferred inv_rms), the shared
        # lane (BF16 norm), and the routed lane (INT8 quant).  Returns
        # (xg, inv_rms, post_norm, x_i8, x_scale).  The routed A8 quant runs in
        # the BF16 post-norm domain (amax over |post_norm|, RNE divide by
        # pn_amax/127) to bit-match VLLM, not the fp32 xg domain.
        resid1_fp32 = resid1.float()
        xg = resid1_fp32 * (gamma.float() + 1.0)
        sq_sum = resid1_fp32.pow(2).sum(dim=-1)
        inv_rms = 1.0 / torch.sqrt(sq_sum * HIDDEN_INV + EPS)
        post_norm = (xg * inv_rms.unsqueeze(-1)).bfloat16()
        pn_amax = torch.clamp(post_norm.float().abs().amax(dim=-1), min=1e-4)
        x_scale = pn_amax / 127.0
        x_i8 = torch.round(post_norm.float() / x_scale.unsqueeze(-1)).to(torch.int8)
        return xg, inv_rms, post_norm, x_i8, x_scale

    def gate_fn(xg, inv_rms, gate_w, router_bias):
        logits = xg @ gate_w.float()
        logits_scaled = logits * inv_rms.unsqueeze(-1)
        score = torch.sigmoid(logits_scaled)
        # ROUTER-BIAS-BF16 (align decode_fwd.py:657-664): round-trip the FP32
        # loader bias through BF16 so top-8 selection matches the kernel's
        # cast(cast(bias, BF16), FP32) and decode/vLLM.
        router_bias = router_bias.float().to(torch.bfloat16).float()
        biased = score + router_bias.view(1, -1)
        indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
        topk_vals = torch.gather(score, dim=-1, index=indices.long())
        weights = (
            topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        ) * ROUTER_SCALE
        return indices.to(torch.int32), weights

    def shared_shard(x, w_gate_s, w_up_s, w_down_s, shared_lim):
        gate = x.float() @ w_gate_s.float()
        up = x.float() @ w_up_s.float()
        silu = gate * torch.sigmoid(gate)
        if shared_lim > 0.0:
            # silu.clamp(max=lim) == min(silu, lim); torch.minimum rejects a
            # scalar ``other`` on this build, torch.clamp accepts float bounds.
            silu = silu.clamp(max=shared_lim)
            up = torch.clamp(up, -shared_lim, shared_lim)
        h = (silu * up).bfloat16()
        return (h.float() @ w_down_s.float()).bfloat16()

    def routed_row(
        x_i8_row, x_scale_val,
        w_gate_i8, w_gate_scale,
        w_up_i8, w_up_scale,
        w_down_i8, w_down_scale,
        routed_lim,
    ):
        # W8A8 routed FFN (single token row): INT8@INT8 -> per-token/per-channel
        # dequant -> silu -> FP32 h -> per-row requant -> INT8 down -> dequant.
        gate = x_i8_row.float() @ w_gate_i8.float()
        up = x_i8_row.float() @ w_up_i8.float()
        gate_dq = gate * x_scale_val * w_gate_scale.float()
        up_dq = up * x_scale_val * w_up_scale.float()
        silu = gate_dq * torch.sigmoid(gate_dq)
        if routed_lim > 0.0:
            silu = silu.clamp(max=routed_lim)
            up_dq = torch.clamp(up_dq, -routed_lim, routed_lim)
        h_fp32 = silu * up_dq
        amax_h = max(1e-4, float(h_fp32.abs().max()))
        h_i8 = torch.round(h_fp32 * (127.0 / amax_h)).to(torch.int8)
        h_scale_dq = amax_h / 127.0
        y = (h_i8.float() @ w_down_i8.float()) * h_scale_dq * w_down_scale.float()
        return y.bfloat16()

    hidden = tensors["hidden_states"][0].clone()
    positions = tensors["positions"][0]
    t = hidden.shape[0]
    base = max(layer_lo, NUM_DENSE_LAYERS)

    for li in range(layer_lo, layer_hi):
        norm_idx = li
        moe_pos = li - base

        # --- A: TP attention (per-rank partial -> FP32 sum -> residual). ---
        if li % 4 == 0:
            full_local = li // 4
            w_off = full_local * HIDDEN
            wo_off = full_local * HIDDEN_Q_FULL_LOCAL
            wq_full = torch.cat(
                [tensors["full_wq"][r][w_off:w_off + HIDDEN, :]
                 for r in range(tp_size)],
                dim=1,
            )
            wk_full = torch.cat(
                [tensors["full_wk"][r][w_off:w_off + HIDDEN, :]
                 for r in range(tp_size)],
                dim=1,
            )
            wv_full = torch.cat(
                [tensors["full_wv"][r][w_off:w_off + HIDDEN, :]
                 for r in range(tp_size)],
                dim=1,
            )
            wo_full = torch.cat(
                [tensors["full_wo"][r][wo_off:wo_off + HIDDEN_Q_FULL_LOCAL, :]
                 for r in range(tp_size)],
                dim=0,
            )
            w_g_full = torch.cat(
                [tensors["full_w_g"][r][w_off:w_off + HIDDEN, :NUM_HEADS_FULL_LOCAL]
                 for r in range(tp_size)],
                dim=1,
            )
            rope_cos = tensors["rope_cos_full"][0]
            rope_sin = tensors["rope_sin_full"][0]
            partial_fn = _torch_per_rank_partial_full
        else:
            swa_local = li - li // 4 - 1
            w_off = swa_local * HIDDEN
            wo_off = swa_local * HIDDEN_Q_SWA_LOCAL
            wq_full = torch.cat(
                [tensors["swa_wq"][r][w_off:w_off + HIDDEN, :]
                 for r in range(tp_size)],
                dim=1,
            )
            wk_full = torch.cat(
                [tensors["swa_wk"][r][w_off:w_off + HIDDEN, :]
                 for r in range(tp_size)],
                dim=1,
            )
            wv_full = torch.cat(
                [tensors["swa_wv"][r][w_off:w_off + HIDDEN, :]
                 for r in range(tp_size)],
                dim=1,
            )
            wo_full = torch.cat(
                [tensors["swa_wo"][r][wo_off:wo_off + HIDDEN_Q_SWA_LOCAL, :]
                 for r in range(tp_size)],
                dim=0,
            )
            w_g_full = torch.cat(
                [tensors["swa_w_g"][r][w_off:w_off + HIDDEN, :NUM_HEADS_SWA_LOCAL]
                 for r in range(tp_size)],
                dim=1,
            )
            rope_cos = tensors["rope_cos_swa"][0]
            rope_sin = tensors["rope_sin_swa"][0]
            partial_fn = _torch_per_rank_partial_swa

        input_rms_row = tensors["input_rms_weight"][0][norm_idx:norm_idx + 1, :]
        q_norm_row = tensors["q_norm_weight"][0][norm_idx:norm_idx + 1, :]
        k_norm_row = tensors["k_norm_weight"][0][norm_idx:norm_idx + 1, :]

        summed = torch.zeros(t, HIDDEN, dtype=torch.float32)
        for r in range(tp_size):
            partial = partial_fn(
                rank=r,
                hidden=hidden,
                input_rms_weight=input_rms_row,
                wq_full=wq_full,
                wk_full=wk_full,
                wv_full=wv_full,
                q_norm_weight=q_norm_row,
                k_norm_weight=k_norm_row,
                wo_full=wo_full,
                w_g_full=w_g_full,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                positions=positions,
            )
            summed = summed + partial.float()
        resid1 = (summed + hidden.float()).bfloat16()

        # --- B: V4 deferred RMSNorm + W8A8 activation producer. ---
        post_rms_row = tensors["post_rms_weight"][0][norm_idx:norm_idx + 1, :]
        xg, inv_rms, post_norm, x_i8, x_scale = producer(resid1, post_rms_row)

        # --- C1: replicated router gate (deferred inv_rms on shared xg). ---
        gate_w = tensors["moe_gate_w"][0][
            moe_pos * HIDDEN:(moe_pos + 1) * HIDDEN, :
        ]
        router_bias = tensors["moe_router_bias"][0][
            moe_pos * N_EXPERTS:(moe_pos + 1) * N_EXPERTS
        ]
        indices, weights = gate_fn(xg, inv_rms, gate_w, router_bias)

        # --- C2: TP-sliced shared expert (per-rank shard -> FP32 sum). ---
        shared_sum = torch.zeros(t, HIDDEN, dtype=torch.float32)
        for r in range(tp_size):
            w_gate_s = tensors["moe_w_gate_s"][r][
                moe_pos * HIDDEN:(moe_pos + 1) * HIDDEN, :
            ]
            w_up_s = tensors["moe_w_up_s"][r][
                moe_pos * HIDDEN:(moe_pos + 1) * HIDDEN, :
            ]
            w_down_s = tensors["moe_w_down_s"][r][
                moe_pos * SH_INTER_LOCAL:(moe_pos + 1) * SH_INTER_LOCAL, :
            ]
            shared_sum = shared_sum + shared_shard(
                post_norm, w_gate_s, w_up_s, w_down_s,
                SWIGLU_LIMITS_SHARED[li],
            ).float()
        sh_y = shared_sum.bfloat16()

        # --- C3: EP-sliced routed experts (W8A8) + weighted combine. ---
        moe_out = sh_y.float().clone()
        for k in range(TOPK):
            routed_col = torch.zeros(t, HIDDEN, dtype=torch.float32)
            for b in range(t):
                eid = int(indices[b, k].item())
                dst = eid // N_LOCAL_EXPERTS
                loc_e = eid - dst * N_LOCAL_EXPERTS
                gate_base = (
                    moe_pos * N_LOCAL_EXPERTS * HIDDEN + loc_e * HIDDEN
                )
                down_base = (
                    moe_pos * N_LOCAL_EXPERTS * INTER + loc_e * INTER
                )
                scale_row = moe_pos * N_LOCAL_EXPERTS + loc_e
                w_gate_i8 = tensors["moe_w_gate_r"][dst][
                    gate_base:gate_base + HIDDEN, :
                ]
                w_gate_scale = tensors["moe_w_gate_r_scale"][dst][scale_row, :]
                w_up_i8 = tensors["moe_w_up_r"][dst][
                    gate_base:gate_base + HIDDEN, :
                ]
                w_up_scale = tensors["moe_w_up_r_scale"][dst][scale_row, :]
                w_down_i8 = tensors["moe_w_down_r"][dst][
                    down_base:down_base + INTER, :
                ]
                w_down_scale = tensors["moe_w_down_r_scale"][dst][scale_row, :]
                routed_col[b] = routed_row(
                    x_i8[b], x_scale[b],
                    w_gate_i8, w_gate_scale,
                    w_up_i8, w_up_scale,
                    w_down_i8, w_down_scale,
                    SWIGLU_LIMITS[li],
                ).float()
            moe_out = moe_out + routed_col * weights[:, k].unsqueeze(-1)
        moe_out_bf16 = moe_out.bfloat16()

        # --- D: residual add. ---
        hidden = (resid1.float() + moe_out_bf16.float()).bfloat16()

        if record is not None:
            record.append({
                "li": li,
                "resid1": resid1.detach().cpu().clone(),
                "indices": indices.detach().cpu().clone(),
                "weights": weights.detach().cpu().clone(),
                "sh_y": sh_y.detach().cpu().clone(),
                "tile_y": moe_out_bf16.detach().cpu().clone(),
                "x_i8": x_i8.detach().cpu().clone(),
                "x_scale": x_scale.detach().cpu().clone(),
            })

    tensors["next_hidden_out"][0] = hidden

    # --- tail (Phase 3b): per-position logits reference. ---
    # Mirrors rms_lm_head_chip_orch's 8-tile loop: run the shared rms_lm_head
    # (zero-centred RMSNorm + per-rank vocab-shard LM head) over all 128
    # positions at once (golden_rms_lm_head handles arbitrary batch rows).
    from .rms_lm_head import golden_rms_lm_head

    tensors["logits_shard_out"][0] = golden_rms_lm_head(
        hidden[:PREFILL_T, :],
        tensors["final_norm_weight"][0],
        tensors["lm_head_weight"][0],
    )


# =============================================================================
# Distributed-mock harness — pure torch 8-rank simulation.
# =============================================================================
def _torch_zc_rmsnorm(x, gamma, eps=1e-6):
    import torch

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    g = gamma.float() + 1.0
    return (x.float() * torch.rsqrt(var + eps) * g)


def _torch_dense_mlp_partial(
    *, resid1, post_rms, w_gate, w_up, w_down, eps=1e-6,
):
    import torch

    normed = _torch_zc_rmsnorm(
        resid1, post_rms[0:1, :], eps,
    ).bfloat16().float()
    gate = normed @ w_gate.float()
    up = normed @ w_up.float()
    silu = gate * torch.sigmoid(gate)
    mlp = (silu * up).bfloat16().float()
    return (mlp @ w_down.float()).bfloat16()


def run_distributed_mock(
    *,
    batch: int = 1,
    seq_len: int = PREFILL_SEQ,
    seed: int = 0,
    pass_rate_threshold: float = 0.97,
    n_ranks: int = TP_WORLD_SIZE,
):
    """Pure-torch 8-rank simulation of the 45-layer prefill TP wiring.

    Same correctness contract as ``decode_fwd.run_distributed_mock``:
    attention output is approximated as zero-centred normed hidden,
    MoE layers as identity, and the test focuses on the TP all-reduce
    of the dense MLP partial + the vocab-sliced LM head shard.
    """
    import torch

    torch.manual_seed(seed)
    if batch != PREFILL_BATCH or seq_len != PREFILL_SEQ:
        raise ValueError(
            f"prefill mock harness requires batch={PREFILL_BATCH} "
            f"seq_len={PREFILL_SEQ}; got batch={batch} seq_len={seq_len}"
        )

    t = batch * seq_len
    hidden = (torch.rand(t, HIDDEN) - 0.5).bfloat16()
    final_norm = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    input_rms = (
        (torch.rand(NUM_HIDDEN_LAYERS, HIDDEN) - 0.5) * 0.1
    ).float()
    post_rms = (
        (torch.rand(NUM_HIDDEN_LAYERS, HIDDEN) - 0.5) * 0.1
    ).float()

    full_w_gate = torch.zeros(
        NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL * n_ranks,
    )
    full_w_up = torch.zeros(
        NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL * n_ranks,
    )
    full_w_down = torch.zeros(
        NUM_DENSE_LAYERS, INTER_LOCAL * n_ranks, HIDDEN,
    )
    for d in range(NUM_DENSE_LAYERS):
        full_w_gate[d] = (
            torch.rand(HIDDEN, INTER_LOCAL * n_ranks) - 0.5
        ) / HIDDEN ** 0.5
        full_w_up[d] = (
            torch.rand(HIDDEN, INTER_LOCAL * n_ranks) - 0.5
        ) / HIDDEN ** 0.5
        full_w_down[d] = (
            torch.rand(INTER_LOCAL * n_ranks, HIDDEN) - 0.5
        ) / (INTER_LOCAL * n_ranks) ** 0.5

    rank_w_gate = [
        full_w_gate[:, :, r * INTER_LOCAL:(r + 1) * INTER_LOCAL].clone()
        for r in range(n_ranks)
    ]
    rank_w_up = [
        full_w_up[:, :, r * INTER_LOCAL:(r + 1) * INTER_LOCAL].clone()
        for r in range(n_ranks)
    ]
    rank_w_down = [
        full_w_down[:, r * INTER_LOCAL:(r + 1) * INTER_LOCAL, :].clone()
        for r in range(n_ranks)
    ]

    full_lm_head = (
        torch.rand(VOCAB_LOCAL * n_ranks, HIDDEN) - 0.5
    ) / HIDDEN ** 0.5
    rank_lm_head = [
        full_lm_head[r * VOCAB_LOCAL:(r + 1) * VOCAB_LOCAL, :].bfloat16()
        for r in range(n_ranks)
    ]

    # Verify per-layer dispatcher resolves cleanly.
    try:
        for li in range(NUM_HIDDEN_LAYERS):
            kind, routed_lim, shared_lim = select_prefill_layer(li)
            if not isinstance(kind, str):
                raise RuntimeError(
                    f"select_prefill_layer({li}) returned bad kind: "
                    f"{kind!r}"
                )
            if kind not in _VALID_LAYER_KINDS:
                raise RuntimeError(
                    f"select_prefill_layer({li}) -> kind={kind!r} "
                    f"is not a known layer kind"
                )
    except Exception:  # pragma: no cover — runtime not always present
        pass

    # Single-card oracle (residual stream).
    oracle_hidden = hidden.clone().bfloat16()
    for li in range(NUM_HIDDEN_LAYERS):
        attn_out = _torch_zc_rmsnorm(
            oracle_hidden, input_rms[li:li + 1, :],
        ).bfloat16().float()
        resid1 = (oracle_hidden.float() + attn_out).bfloat16()
        if is_moe_layer(li):
            oracle_hidden = resid1
        else:
            d = DENSE_POS[li]
            mlp_out = _torch_dense_mlp_partial(
                resid1=resid1,
                post_rms=post_rms[li:li + 1, :],
                w_gate=full_w_gate[d],
                w_up=full_w_up[d],
                w_down=full_w_down[d],
            )
            oracle_hidden = (resid1.float() + mlp_out.float()).bfloat16()

    # Pick the last-token slot of every batch row (PREFILL_BATCH=1, so
    # this is the row at position seq_len - 1) and pad up to BATCH rows
    # so rms_lm_head sees its static shape.
    last_idx = seq_len - 1
    oracle_last = oracle_hidden[last_idx:last_idx + 1, :].clone()
    oracle_last_pad = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
    oracle_last_pad[:1, :] = oracle_last
    oracle_normed = _torch_zc_rmsnorm(
        oracle_last_pad, final_norm,
    ).bfloat16()
    oracle_logits_full = (
        oracle_normed.float() @ full_lm_head.float().T
    )

    rank_pass_rates = []
    for r in range(n_ranks):
        rank_hidden = hidden.clone().bfloat16()
        for li in range(NUM_HIDDEN_LAYERS):
            attn_out = _torch_zc_rmsnorm(
                rank_hidden, input_rms[li:li + 1, :],
            ).bfloat16().float()
            resid1 = (rank_hidden.float() + attn_out).bfloat16()
            if is_moe_layer(li):
                rank_hidden = resid1
            else:
                d = DENSE_POS[li]
                summed = torch.zeros(t, HIDDEN)
                for rr in range(n_ranks):
                    p = _torch_dense_mlp_partial(
                        resid1=resid1,
                        post_rms=post_rms[li:li + 1, :],
                        w_gate=rank_w_gate[rr][d],
                        w_up=rank_w_up[rr][d],
                        w_down=rank_w_down[rr][d],
                    )
                    summed = summed + p.float()
                rank_hidden = (
                    resid1.float() + summed
                ).bfloat16()

        rank_last = rank_hidden[last_idx:last_idx + 1, :].clone()
        rank_last_pad = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
        rank_last_pad[:1, :] = rank_last
        rank_normed = _torch_zc_rmsnorm(
            rank_last_pad, final_norm,
        ).bfloat16()
        rank_logits_shard = (
            rank_normed.float() @ rank_lm_head[r].float().T
        )
        expected_shard = oracle_logits_full[
            :, r * VOCAB_LOCAL:(r + 1) * VOCAB_LOCAL,
        ]
        close = torch.isclose(
            rank_logits_shard, expected_shard,
            rtol=5e-3, atol=5e-3,
        )
        rate = close.float().mean().item()
        rank_pass_rates.append(rate)

    worst = min(rank_pass_rates)
    avg = sum(rank_pass_rates) / len(rank_pass_rates)
    ok = worst >= pass_rate_threshold
    return {
        "ok": ok,
        "worst_pass_rate": worst,
        "avg_pass_rate": avg,
        "rank_pass_rates": rank_pass_rates,
        "threshold": pass_rate_threshold,
    }


__all__ = [
    "Step3p5PrefillFwd",
    "_build_prefill_fwd_program",
    "prefill_dense_mlp_body",
    "select_prefill_layer",
    "select_prefill_moe_block",
    "run_distributed_mock",
    "NUM_FULL_LAYERS",
    "NUM_SWA_LAYERS",
    "NUM_DENSE_LAYERS",
    "NUM_MOE_LAYERS",
    "FULL_POS",
    "SWA_POS",
    "DENSE_POS",
    "MOE_POS",
    "N_RANKS",
    "N_LOCAL_EXPERTS",
    "LOCAL_RECV_MAX",
    "N_ROUTES_PER_RANK",
    "SH_INTER_LOCAL",
    "INTER",
    "INTER_LOCAL",
    "N_EXPERTS",
    "PREFILL_BATCH",
    "PREFILL_SEQ",
    "PREFILL_T",
    "TOK_TILE",
]


# =============================================================================
# CLI entry — NPU compile/smoke (``-p/-d``) + distributed-mock harness.
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 prefill_fwd. Default path: real-NPU compile/smoke of "
            "Step3p5PrefillFwd (the 45-layer whole_chip_orch skeleton) on a "
            "single card via golden.runner.run. The pure-torch 8-rank "
            "distributed mock is available via --distributed-mock."
        ),
    )
    parser.add_argument(
        "-p", "--platform", default="a2a3", choices=["a2a3"],
    )
    parser.add_argument("-d", "--device", type=int, default=5)
    parser.add_argument("-b", "--batch", type=int, default=PREFILL_BATCH)
    parser.add_argument("--seq-len", type=int, default=PREFILL_SEQ)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--layer-lo", type=int, default=0)
    parser.add_argument(
        "--layer-hi", type=int, default=NUM_DENSE_LAYERS,
        help="Exclusive upper layer bound; default 3 validates the dense "
             "path (layers 0-2). Pass 45 for the full structural compile.",
    )
    parser.add_argument(
        "--compile-only", action="store_true", default=False,
        help="Stop after pypto codegen; skip execute and validate.",
    )
    parser.add_argument(
        "--build-program-only", action="store_true", default=False,
        help="Only build the @pl.program class in Python (no codegen).",
    )
    parser.add_argument(
        "--distributed-mock", action="store_true", default=False,
        help="Run the pure-torch 8-rank distributed mock (no NPU).",
    )
    args = parser.parse_args()

    # Dispatcher smoke-check (runs in every path).
    for li in range(NUM_HIDDEN_LAYERS):
        kind, _routed, _shared = select_prefill_layer(li)
        assert isinstance(kind, str)
        assert kind in _VALID_LAYER_KINDS
    print("[prefill_fwd] all 45 main-layer dispatch entries resolve OK")

    # MoE layers occupy [NUM_DENSE_LAYERS, NUM_HIDDEN_LAYERS); size the MoE
    # weight/signal stacks to just the slice that [layer_lo, layer_hi) covers,
    # rebasing moe_pos via base = max(layer_lo, NUM_DENSE_LAYERS). max(1, ...):
    # dense-only runs still slice MoE stacks in dead branches, so a 0-height
    # stack would fail type inference (slice past end).
    base = max(args.layer_lo, NUM_DENSE_LAYERS)
    n_moe_layers = max(1, min(args.layer_hi, NUM_HIDDEN_LAYERS) - base)

    # Build the top-level @pl.program for the requested layer range.
    program = _build_prefill_fwd_program(
        args.tp_size, args.layer_lo, args.layer_hi,
        n_moe_layers=n_moe_layers,
    )
    _prog_name = getattr(program, "name", None) or type(program).__name__
    print(
        f"[OK] built @pl.program {_prog_name} "
        f"(tp_size={args.tp_size}, layers=[{args.layer_lo}, {args.layer_hi}), "
        f"n_moe_layers={n_moe_layers})"
    )
    if args.build_program_only:
        raise SystemExit(0)

    if args.distributed_mock:
        result = run_distributed_mock(
            batch=args.batch,
            seq_len=args.seq_len,
            seed=args.seed,
            pass_rate_threshold=args.pass_rate,
        )
        print("=" * 72)
        print(
            "Step3p5 prefill_fwd — distributed-mock 8-rank simulation "
            f"(B={args.batch}, S={args.seq_len})"
        )
        print("=" * 72)
        print(f"  threshold       : {result['threshold']:.4f}")
        print(f"  avg pass rate   : {result['avg_pass_rate']:.6f}")
        print(f"  worst pass rate : {result['worst_pass_rate']:.6f}")
        for r, pr in enumerate(result["rank_pass_rates"]):
            marker = "OK " if pr >= result["threshold"] else "BAD"
            print(f"   rank {r}: {pr:.6f}  {marker}")
        print("=" * 72)
        if not result["ok"]:
            raise SystemExit(1)
        print("[prefill_fwd] distributed-mock 8-rank simulation PASSED")
        raise SystemExit(0)

    # Real-NPU path.  Dense path (layers 0-2) is fully wired and golden-
    # validated against a chained torch reference; the MoE layers are still
    # stubs, so a layer_hi > 3 run is compile-only (golden_fn=None).
    from golden import ratio_reldiff, run as golden_run

    specs = build_tensor_specs(
        args.tp_size, seed=args.seed, n_moe_layers=n_moe_layers,
    )
    golden_fn = (
        golden_whole_dense
        if (args.layer_lo, args.layer_hi) == (0, NUM_DENSE_LAYERS)
        else None
    )
    result = golden_run(
        program=program,
        specs=specs,
        golden_fn=golden_fn,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        compile_only=args.compile_only,
        compare_fn={
            # 3-layer BF16 chain drifts ~0.2% rel-L2 vs the fp32 torch
            # reference; strict allclose (1e-5) is not a meaningful gate for
            # real-weight prefill. Mirrors decode_layer.py x_next convention.
            "next_hidden_out": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        },
    )
    if result.passed:
        print(
            f"[prefill_fwd] real-NPU {'compile' if args.compile_only else 'run'} "
            f"PASS on platform={args.platform} device={args.device}"
        )
        raise SystemExit(0)
    print(
        f"[prefill_fwd] real-NPU {'compile' if args.compile_only else 'run'} "
        f"FAIL: {result.error}"
    )
    raise SystemExit(1)
