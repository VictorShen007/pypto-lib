# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident whole-net prefill holder (per-position logits).

Mirrors ``whole_decode_holder.WholeDecodeHolder`` but drives the Step3p5
PREFILL graph ``models.step3p5.prefill_fwd.Step3p5PrefillFwd`` (45 layers,
``PREFILL_T = 128`` tokens) instead of the per-token decode graph.

Differences from the decode holder (each is a deliberate T2 decision):

* **Per-position logits.**  ``run()`` returns BOTH the post-45-layer hidden
  (``next_hidden``, ``[PREFILL_T, HIDDEN]`` BF16) and the vocab-sliced
  per-position logits shard (``logits``, ``[PREFILL_T, VOCAB_LOCAL]`` FP32).
  The decode holder is hidden-only; prefill owns the final RMSNorm + LM-head
  tail (Phase 3b), so the exporter must run with
  ``production_hidden_only=False`` and the holder must feed ``final_norm`` /
  ``lm_head`` back in.
* **Flattened weight ABI.**  ``prefill_fwd`` declares its attention/MLP/MoE
  weight stacks in a layer-major FLATTENED layout (e.g. ``full_wq`` is
  ``[LAYER_HIDDEN_ROWS_DYN=49152, HIDDEN_Q_FULL_LOCAL]`` = 12 full layers x
  4096 rows stacked on the leading dim), whereas ``weight_loader`` emits the
  decode-faithful per-layer 3D/4D buckets (``KEY_WQ_FULL`` = ``[12, 4096,
  1024]``).  The two layouts are the SAME bytes in the same layer order, so
  the holder imports the zero-copy IPC pool and re-views each weight through
  ``DeviceTensor.reshape`` (metadata-only, no copy).
* **Empty KV pool (Gap C).**  ``kv_ipc`` is not used; ``k_cache``/``v_cache``
  are plain resident host tensors zeroed once.  This is a diagnostic ABI, not
  a live vLLM bridge.
* **Dense gate/up exact reshape.**  ``dense_w_gate``/``dense_w_up`` are
  declared ``[LAYER_HIDDEN_ROWS_DYN_DENSE=12288, 1408]`` = 3 dense layers x 4096 rows
  (blue MAJOR: previously reused ``LAYER_HIDDEN_ROWS_DYN=49152``).  The
  checkpoint supplies exactly 3 dense layers, so the holder uses a pure
  reshape — no zero-padding.  ``dense_w_down`` is ``[3*1408, 4096]`` and is
  also a pure reshape.

Arg order MUST match the compiled ``host_orch`` signature verbatim (see
``_build_args``).
"""
from __future__ import annotations

import os
import time

import torch

_BF16 = torch.bfloat16
_F32 = torch.float32
_I32 = torch.int32
_I8 = torch.int8
MAIN_PROGRAM = "whole_prefill_step3p5"


def _zsh(*shape, dtype=_BF16):
    """Host tensor for the DistributedWorker contract: share_memory + pre-fork."""
    return torch.zeros(shape, dtype=dtype).share_memory_()


class WholePrefillHolder:
    """Resident whole-net prefill: build + prepare once, ``run()`` reuses ``rt``.

    Typical use::

        h = WholePrefillHolder(device_ids=[0..7], out_dir="/tmp/n1_prefill",
                               ckpt=CKPT)
        h.build()
        with h:
            h.set_prefill_input(embedded_128_tokens)   # [128, HIDDEN] BF16
            res = h.run()                              # {logits, next_hidden}
    """

    def __init__(
        self,
        device_ids,
        out_dir,
        ckpt,
        *,
        platform="a2a3",
        layer_lo=0,
        layer_hi=None,
        golden_topk_dir=None,
    ):
        self.device_ids = list(device_ids)
        self.tp = len(self.device_ids)
        self.dev_offset = self.device_ids[0]
        self.out_dir = out_dir
        self.ckpt = ckpt
        self.program_name = MAIN_PROGRAM
        self.platform = platform
        # Layer range [layer_lo, layer_hi).  The whole-net harness uses the
        # default [0, None=45); a late-MoE slice (e.g. layer_lo=43) is sized
        # via the explicit n_moe_layers derivation in build().  NOTE: a slice
        # still needs the checkpoint MoE weights sliced to the same range
        # (out of scope for the whole-net holder); the flat() reshape fails
        # closed on the element-count mismatch if that slicing is absent.
        self._layer_lo = layer_lo
        self._layer_hi = layer_hi
        self._moe_base = None
        self._n_moe_layers = None
        self.golden_topk_dir = golden_topk_dir

        self.compiled = None
        self._cfg = None
        self._K = None
        self._pf = None
        self._consts = {}

        # populated by __enter__()
        self._prepare_cm = None
        self.rt = None
        self._wmaps = None
        self._args_list = None
        self.current_hidden = None
        self.positions = None
        self.seq_lens = None
        self.slot_mapping = None
        self.block_table = None
        self.gate_r_full = self.gate_r_swa = None
        self.rope_cf = self.rope_sf = self.rope_cs = self.rope_ss = None
        self.k_cache = self.v_cache = None
        self.golden_topk_ids = self.golden_topk_weights = None
        self._next_hidden_out = None
        self._logits_shard_out = None
        self._last_run_sec = 0.0
        self._rope_ready = False

    # ---- build (compile; no device prepare yet) ----------------------------

    def build(self):
        """Compile the program + resolve constants.  No device prepare."""
        from pypto.backend import BackendType, set_backend_type  # noqa: PLC0415
        set_backend_type(BackendType.Ascend910B)

        # Prefill diagnostic ABI env defaults.  The harness may set these
        # explicitly before importing config; setdefault preserves that.
        os.environ.setdefault("PYPTO_STEP3P5_MAX_SEQ", "128")
        os.environ.setdefault("PYPTO_STEP3P5_ROPE_SEQ", "128")
        os.environ.setdefault("PYPTO_STEP3P5_KV_NUM_LAYERS", "45")
        # One 128-token block per layer: 45 layers x 128 rows.
        os.environ.setdefault("PYPTO_STEP3P5_KV_CACHE_ROWS", str(45 * 128))
        os.environ.setdefault("PYPTO_STEP3P5_BLOCK_TABLE_FLAT", "16")

        import models.step3p5.config as cfg  # noqa: PLC0415
        from models.step3p5 import weight_loader as K  # noqa: PLC0415
        self._cfg = cfg
        self._K = K
        assert self.tp == cfg.TP_WORLD_SIZE, (
            f"need {cfg.TP_WORLD_SIZE} cards; got {self.tp}"
        )
        import models.step3p5.prefill_fwd as pf  # noqa: PLC0415
        self._pf = pf

        # Explicit n_moe_layers derivation (design §12.3, mirror of
        # prefill_fwd.py __main__): do NOT hardcode 42 and do NOT rely on the
        # factory's ``n_moe_layers=None -> NUM_MOE_LAYERS`` default.  The MoE
        # weight stacks are indexed by the rebased slot ``moe_pos = li - base``
        # (base = max(layer_lo, NUM_DENSE_LAYERS)) and sized to just the MoE
        # slice [layer_lo, layer_hi) covers.  Whole-net [0,45) -> base=3,
        # n_moe_layers=42 (bit-for-bit unchanged); a late-MoE slice
        # (layer_lo=43) -> base=43, n_moe_layers=2 so the routed-expert stack
        # shrinks ~22.5 -> 0.7 GiB/rank (int8 routed stack) instead of
        # re-allocating the full 42-layer stack (which is what re-triggered
        # the HBM OOM).
        layer_hi = pf.NUM_HIDDEN_LAYERS if self._layer_hi is None else self._layer_hi
        if not (0 <= self._layer_lo < layer_hi <= pf.NUM_HIDDEN_LAYERS):
            raise ValueError(
                f"layer range [{self._layer_lo}, {layer_hi}) must satisfy "
                f"0 <= layer_lo < layer_hi <= {pf.NUM_HIDDEN_LAYERS}"
            )
        moe_base = max(self._layer_lo, pf.NUM_DENSE_LAYERS)
        n_moe_layers = max(1, min(layer_hi, pf.NUM_HIDDEN_LAYERS) - moe_base)
        self._layer_hi = layer_hi
        self._moe_base = moe_base
        self._n_moe_layers = n_moe_layers

        from pypto import ir  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import (  # noqa: PLC0415
            DistributedConfig,
        )
        os.environ.setdefault(
            "PYPTO_PROG_BUILD_DIR",
            os.path.join(self.out_dir, "build_output"),
        )
        _mplan = None
        if os.environ.get("PYPTO_MEM_PLANNER", "").lower() == "ptoas":
            from pypto.pypto_core import passes as _passes  # noqa: PLC0415
            _mplan = _passes.MemoryPlanner.PTOAS
        program = pf._build_prefill_fwd_program(
            self.tp, self._layer_lo, layer_hi, n_moe_layers=n_moe_layers,
        )
        self.compiled = ir.compile(
            program, platform=self.platform,
            distributed_config=DistributedConfig(
                device_ids=self.device_ids, num_sub_workers=0,
            ),
            skip_ptoas=False, dump_passes=False, memory_planner=_mplan,
        )
        print(
            f"[prefill-holder] compile OK => {self.compiled.output_dir} "
            f"(layers=[{self._layer_lo},{layer_hi}) n_moe_layers={n_moe_layers})",
            flush=True,
        )

        self._consts = dict(
            HIDDEN=cfg.HIDDEN,
            HEAD_DIM=cfg.HEAD_DIM,
            BATCH=cfg.BATCH,
            PREFILL_T=pf.PREFILL_T,
            UBD=cfg.USER_BATCH_DYN,
            BTF=cfg.BLOCK_TABLE_FLAT_DYN,
            RSD=cfg.ROPE_SEQ_DYN,
            KVC=cfg.KV_CACHE_ROWS_DYN,
            ROT_FULL=cfg.ROTARY_HALF_FULL * 2,
            ROT_SWA=cfg.ROTARY_HALF_SWA * 2,
            NHF_PAD=cfg.NUM_HEADS_FULL_LOCAL_PAD,
            NHS_PAD=cfg.NUM_HEADS_SWA_LOCAL_PAD,
            NHF=cfg.NUM_HEADS_FULL_LOCAL,
            NHS=cfg.NUM_HEADS_SWA_LOCAL,
            HQ_FULL=cfg.HIDDEN_Q_FULL_LOCAL,
            HQ_SWA=cfg.HIDDEN_Q_SWA_LOCAL,
            N_FULL=pf.NUM_FULL_LAYERS,
            N_SWA=pf.NUM_SWA_LAYERS,
            N_DENSE=pf.NUM_DENSE_LAYERS,
            N_MOE=n_moe_layers,
            LDH_ROWS=cfg.LAYER_HIDDEN_ROWS_DYN,
            LDH_ROWS_SWA=pf.LAYER_HIDDEN_ROWS_DYN_SWA,
            LDD_ROWS=pf.LAYER_HIDDEN_ROWS_DYN_DENSE,
            LDQ_ROWS_FULL=pf.LAYER_QHIDDEN_ROWS_DYN_FULL,
            LDQ_ROWS_SWA=pf.LAYER_QHIDDEN_ROWS_DYN_SWA,
            LDI_ROWS=cfg.LAYER_INTER_ROWS_DYN,
            INTER_LOCAL=cfg.INTERMEDIATE_LOCAL,
            SH_INTER_LOCAL=pf.SH_INTER_LOCAL,
            INTER=pf.INTER,
            N_EXPERTS=pf.N_EXPERTS,
            N_LOCAL_EXPERTS=pf.N_LOCAL_EXPERTS,
            VOCAB_LOCAL=cfg.VOCAB_LOCAL,
        )
        c = self._consts
        if c["KVC"] % (45 * 128) != 0:
            raise ValueError(
                "prefill holder needs KV_CACHE_ROWS_DYN divisible by 45*128, "
                f"got {c['KVC']} (set PYPTO_STEP3P5_KV_CACHE_ROWS=45*128*blocks)"
            )
        if c["RSD"] < c["PREFILL_T"]:
            raise ValueError(
                f"ROPE_SEQ_DYN={c['RSD']} shorter than PREFILL_T={c['PREFILL_T']}"
            )
        return self

    # ---- resident prepare + arg wiring -------------------------------------

    def __enter__(self):
        assert self.compiled is not None, "call build() before entering holder"
        c = self._consts
        tp = self.tp
        hidden, head_dim = c["HIDDEN"], c["HEAD_DIM"]
        pft = c["PREFILL_T"]
        ubd, btf, rsd, kvc = c["UBD"], c["BTF"], c["RSD"], c["KVC"]
        rot_full, rot_swa = c["ROT_FULL"], c["ROT_SWA"]

        # Resident host tensors (allocated BEFORE prepare so forked chips see them).
        self.current_hidden = _zsh(tp, pft, hidden)
        self.positions = (
            torch.arange(pft, dtype=_I32)
            .unsqueeze(0).repeat(tp, 1).contiguous().share_memory_()
        )
        self.seq_lens = torch.full((tp, ubd), pft, dtype=_I32).share_memory_()
        self.slot_mapping = (
            torch.arange(pft, dtype=_I32)
            .unsqueeze(0).repeat(tp, 1).contiguous().share_memory_()
        )
        # Identity paged metadata: block_table[bt_idx] = bt_idx, slot_mapping[t]=t
        # so KV write (cache_row = layer_base + slot) and attention read-back
        # (cache_row0 = layer_base + block_table[bt_idx]*128) address the same
        # rows.  arange (not zeros) is required for multi-block prefill
        # (PREFILL_T > 128): block bt_idx must map to physical block bt_idx.
        self.block_table = (
            torch.arange(btf, dtype=_I32)
            .unsqueeze(0).repeat(tp, 1).contiguous().share_memory_()
        )

        # Block-diagonal gate expanders R (replicated, layer-independent).
        self.gate_r_full = _zsh(tp, c["N_FULL"] * c["NHF_PAD"], c["HQ_FULL"])
        self.gate_r_swa = _zsh(tp, c["N_SWA"] * c["NHS_PAD"], c["HQ_SWA"])
        full_r = self._pf._gate_r_stack(
            c["N_FULL"], c["NHF_PAD"], c["NHF"], c["HQ_FULL"],
        )
        swa_r = self._pf._gate_r_stack(
            c["N_SWA"], c["NHS_PAD"], c["NHS"], c["HQ_SWA"],
        )
        self.gate_r_full[:] = full_r
        self.gate_r_swa[:] = swa_r

        self.rope_cf = _zsh(tp, rsd, rot_full, dtype=_F32)
        self.rope_sf = _zsh(tp, rsd, rot_full, dtype=_F32)
        self.rope_cs = _zsh(tp, rsd, rot_swa, dtype=_F32)
        self.rope_ss = _zsh(tp, rsd, rot_swa, dtype=_F32)
        self.k_cache = _zsh(tp, kvc, head_dim)
        self.v_cache = _zsh(tp, kvc, head_dim)
        self._next_hidden_out = _zsh(tp, pft, hidden)
        self._logits_shard_out = _zsh(tp, pft, c["VOCAB_LOCAL"], dtype=_F32)
        self.golden_topk_ids = _zsh(tp, c["N_MOE"] * pft, self._pf.TOPK, dtype=_I32)
        self.golden_topk_weights = _zsh(tp, c["N_MOE"] * pft, self._pf.TOPK, dtype=_F32)
        self._load_golden_topk()

        # Per-module diagnostic dumps (12 pl.Out params, see prefill_fwd host_orch).
        self._input_norm_dump = _zsh(tp, pft, hidden)
        self._q_proj_dump = _zsh(tp, pft, c["HQ_SWA"], dtype=_F32)
        self._k_proj_dump = _zsh(tp, pft, head_dim, dtype=_F32)
        self._v_proj_dump = _zsh(tp, pft, head_dim, dtype=_F32)
        self._v_tile_dump = _zsh(tp, pft, head_dim)
        self._q_norm_dump = _zsh(tp, pft, c["HQ_SWA"], dtype=_F32)
        self._k_norm_dump = _zsh(tp, pft, head_dim, dtype=_F32)
        self._gate_logits_dump = _zsh(tp, pft, c["NHF_PAD"])
        self._resid1_dump = _zsh(tp, pft, hidden)
        self._attn_delta_dump = _zsh(tp, pft, hidden)
        self._post_norm_dump = _zsh(tp, pft, hidden)
        self._ffn_output_dump = _zsh(tp, pft, hidden)
        # Attention-core dumps (produced by layer 44 full_moe_swiglu7 only):
        # q_rot/k_rot/attn_out/attn_out_gated/o_proj_local + full-column scores.
        self._q_rot_dump = _zsh(tp, pft, c["HQ_FULL"])
        self._k_rot_dump = _zsh(tp, pft, head_dim)
        self._attn_out_dump = _zsh(tp, pft, c["HQ_FULL"])
        self._attn_out_gated_dump = _zsh(tp, pft, c["HQ_FULL"])
        self._o_proj_local_dump = _zsh(tp, pft, hidden)
        scores_pad_cols = ((pft + 127) // 128) * 128
        self._scores_dump = _zsh(tp, pft * 16, scores_pad_cols, dtype=_F32)

        from tools.step3p5.pypto_weight_ipc import (  # noqa: PLC0415
            build_stacked_weight,
            import_weights_all,
        )
        K = self._K

        self._prepare_cm = self.compiled.prepare(persistent=True)
        self.rt = self._prepare_cm.__enter__()
        self._wmaps = import_weights_all(
            self.rt, self.out_dir, tp=tp, dev_offset=self.dev_offset,
        )

        self._initialize_rope_tables()

        def W(key):
            return build_stacked_weight(self._wmaps, key)

        def flat(key, tail):
            """Flatten a decode per-layer 3D/4D stack to the prefill 2D ABI."""
            stacked = W(key)
            return self._flatten_weight(stacked, tail)

        def moe(key, tail):
            """Slice a layer-major MoE stack to the [0, n) layers, then flatten.

            The MoE weight stacks are the ONLY whole_chip_orch params declared
            against ``n_moe_layers`` (prefill_fwd.py whole_chip_orch), so a
            restricted layer range must shrink the checkpoint's 42-layer
            layer-major stack to the SAME layers before the flat() reshape.
            The exporter pre-slices the MoE stacks to the active
            ``[moe_base, layer_hi)`` range (see ``weight_loader.moe_slice_params``),
            so the imported stack already starts at slot 0 — the slice here is
            ``s[0:n]`` (a no-op view on a correctly-sliced pool, but fail-closed
            via the element-count check if the exporter exported the full stack).
            """
            from pypto.runtime.device_tensor import (  # noqa: PLC0415
                StackedDeviceTensor,
            )
            stacked = W(key)
            n = self._n_moe_layers
            shards = tuple(s[0:n] for s in stacked.shards)
            sliced = StackedDeviceTensor(
                shards, (len(shards), n, *stacked.full_shape[2:]),
                stacked.worker_ids,
            )
            return self._flatten_weight(sliced, tail)

        args = self._build_args(W, flat, moe)
        self._args_list = args
        print(
            f"[prefill-holder] resident: built {len(args)} args; "
            f"program={self.program_name}",
            flush=True,
        )
        return self

    def _load_golden_topk(self):
        """Load golden per-token MoE routing (topk_ids/topk_weights) for the
        active MoE layer slice, padded to PREFILL_T and broadcast to all ranks.

        The golden dump holds RAW_T(=331) real tokens; rows RAW_T..PREFILL_T
        are zero-filled (topk_ids=0 is a valid expert, topk_weights=0 so the
        padding rows contribute nothing to the weighted combine).  File name
        maps the rebased MoE slot ``moe_base + i`` back to the absolute layer
        id (dump files are named by absolute layer).
        """
        if not self.golden_topk_dir:
            return
        import glob

        c = self._consts
        pft = c["PREFILL_T"]
        topk = self._pf.TOPK
        ids = self.golden_topk_ids
        wts = self.golden_topk_weights
        n = self._n_moe_layers
        base = self._moe_base
        ids_buf = torch.zeros(pft, topk, dtype=_I32)
        wts_buf = torch.zeros(pft, topk, dtype=_F32)
        for i in range(n):
            layer = base + i
            path = f"{self.golden_topk_dir}/1_rank0_layer_{layer:02d}_moe_router.pt"
            if not glob.glob(path):
                raise FileNotFoundError(
                    f"missing golden topk dump for MoE layer {layer}: {path}"
                )
            obj = torch.load(path, map_location="cpu", weights_only=True)
            ids_buf[:obj["topk_ids"].shape[0]] = obj["topk_ids"].to(_I32)
            wts_buf[:obj["topk_weights"].shape[0]] = obj["topk_weights"].to(_F32)
            ids[:, i * pft:(i + 1) * pft] = ids_buf
            wts[:, i * pft:(i + 1) * pft] = wts_buf
        print(
            f"[prefill-holder] golden topk loaded: n_moe={n} base={base} "
            f"pft={pft} topk={topk}",
            flush=True,
        )

    def _flatten_weight(self, stacked, new_tail):
        """Re-view each rank shard of a stacked weight to ``new_tail`` (no copy)."""
        from pypto.runtime.device_tensor import (  # noqa: PLC0415
            StackedDeviceTensor,
        )
        shards = tuple(s.reshape(new_tail) for s in stacked.shards)
        return StackedDeviceTensor(
            shards, (len(shards), *new_tail), stacked.worker_ids,
        )

    def _build_args(self, W, flat, moe):
        """Build the 62-arg ``host_orch`` list in exact signature order."""
        K = self._K
        c = self._consts
        hidden, head_dim = c["HIDDEN"], c["HEAD_DIM"]
        args = [self.current_hidden]
        # Norms (already FP32 in the exported IPC pool).
        args += [
            W(K.KEY_INPUT_RMS), W(K.KEY_POST_ATTN_RMS),
            W(K.KEY_Q_NORM), W(K.KEY_K_NORM),
        ]
        # Full-attention weights + gate R.
        args += [
            flat(K.KEY_WQ_FULL, (c["LDH_ROWS"], c["HQ_FULL"])),
            flat(K.KEY_WK_FULL, (c["LDH_ROWS"], c["HEAD_DIM"])),
            flat(K.KEY_WV_FULL, (c["LDH_ROWS"], c["HEAD_DIM"])),
            flat(K.KEY_WO_FULL, (c["LDQ_ROWS_FULL"], hidden)),
            flat(K.KEY_WG_FULL, (c["LDH_ROWS"], c["NHF_PAD"])),
            self.gate_r_full,
        ]
        # SWA attention weights + gate R.
        args += [
            flat(K.KEY_WQ_SWA, (c["LDH_ROWS_SWA"], c["HQ_SWA"])),
            flat(K.KEY_WK_SWA, (c["LDH_ROWS_SWA"], c["HEAD_DIM"])),
            flat(K.KEY_WV_SWA, (c["LDH_ROWS_SWA"], c["HEAD_DIM"])),
            flat(K.KEY_WO_SWA, (c["LDQ_ROWS_SWA"], hidden)),
            flat(K.KEY_WG_SWA, (c["LDH_ROWS_SWA"], c["NHS_PAD"])),
            self.gate_r_swa,
        ]
        # Dense MLP (gate/up/down: pure reshape of the 3 dense layers).
        args += [
            flat(K.KEY_DENSE_GATE, (c["LDD_ROWS"], c["INTER_LOCAL"])),
            flat(K.KEY_DENSE_UP, (c["LDD_ROWS"], c["INTER_LOCAL"])),
            flat(K.KEY_DENSE_DOWN, (c["LDI_ROWS"], hidden)),
        ]
        # MoE gate/router + routed BF16 experts + shared experts.
        # These are the only stacks sized by n_moe_layers, so they are sliced
        # to the active MoE layer range before the flat() reshape (moe()).
        args += [
            moe(K.KEY_MOE_GATE_W, (c["N_MOE"] * hidden, c["N_EXPERTS"])),
            moe(K.KEY_MOE_ROUTER_BIAS, (c["N_MOE"] * c["N_EXPERTS"],)),
            moe(K.KEY_MOE_W_GATE_R,
                (c["N_MOE"] * c["N_LOCAL_EXPERTS"] * hidden, c["INTER"])),
            moe(K.KEY_MOE_W_GATE_R_SCALE,
                (c["N_MOE"] * c["N_LOCAL_EXPERTS"], c["INTER"])),
            moe(K.KEY_MOE_W_UP_R,
                (c["N_MOE"] * c["N_LOCAL_EXPERTS"] * hidden, c["INTER"])),
            moe(K.KEY_MOE_W_UP_R_SCALE,
                (c["N_MOE"] * c["N_LOCAL_EXPERTS"], c["INTER"])),
            moe(K.KEY_MOE_W_DOWN_R,
                (c["N_MOE"] * c["N_LOCAL_EXPERTS"] * c["INTER"], hidden)),
            moe(K.KEY_MOE_W_DOWN_R_SCALE,
                (c["N_MOE"] * c["N_LOCAL_EXPERTS"], hidden)),
            moe(K.KEY_MOE_W_GATE_S, (c["N_MOE"] * hidden, c["SH_INTER_LOCAL"])),
            moe(K.KEY_MOE_W_UP_S, (c["N_MOE"] * hidden, c["SH_INTER_LOCAL"])),
            moe(K.KEY_MOE_W_DOWN_S, (c["N_MOE"] * c["SH_INTER_LOCAL"], hidden)),
        ]
        # Paged-KV / RoPE metadata.
        args += [
            self.block_table, self.slot_mapping,
            self.rope_cf, self.rope_sf, self.rope_cs, self.rope_ss,
        ]
        # InOut KV cache (empty, Gap C).
        args += [self.k_cache, self.v_cache]
        # Positions.
        args += [self.positions]
        # Outputs.
        args += [self._next_hidden_out]
        # Tail weights (final norm needs [1, HIDDEN] view; lm head unchanged).
        args += [
            flat(K.KEY_FINAL_NORM, (1, hidden)),
            W(K.KEY_LM_HEAD),
        ]
        # Dynamic batch metadata + logits out.
        args += [self.seq_lens, self._logits_shard_out]
        # Per-module diagnostic dumps (18 pl.Out params, exact host_orch order).
        args += [
            self._input_norm_dump, self._q_proj_dump, self._k_proj_dump,
            self._v_proj_dump, self._v_tile_dump, self._q_norm_dump,
            self._k_norm_dump,
            self._gate_logits_dump, self._resid1_dump, self._attn_delta_dump,
            self._post_norm_dump, self._ffn_output_dump,
            self._q_rot_dump, self._k_rot_dump, self._attn_out_dump,
            self._attn_out_gated_dump, self._o_proj_local_dump,
            self._scores_dump,
        ]
        return args

    def __exit__(self, exc_type, exc, tb):
        if self._prepare_cm is not None:
            self._prepare_cm.__exit__(exc_type, exc, tb)
            self._prepare_cm = None
            self.rt = None

    # ---- per-step input + rope ---------------------------------------------

    def _initialize_rope_tables(self):
        if self._rope_ready:
            return
        from models.step3p5._ops import (  # noqa: PLC0415
            build_llama3_yarn_rope_tables,
            build_plain_rope_tables,
        )
        cfg = self._cfg
        c = self._consts
        rsd = c["RSD"]
        full_cos, full_sin = build_llama3_yarn_rope_tables(
            rsd,
            cfg.ROTARY_HALF_FULL * 2,
            cfg.LAYER_ROPE_THETA[0],
            factor=cfg.ROPE_SCALING["factor"],
            low=cfg.ROPE_SCALING["low_freq_factor"],
            high=cfg.ROPE_SCALING["high_freq_factor"],
            orig_max=cfg.ROPE_SCALING["original_max_position_embeddings"],
        )
        swa_cos, swa_sin = build_plain_rope_tables(
            rsd,
            cfg.ROTARY_HALF_SWA * 2,
            cfg.LAYER_ROPE_THETA[1],
        )
        for rank in range(self.tp):
            self.rope_cf[rank].copy_(full_cos)
            self.rope_sf[rank].copy_(full_sin)
            self.rope_cs[rank].copy_(swa_cos)
            self.rope_ss[rank].copy_(swa_sin)
        self._rope_ready = True

    def set_prefill_input(self, hidden):
        """Feed one 128-token prefill sequence (replicated across TP ranks).

        ``hidden`` must be BF16 ``[PREFILL_T, HIDDEN]``.  Positions / seq_lens /
        slot_mapping / block_table are fixed for a single 128-token block and
        were already initialized in ``__enter__``; only the hidden content and
        the KV cache are per-invocation.
        """
        c = self._consts
        expected = (c["PREFILL_T"], c["HIDDEN"])
        if hidden.dtype != _BF16 or tuple(hidden.shape) != expected:
            raise ValueError(
                f"prefill hidden must be BF16 {expected}, got "
                f"{hidden.dtype} {tuple(hidden.shape)}"
            )
        self.current_hidden.zero_()
        self.current_hidden[:] = hidden.to(_BF16)
        self.k_cache.zero_()
        self.v_cache.zero_()

    # ---- run ----------------------------------------------------------------

    def run(self):
        """Run one whole-net prefill forward; return per-position logits + hidden."""
        assert self.rt is not None, "enter holder (with h:) before run()"
        t0 = time.time()
        _dfx = os.environ.get("N1_DFX", "")
        if _dfx:
            from pypto.runtime.runner import RunConfig  # noqa: PLC0415
            _pmu = int(os.environ.get("N1_PMU", "1")) if "pmu" in _dfx else 0
            _rc = RunConfig(
                platform=self.platform,
                enable_dep_gen=("dep" in _dfx),
                enable_scope_stats=("scope" in _dfx),
                enable_l2_swimlane=("swim" in _dfx or "l2" in _dfx),
                enable_pmu=_pmu,
                enable_dump_args=(1 if "dump" in _dfx else 0),
            )
            self.rt.run(self.compiled, *self._args_list, config=_rc)
        else:
            self.rt.run(self.compiled, *self._args_list)
        self._last_run_sec = time.time() - t0
        return {
            "logits": self._logits_shard_out,
            "next_hidden": self._next_hidden_out,
        }


__all__ = ["WholePrefillHolder"]
