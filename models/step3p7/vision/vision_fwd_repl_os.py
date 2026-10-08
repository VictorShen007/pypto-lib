# Copyright (c) PyPTO Contributors. SPDX-License-Identifier: Apache-2.0
# Step3.7 vision REPL tower — ONLINE-SOFTMAX single-pass QK attention draft.
#
# M-DECOUPLING (promoted, device-verified): the four FixPipe GEMMs'
# M-tile decoupled from the elementwise passes. GEMM M 48/80 -> QTM_G/QTM1_G=128
# (Acc [128,256] fp32 = 128KB = 100% L0C cap — the ceiling; M=160 would blow it;
# Left [128,64]x2Bx2 = 32KB = 50%). Elementwise passes stay QTM=48/QTM1=80 (the
# Vec wall: v1b [80,256]=87.8%, v3e/v4e [48,256]=94.6%). nqg_G 57->22, nqg1_G
# 34->22 blocks (last block [2688,2816)=16 real+112 pad, 4.0% pad-FLOP).
# Same-session A/B (8x910B2C, device_wall fallback_flattened): 365.7->318.6ms =
# -12.9%. L0 tight (5e-3/5e-3/2%, layers 0+46) + full-tower loose (1.2e-1/8e-2/1%)
# PASS.
#
# N256 LADDER (promoted, device-verified): all four GEMM
# N-chunks go 128 -> FP_N=256 with the DERIVED K-chunk FP_KC=64
# (Right/L0B = [64,256]x2Bx2stage = 64KB cap-exact; the 128B-A-row L2-line
# rule holds). v3g's K-chunk HDP=128 also drops to 64 (32 half-head chunks,
# per-column FMA order unchanged). The four elementwise passes (v1b/v3e/v4e/
# v4ae) chunk at RES_N=256 — v1b tiles are [80,256] (161.5KB=87.8% Vec),
# v3e/v4e [48,256] (174.0KB=94.6%, the tightest). Same-session interleaved
# 5-run A/B (8x910B, harness device_wall fallback_flattened): 365.3ms vs
# production donor 392.2ms = -6.8% (donor reproduced its 392.07 standing;
# no host drift, no interleave trend). L0 tight gates (5e-3/5e-3/2%,
# layers 0+46) + full-tower loose PASS.
#
# MLP_F1 SPLIT (promoted, device-verified): on top of the FixPipe
# split, mlp_f1 (v4a, the LAST fused GEMM: bias + gelu sigmoid epilogue)
# splits into mlp_f1_gemm_fixpipe (pure cube, direct store to p1buf) +
# mlp_f1_gelu_fixpipe (Vec pass: bias + gelu + cast). N-chunk QKN1 64 ->
# FPN 128, K-chunk QN2 256 -> KQN 128 (Right [128,128]x2Bx2stage = 64KB
# cap-exact). FixPipe pre-rounds acc to BF16 BEFORE the bias add and the
# gelu nonlinearity — L0 tight gates (5e-3/5e-3/2%, layers 0 + 46) and the
# full-tower loose gate all PASS on device, so the delta is gate-safe.
# Same-session A/B (8x910B, harness device_wall fallback_flattened):
# 392.07ms vs production-donor 419.96ms = -6.6% (the p1buf GM round-trip
# tax did not eat the N-halving win).
#
# FIXPIPE SPLIT (promoted, device-verified): the patch-tower
# FixPipe recipe ported to the os tower. v3r/v4r (fused residual epilogues)
# and v1g (fused bias) split into pure-cube FixPipe direct-store GEMMs
# (v3g/v4g/v1g: acc -> GM via PIPE_FIX, zero Vec) plus elementwise passes
# (v3e/v4e resid, v1b bias). N-chunk 64 -> FPN=128 everywhere (v3 K-chunk
# HDP=128 unchanged; v4/v1g K-chunk 256 -> 128 so Right [128,128]x2Bx2stage
# = 64KB cap-exact). Numeric delta vs the fused forms: FixPipe pre-rounds
# acc to BF16 BEFORE the bias/scale adds (~1 ULP) — L0 tight gates
# (5e-3/5e-3/2%, layers 0 + 46) and the full-tower loose gate
# (1.2e-1/8e-2/1%) all PASS on device. mlp_f1 (gelu) stays fused.
# Same-session A/B (8x910B, harness device_wall fallback_flattened):
# 423.10ms vs production-donor 473.79ms = -10.7% (matches the patch-tower
# -10.6% landing).
#
# Committed GEMM chunk geometry: same math as the original
# vision_fwd_repl_os (per-output-column K order unchanged -> bit-identical),
# with GEMM N-chunking widened where the 64KB L0B cap allows (M=QT=16 regime is
# Vec-safe — acc [16,256] fp32 = 16KB/tile, far under the 188KB wall):
#   qkv      QKN 64 -> 128   (wt [256,128] = 64KB, at cap; 36 chunks)
#   mlp_f2   N 128 -> 256    (wt [128,256] = 64KB, at cap; K-chunk stays QN=128)
#   vb3      N 128 -> 256    (wt [128,256] = 64KB, at cap; K-chunk stays HDP=128)
#   mlp_f1   N 128 -> 256    (wt [128,256] = 64KB, at cap) via K-chunk 256 -> 128
#            (x rows [QT,128] = 256B = one cache line, same as the committed
#            mlp_f2 K-chunk; A-side per-column K order unchanged -> bit-identical)
# Bench (8x910B, ROUNDS=5/WARMUP=2): 63.8 -> 55.5 ms mean (-13.0%), L3 PASS.
# Chunked MQA=208 (below): 55.5 -> 51.3 ms (-7.7%), L3 PASS (promoted).
# v1g M-tile rework: QKV [16,128] -> [48,64] (patch-production
# geometry; vT=2704=48*56+16 needs a [16,64] tail block). See qkv_gemm_fixpipe below.
#
# Phase 1 of the DSV4-style attention rework: vb2_all's v2g
# (global row-max sweep) + v2p (QK recompute + exp) are merged into ONE
# v2o pass that computes per-KV-block LOCAL row max / local exp / local row
# sum, materializing pe_s with local-max scaling plus per-block stats
# (mb_s = local max, lb_s = local sum). The v2f PV pass then rescales each
# block's matmul OUTPUT by exp(m_kb - mi) (cube output scaled in the vector
# domain — safe direction, A operands stay pure GM slices) and sums the
# corrected denominators. Mathematically exact softmax: pe_kb = exp(s-m_kb),
# contribution = exp(m_kb-mi) * pe_kb @ V = exp(s-mi) @ V.
#
# vs the three-phase form: QK matmul sweeps go 2 -> 1 (attention cube flops
# 320 -> 224 units per head-row, -30%), pe_s GM layout unchanged (kept for
# the clean GM->cube A operand; full pe_s elimination is Phase 2).
#
# Everything else (v4 split LN/GEMM bodies, vb3, residual adds, chunked
# host_orch) is copied verbatim from the retired three-phase repl tower. The attention body
# is defined LOCALLY in this module (not shared via vision_fwd.vb2_all) so
# its free variables resolve against this module's full-dim globals without
# ambiguity; KT here is the flash K-tile (208), unlike the donor module's
# vestigial local KT=16.
#
# Chunked-attention MQA=208 (PROMOTED from a probe):
# the literal QT->104 flip fails the vec UB wall (v2o/v2f materialize [M,208]
# fp32 softmax intermediates; M=16 already ~50% of the 188KB wall). Instead,
# each v2o/v2f lane now covers MQA=208 query rows as RQA=13 chunks of CQA=16
# rows. The kt/vt tiles are loaded ONCE per (lane,kb) and shared across the
# RQA chunk matmuls (probe-verified: the AIC kernel hoists the kt TLOAD and
# reuses it via TMOV; the AIV softmax tiles stay [16,208] fp32 = current size).
# KV re-read (kr 1.4GB + vp 1.9GB/layer at M=16) drops ~13x; flash semantics
# are per-query-row exact (each row's softmax over the full 2704 keys, chunk
# grouping is only a lane assignment) -> bit-identical to the M=16 form.
# mb_s/lb_s are per-row [sl_a*nk, MQA] (v2o writes each chunk's [1,CQA] stats
# via GM column-offset stores at [fa*nk+kb, sub*CQA]; v2f reduces mi/acc_l as
# whole [1,MQA] tiles); pe_s [OI,vT] unchanged
# (OI=vT*HL is M-invariant). v2r stays at tower-wide QT=16 (no KV re-read there).
# Tower-wide QT=16 is UNTOUCHED (GEMM/vec bodies need it for the 188KB wall).
#
# Correctness note (root cause): pl.create_tensor GM scratch is NOT
# zero-initialized, and the compiler caches the RMW accumulator UB-resident, so
# the first TADD read stale UB -> 99.3% error. A pl.full([MQA,64],0.0) seed is
# REJECTED at compile (its TEXPANDS/vector_dup lands in the CUBE kernel where
# vector fills are illegal). Fix: kb=0 is a pure WRITE pass (po assembled
# directly into acc_o0/acc_o1, no read of uninit), kb=1..12 RMW; per-row
# accumulation order over kb is unchanged -> still bit-identical to M=16.
import os
import pypto.language as pl; import pypto.language.distributed as pld
from .vision_config import (V_LAYERS,V_TOKENS,V_GLOBAL_BATCH,V_WIDTH,V_EPS,V_WIDTH_INV,V_ATTN_SCALE,V_HEAD_DIM_PAD,V_HEADS,V_MLP_HIDDEN,V_MLP_HIDDEN_PAD,V_WIDTH_PAD)

# ── FULL-dim constants (same as the retired three-phase repl tower) ──
D=V_WIDTH;DT=128;QT=16;QN=128;QKN=128;QKK=256;RQT=8;QN3=256
DL=V_WIDTH;HD=96;HL=V_HEADS;IL=V_MLP_HIDDEN;ILP=V_MLP_HIDDEN_PAD;_QG=1.702
HDP=V_HEAD_DIM_PAD;DLP=V_WIDTH_PAD;KO=3*DL
KT=208;NQM=V_TOKENS//QT;SL=NQM*HL;OI=SL*QT;HA=HD//2  # KT=208: flash K-tile (vT=2704=13*208), USED by vb2_all_os
CQA=16;RQA=13;MQA=CQA*RQA  # chunked attention: CQA-row softmax chunk, RQA chunks/lane, MQA=M_eff=208 (vT=2704=13*MQA); KV re-read ~1/MQA
SL_A=(V_TOKENS//MQA)*HL  # sl_a = nqa*HL = 208 — module CONSTANT, not a kernel-local:
# the B-axis guard's `fa//SL_A` lane->image decode needs a compile-time value;
# a kernel-local sl_a would become a runtime variable and its shape expression
# (mb_s rows = VB*SL_A*nk) would not be invertible for wrapper codegen.
vT=V_TOKENS;VB=V_GLOBAL_BATCH;vTe=VB*vT;vWQ=V_LAYERS*V_WIDTH;vWF1=V_LAYERS*V_WIDTH;vWF2=V_LAYERS*ILP
qkn=KO//QKN;qkk=D//QKK;dn=D//DT;on=D//QN;inn=ILP//QN;QN2=256;on2=D//QN2;on3=ILP//QN3;KQN=128;qkk2=D//KQN
QTM=48;QKN1=64;nqg=(vT+QTM-1)//QTM;qkng=KO//QKN1;nn1=ILP//QKN1;dn1=D//QKN1  # v4a/v4b/v3 M/N tiles (M-tile lever): donor [QT,QKN]=[16,128] ran ~25-29 TFLOPS (each [QKK,N] weight chunk amortized over only 16 rows); patch's production [48,64] measures 50 TFLOPS on the same silicon. vT=2704=48*56+16, so a uniform M is impossible: ptoas alloc_tile REQUIRES M%16==0 (M=52 rejected: "tile rows to be a multiple of innerRows (16)"), and 2704=16*13^2 has no usable 16-aligned divisor besides 16. An `if`-guarded [16,·] tail INSIDE the spmd is also out: its IfStmt merge breaks the return lineage of the loop-carried Out param (NormalizeReturnOrder cannot canonicalize `return result` -> orchestration codegen refuses the 2-Out-param alias). Solution = STRIDE-PAD: per-image stride >= vT with the last block's extra rows never-read pad. [48,128] fp32 tiles (~11 live = 264KB) would blow the 188,416B Vec wall; [48,64] = ~132KB fits (patch also measured N=32 killing cube efficiency 176.3->206.2ms). QKN/qkn above stay for reference/probes.
QTM1=80;nqg1=(vT+QTM1-1)//QTM1  # v1g M-tile third wave (vLLM-gap attack): QKV alone goes 48 -> 96. v1g_aic report: Right 100% ([256,64]x2 double-buffered = the 64KB L0B cap — N-widening is pinned for BF16), Left 75%, Acc 9.4% — the open axis is M. M=96: A-chunk [96,256] = 48KB single-buffered Left (the K loop drops pl.pipeline -> plain range: stage=2 double-buffering would need 96KB > 64KB Left cap), Right [256,64] = 32KB single-buffered, Acc [96,64] fp32 = 24KB. wq re-read per layer drops 57 -> 29 blocks = 807 -> 410MB (-49%); vLLM/CANN runs M=128-256 tiles on the same silicon at 89 TFLOPS vs our 37. K order per column UNCHANGED (same 6 serial [.,256] chunks) -> bit-identical expected. nqg1 blocks cover [0,2784); the v4x phases keep [48,64] blocks covering [0,2736) — both <= vTp, so the SHARED per-image stride vTp = max(nqg1*QTM1, nqg*QTM) = 2784 works: each phase tiles its own prefix of the image's stride region, rows above a phase's coverage are that phase's never-read pad. vT=2704=80*33+64 -> 34 blocks, 34th covers 64 real + 16 pad rows (0.6% pad FLOPs); aiv Vec scales linearly with M (M=48->108.4KB, M=96 probe->216.4KB FAIL), so M=80 ~180.7KB is the edge of the wall; aic side verified open at M=96 already.
QTM_G=128;QTM1_G=128  # GEMM M-chunk PINNED at 128 (promoted, M-decoupling): cube GEMM M decoupled from the elementwise M. Cube A=[M,FP_KC=64], Acc=[M,256] fp32 = 128KB = 100% L0C cap (M=160 would blow); Left [128,64]x2Bx2 = 32KB = 50%. Elementwise passes (v1b/v3e/v4ae/v4e) keep QTM1=80/QTM=48 (Vec wall).
nqg_G=(vT+QTM_G-1)//QTM_G;nqg1_G=(vT+QTM1_G-1)//QTM1_G
vTp=max(nqg1*QTM1,nqg*QTM,nqg1_G*QTM1_G,nqg_G*QTM_G);vTpE=VB*vTp  # shared per-image stride = max of the phases' coverages (at QTM1=80: max(2720,2736)=2736; at 96: 2784) — a phase's tiling must NEVER exceed the stride or its last block would spill into image b+1's rows
FP_N=256;FP_KC=64  # N-chunk PINNED at 256 (promoted after same-session device A/B: 365.3ms vs the 392.2ms donor = -6.8%); Right=[FP_KC,FP_N]x2Bx2stage = 64KB cap-exact; FP_KC=64 means 128B A rows = exactly the L2-line rule
RES_N=FP_N if FP_N<=256 else 256  # elementwise-pass N-chunk: FP32 tile sets must stay off the 184KB Vec wall (v1b tiles are [80,RES_N] — os-only M=80)
on_fpn=D//FP_N;on_fpn1=KO//FP_N;on_fpnm=ILP//FP_N  # GEMM N-chunk counts (v3g/v4g: 12 at 128 -> 6 at 256; v1g: 36 -> 18; v4ag: 70 -> 35)
v1g_kk=D//FP_KC;v3g_kk=DLP//FP_KC;v4g_kk=ILP//FP_KC;on2f=D//FP_KC  # GEMM K-chunk counts (v1g: 12 -> 24; v3g: 16=HL at 128 -> 32 at 64; v4g: 70 -> 140; v4ag: 12 -> 24); per-column FMA order unchanged
on_res=D//RES_N;on_res1=KO//RES_N;on_resm=ILP//RES_N  # elementwise-pass chunk counts
# M-tile promotion of v4a/v4b/v3 (second stride-pad wave): same [QTM,QKN1]=[48,64] geometry, K chunks unchanged (v4a QN=128 / v4b QN=128 / v3 HDP=128 -> per-column K order unchanged -> bit-identity expected, the N-widening + MQA=208 M-regroup + v1g probe all kept it). spmd 169 -> 57 blocks/image. The A-operand buffers feeding these GEMMs and their Out buffers get the SAME per-image vTp stride as lnbuf/qf: gbuf [vTpE,ILP] (v4a out -> v4b in), attn_ctx [vTpE,DLP] (v2f out -> v3 in, v2f write base b*vTp). Pad-row confinement is the v1g argument: a GEMM output row depends only on the SAME input row, so garbage pad rows ([2704,2736) per image — stale never-written scratch for attn_ctx/lnbuf, propagated garbage for gbuf) can only reach never-read pad rows downstream; qr/kr/vp keep the vT stride. r1/r2 stay M=16 (61µs phases, not the wall).
# Fused residual epilogues (Phase-A retry at [48,64]): vb3r1_fused / mlp_f2_r2_fused eliminated pbuf/p2buf entirely, so rbb/x1buf ALSO moved to the vTp stride (block 57 must write pad rows, not image b+1's real rows) — ln_fwd reads x at b*vTp (real rows only, tg<169), layer-0's rbb comes from a one-shot cp0 copy of x_in, and wrto reads rbb at b*vTp / writes rto at b*vT. The FixPipe split restores pbuf/p2buf (now the direct-store targets) but KEEPS the rbb/x1buf vTp stride: the resid passes read/write at b*vTp. The epilogue reproduces the pbuf/p2buf values via an on-chip BF16 rint round-trip with the donor r1/r2 add nesting -> bit-identical on all real rows.

# ── ONLINE-SOFTMAX attention body (Phase 1) ──
# v2r RoPE pass is inlined verbatim from vision_fwd.vb2_all (proven; kept in
# the same body — no jit-inline helper calls across the pl.inline splice).
@pl.jit.inline
def vb2_all_os(qkv_full:pl.Tensor[[vTpE,KO],pl.BF16],cos:pl.Tensor[[vT,HD],pl.FP32],sin:pl.Tensor[[vT,HD],pl.FP32],attn_ctx:pl.Tensor[[vTpE,DLP],pl.BF16],active:pl.Scalar[pl.INDEX]):
    nk=vT//KT;nqa=vT//MQA
    qr=pl.create_tensor([vTe,DL],dtype=pl.BF16);kr=pl.create_tensor([vTe,DL],dtype=pl.BF16);vp=pl.create_tensor([vTe,DLP],dtype=pl.BF16)
    pe_s=pl.create_tensor([VB*OI,vT],dtype=pl.BF16);mb_s=pl.create_tensor([VB*SL_A*nk,MQA],dtype=pl.FP32);lb_s=pl.create_tensor([VB*SL_A*nk,MQA],dtype=pl.FP32)
    # v2r: RoPE (verbatim math from vision_fwd.vb2_all, incl. the BUG-0002 zero pad).
    # crop-block merge: spmd((vT//QT)*HL) + serial inner b loop
    # (was spmd(VB*(vT//QT)*HL) with lane-decoded b) — 1 submit either way, but
    # block_num drops VB* (at active=1 the old form drained 18928 idle blocks/
    # layer through the AIV block scheduler). RoPE index constants are b- and
    # h-independent and hoisted out of both loops.
    # head-fold (swimlane-driven: v2r was 2704 blocks/layer at 16.5µs
    # with a ~24KB working set — latency-dominated): spmd(vT//QT) + serial inner
    # h,b loops. block_num 2704->169/layer; the [QT,HD] cos/sin tiles are sliced
    # ONCE per row-chunk lane and row-sliced from that resident tile (the old
    # form made each of the HL head lanes re-read the same cos/sin rows from GM).
    # Same (b,row,head) op sequence -> bit-identical output.
    # MEASURED (8x910B2C, plain+resident steady state): 637.2/638.2ms
    # vs donor 649.3/651.7/656.1ms = -1.8..-2.7% — small but reproducible win,
    # KEPT. v2r sits on the AIV critical path only partially (mostly overlapped),
    # so the full 42->~20ms v2r saving does not reach the wall. Numerics:
    # bit-identical at active=1 (0/33,226,752) and active=2 (0/66,453,504,
    # both crops), double golden gates PASS.
    # PITFALL (recorded so it is not re-learned): the "persistent" bench mode
    # name does NOT imply resident weights (persistent=True, resident=False
    # re-uploads ~2.7GB/card every dispatch and reads ~810ms).
    for rc in pl.spmd(vT//QT,name_hint='v2r'):
        pos0=rc*QT
        cts=pl.slice(cos,[QT,HD],[pos0,0]);sts=pl.slice(sin,[QT,HD],[pos0,0])
        v2o=pl.full([QT,HD],dtype=pl.FP32,value=1.0)
        v2col=pl.col_expand_mul(v2o,pl.cast(pl.arange(0,[1,HD],dtype=pl.INT32),target_type=pl.FP32))
        v2dup=pl.cast(pl.cast(pl.mul(v2col,0.5),target_type=pl.INT32,mode="trunc"),target_type=pl.FP32)
        v2lane=pl.sub(v2col,pl.mul(v2dup,2.0))
        v2swap=pl.cast(pl.sub(pl.add(v2col,1.0),pl.mul(v2lane,2.0)),target_type=pl.INT32)
        for h in pl.range(HL):
            h0=h*HD;hv0=h*HDP
            for b in pl.range(VB):
                if b<active:
                    qg=b*vT+pos0;qgf=b*vTp+pos0  # qgf: qkv_full row base (vTp stride); qr/kr/vp writes keep the vT-stride qg
                    v2qf=pl.cast(pl.slice(qkv_full,[QT,HD],[qgf,h0]),target_type=pl.FP32)
                    v2qs=pl.gather(v2qf,dim=-1,index=v2swap)
                    v2kf=pl.cast(pl.slice(qkv_full,[QT,HD],[qgf,DL+h0]),target_type=pl.FP32)
                    v2ks=pl.gather(v2kf,dim=-1,index=v2swap)
                    for qi in pl.range(QT):
                        ct1=pl.slice(cts,[1,HD],[qi,0]);st1=pl.slice(sts,[1,HD],[qi,0])
                        q1=pl.slice(v2qf,[1,HD],[qi,0]);qs1=pl.slice(v2qs,[1,HD],[qi,0])
                        qr=pl.assemble(qr,pl.cast(pl.add(pl.col_expand_mul(q1,ct1),pl.col_expand_mul(qs1,st1)),target_type=pl.BF16,mode='rint'),[qg+qi,h0])
                        k1=pl.slice(v2kf,[1,HD],[qi,0]);ks1=pl.slice(v2ks,[1,HD],[qi,0])
                        kr=pl.assemble(kr,pl.cast(pl.add(pl.col_expand_mul(k1,ct1),pl.col_expand_mul(ks1,st1)),target_type=pl.BF16,mode='rint'),[qg+qi,h0])
                    vr=pl.slice(qkv_full,[QT,HD],[qgf,2*DL+h0]);vp=pl.assemble(vp,vr,[qg,hv0])
                    vp=pl.assemble(vp,pl.full([QT,HDP-HD],dtype=pl.BF16,value=0.0),[qg,hv0+HD])
    # v2o: chunked SINGLE QK sweep — kt loaded once per (fa,kb), shared across
    # the RQA CQA-row chunks. Per-chunk local row max / exp / sum, pe_s rows and
    # per-chunk (m_kb,l_kb) stats materialized to GM.
    # crop-block merge: spmd(SL_A) + serial inner b loop (was
    # spmd(VB*SL_A)); fa=b*SL_A+rr keeps the EXACT fa numbering, so pe_s/mb_s/lb_s
    # layouts and every index formula are unchanged. Same 1 submit, block_num
    # VB*SL_A -> SL_A.
    # REVERT of the v2o+v2f single-pass online-softmax fusion: the
    # sub-outer (query-chunk-outer) tiling forced 13x K/V re-read per layer
    # (~2.8GB extra GM traffic) vs this kb-outer shared-load (1x KV read) —
    # measured +23% (608.5ms vs 495.0ms). DSv4's "flash" is sparse + keeps
    # probability materialization (cube/vector are separate cores); VIT is dense
    # and AIV-compute-bound, so the premise did not transfer. Two-pass restored.
    for rr in pl.spmd(SL_A,name_hint='v2o'):
        for b in pl.range(VB):
            if b<active:
                fa=b*SL_A+rr;qt=rr//HL;h=rr%HL;qg=b*vT+qt*MQA;h0=h*HD;ois=fa*MQA
                for kb in pl.range(nk):
                    k0=kb*KT;kt=pl.slice(kr,[KT,HD],[b*vT+k0,h0])
                    for sub in pl.range(RQA):
                        q16=pl.slice(qr,[CQA,HD],[qg+sub*CQA,h0])
                        raw=pl.mul(pl.matmul(q16,kt,b_trans=True,out_dtype=pl.FP32),V_ATTN_SCALE)
                        mkb=pl.reshape(pl.row_max(raw),[1,CQA])
                        e=pl.exp(pl.row_expand_sub(raw,pl.reshape(mkb,[CQA,1])))
                        pe_s=pl.assemble(pe_s,pl.cast(e,target_type=pl.BF16,mode="rint"),[ois+sub*CQA,k0])
                        mb_s=pl.assemble(mb_s,mkb,[fa*nk+kb,sub*CQA])
                        lb_s=pl.assemble(lb_s,pl.reshape(pl.row_sum(e),[1,CQA]),[fa*nk+kb,sub*CQA])
    # v2f: chunked PV — vt loaded once per (fa,kb), shared across the RQA
    # chunks. mb_s/lb_s are per-row [sl_a*nk,MQA] (v2o writes each chunk's
    # [1,CQA] stats via GM column-offset stores), so mi/acc_l reduce as WHOLE
    # [1,MQA] tiles (the baseline pattern). acc_o is a GM accumulator, but the
    # compiler caches it UB-resident for the RMW — a [MQA,128] fp32 tile
    # (104KB) overflows the 188,416B vec wall, so HDP=128 is processed as two
    # sequential 64-wide halves (acc_o0/acc_o1, 52KB resident each). Assemble
    # targets GM only (ptoas rejects subview-tmov into UB tiles for computed
    # sources). Cost vs the ideal 1x: pe_l is re-read per half (pe_s 2x =
    # 468MB/layer), far less than the 13x KV re-read replaced. Per-column
    # accumulation order over kb is unchanged -> bit-identical.
    # crop-block merge: spmd(SL_A) + serial inner b loop (was
    # spmd(VB*SL_A)); fa=b*SL_A+rr — same fa numbering, buffers/indices
    # unchanged. mi/acc_l/acc_o0/acc_o1 are per-(b,fa) state and live inside
    # the b loop; create_tensor inside a serial loop is the per_rank layer-loop
    # pattern (x1buf), so per-iteration scratch is proven codegen.
    for rr in pl.spmd(SL_A,name_hint='v2f'):
        for b in pl.range(VB):
            if b<active:
                fa=b*SL_A+rr;qt=rr//HL;h=rr%HL;qg=b*vTp+qt*MQA;hv0=h*HDP;ois=fa*MQA  # qg: attn_ctx row base — vTp stride (M-tile promotion of v3; ctx rows [2704,2736)/image are never-written pad, garbage confined to pbuf pad rows)
                mi=pl.full([1,MQA],dtype=pl.FP32,value=-3.0e38)
                for kb in pl.range(nk):
                    mi=pl.maximum(mi,pl.slice(mb_s,[1,MQA],[fa*nk+kb,0]))
                acc_l=pl.full([1,MQA],dtype=pl.FP32,value=0.0)
                for kb in pl.range(nk):
                    corr=pl.exp(pl.sub(pl.slice(mb_s,[1,MQA],[fa*nk+kb,0]),mi))  # [1,MQA], in (0,1]
                    acc_l=pl.add(acc_l,pl.mul(pl.slice(lb_s,[1,MQA],[fa*nk+kb,0]),corr))
                # NOTE: reshape of a FULL [1,MQA] tile (no slice) is safe; reshaping a
                # column-offset [1,CQA] row-vector slice DROPS the column offset in
                # ptoas codegen (subview becomes dead, tile re-based at the tile base
                # addr). So the per-chunk [CQA,1] rescale is taken as a ROW-offset 2D
                # sub-slice of the col-vector (2D row offsets carry, like acc_o0).
                acc_l_col=pl.reshape(acc_l,[MQA,1])
                # -- columns [0,64) --
                acc_o0=pl.create_tensor([MQA,64],dtype=pl.FP32)
                # create_tensor GM scratch is NOT zero-initialized; the compiler caches
                # this RMW accumulator UB-resident, so the first TADD would read stale
                # UB. A pl.full([MQA,64],0.0) seed is rejected (its TEXPANDS lands in the
                # CUBE kernel where vector fills are illegal). Instead kb=0 is a pure
                # WRITE pass (po lands directly in acc_o0, no read of uninit), then
                # kb=1..12 RMW. Per-row accumulation order over kb is unchanged.
                vt0=pl.slice(vp,[KT,64],[b*vT+0,hv0])
                corr0=pl.exp(pl.sub(pl.slice(mb_s,[1,MQA],[fa*nk,0]),mi))
                corr0_col=pl.reshape(corr0,[MQA,1])
                for sub in pl.range(RQA):
                    pe_l=pl.slice(pe_s,[CQA,KT],[ois+sub*CQA,0])
                    po=pl.row_expand_mul(pl.matmul(pe_l,vt0,out_dtype=pl.FP32),pl.slice(corr0_col,[CQA,1],[sub*CQA,0]))
                    acc_o0=pl.assemble(acc_o0,po,[sub*CQA,0])
                for kb in pl.range(1,nk):
                    k0=kb*KT;vt=pl.slice(vp,[KT,64],[b*vT+k0,hv0])
                    corr=pl.exp(pl.sub(pl.slice(mb_s,[1,MQA],[fa*nk+kb,0]),mi))  # [1,MQA], in (0,1]
                    corr_col=pl.reshape(corr,[MQA,1])
                    for sub in pl.range(RQA):
                        pe_l=pl.slice(pe_s,[CQA,KT],[ois+sub*CQA,k0])
                        po=pl.row_expand_mul(pl.matmul(pe_l,vt,out_dtype=pl.FP32),pl.slice(corr_col,[CQA,1],[sub*CQA,0]))
                        acc_o0=pl.assemble(acc_o0,pl.add(pl.slice(acc_o0,[CQA,64],[sub*CQA,0]),po),[sub*CQA,0])
                for sub in pl.range(RQA):
                    ctx=pl.row_expand_div(pl.slice(acc_o0,[CQA,64],[sub*CQA,0]),pl.slice(acc_l_col,[CQA,1],[sub*CQA,0]))
                    attn_ctx=pl.assemble(attn_ctx,pl.cast(ctx,target_type=pl.BF16,mode="rint"),[qg+sub*CQA,hv0])
                # -- columns [64,128) --
                acc_o1=pl.create_tensor([MQA,64],dtype=pl.FP32)
                vt0=pl.slice(vp,[KT,64],[b*vT+0,hv0+64])
                corr0=pl.exp(pl.sub(pl.slice(mb_s,[1,MQA],[fa*nk,0]),mi))
                corr0_col=pl.reshape(corr0,[MQA,1])
                for sub in pl.range(RQA):
                    pe_l=pl.slice(pe_s,[CQA,KT],[ois+sub*CQA,0])
                    po=pl.row_expand_mul(pl.matmul(pe_l,vt0,out_dtype=pl.FP32),pl.slice(corr0_col,[CQA,1],[sub*CQA,0]))
                    acc_o1=pl.assemble(acc_o1,po,[sub*CQA,0])
                for kb in pl.range(1,nk):
                    k0=kb*KT;vt=pl.slice(vp,[KT,64],[b*vT+k0,hv0+64])
                    corr=pl.exp(pl.sub(pl.slice(mb_s,[1,MQA],[fa*nk+kb,0]),mi))  # [1,MQA], in (0,1]
                    corr_col=pl.reshape(corr,[MQA,1])
                    for sub in pl.range(RQA):
                        pe_l=pl.slice(pe_s,[CQA,KT],[ois+sub*CQA,k0])
                        po=pl.row_expand_mul(pl.matmul(pe_l,vt,out_dtype=pl.FP32),pl.slice(corr_col,[CQA,1],[sub*CQA,0]))
                        acc_o1=pl.assemble(acc_o1,pl.add(pl.slice(acc_o1,[CQA,64],[sub*CQA,0]),po),[sub*CQA,0])
                for sub in pl.range(RQA):
                    ctx=pl.row_expand_div(pl.slice(acc_o1,[CQA,64],[sub*CQA,0]),pl.slice(acc_l_col,[CQA,1],[sub*CQA,0]))
                    attn_ctx=pl.assemble(attn_ctx,pl.cast(ctx,target_type=pl.BF16,mode="rint"),[qg+sub*CQA,hv0+64])
    return attn_ctx

# ── v4 split bodies (LN once + plain BF16 GEMMs) — verbatim from the retired three-phase repl tower ──
@pl.jit.inline
def ln_fwd(x:pl.Tensor[[vTpE,D],pl.BF16],g1:pl.Tensor[[1,D],pl.FP32],b1:pl.Tensor[[1,D],pl.FP32],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//QT
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='lnf'):
                qx=b*vTp+tg*QT;xb=pl.cast(pl.slice(x,[QT,D],[qx,0]),target_type=pl.FP32)  # x is vTp-strided too (fused residual path: rbb/x1buf carry the pad stride); reads only real rows tg<169
                qw=b*vTp+tg*QT  # lnbuf has the v1g stride-padded per-image stride vTp
                v_s=pl.full([1,QT],dtype=pl.FP32,value=0.0);v_s2=pl.full([1,QT],dtype=pl.FP32,value=0.0)
                for v_db in pl.range(dn):
                    v_d0=v_db*DT;v_zc=pl.slice(xb,[QT,DT],[0,v_d0])
                    v_s=pl.add(v_s,pl.reshape(pl.row_sum(v_zc),[1,QT]));v_s2=pl.add(v_s2,pl.reshape(pl.row_sum(pl.mul(v_zc,v_zc)),[1,QT]))
                v_m=pl.mul(v_s,V_WIDTH_INV);v_v=pl.sub(pl.mul(v_s2,V_WIDTH_INV),pl.mul(v_m,v_m))
                v_iv=pl.rsqrt(pl.add(v_v,V_EPS),high_precision=True);v_mt=pl.reshape(v_m,[QT,1]);v_it=pl.reshape(v_iv,[QT,1])
                for v_cb in pl.range(qkk):
                    v_c0=v_cb*QKK;v_xc=pl.slice(xb,[QT,QKK],[0,v_c0])
                    v_o=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
                    v_cs=pl.sub(pl.row_expand_mul(v_xc,v_it),pl.row_expand_mul(v_o,pl.mul(v_mt,v_it)))
                    v_gc=pl.slice(g1,[1,QKK],[0,v_c0]);v_bc=pl.slice(b1,[1,QKK],[0,v_c0])
                    v_l=pl.cast(pl.add(pl.col_expand_mul(v_cs,v_gc),pl.col_expand_mul(v_o,v_bc)),target_type=pl.BF16,mode='rint')
                    out=pl.assemble(out,v_l,[qw,v_c0])
    return out

@pl.jit.inline
def qkv_gemm_fixpipe(ln:pl.Tensor[[vTpE,D],pl.BF16],wq:pl.Tensor[[D,KO],pl.BF16],qbuf:pl.Tensor[[vTpE,KO],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # FixPipe split (patch-tower recipe ported): pure cube, acc
    # drains L0C->GM through PIPE_FIX (FP32->BF16 in hardware, zero Vec).
    # N-chunk 64->FP_N, K-chunk QKK 256->FP_KC (Right [FP_KC,FP_N]x2Bx2
    # stages = 64KB cap-exact). M now QTM1_G=128 (M-decoupling): Left
    # [128,FP_KC]x2Bx2 = 32KB at FP_KC=64 <= 64KB, so the K loop keeps
    # pl.pipeline(stage=2). The bias epilogue moved to qkv_bias_fixpipe;
    # FixPipe pre-rounds acc to BF16 BEFORE the bias add (~1 ULP drift vs
    # the fused form, patch-tower L0-tight-gate-proven safe). K-split is
    # rounding-transparent (matmul_acc per-element FMA order unchanged).
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg1_G,name_hint='v1g'):
                qg=b*vTp+tg*QTM1_G
                for qq_nb in pl.range(on_fpn1):
                    qq_n0=qq_nb*FP_N
                    qq_a=pl.load(ln,[qg,0],[QTM1_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    qq_b=pl.load(wq,[0,qq_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    qq_acc=pl.matmul(pl.move(qq_a,target_memory=pl.MemorySpace.Left),pl.move(qq_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                    for qq_kb in pl.pipeline(v1g_kk-1,stage=2):
                        qq_k0=(qq_kb+1)*FP_KC
                        qq_a=pl.load(ln,[qg,qq_k0],[QTM1_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                        qq_b=pl.load(wq,[qq_k0,qq_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                        qq_acc=pl.matmul_acc(qq_acc,pl.move(qq_a,target_memory=pl.MemorySpace.Left),pl.move(qq_b,target_memory=pl.MemorySpace.Right))
                    qbuf=pl.store(qq_acc,[qg,qq_n0],qbuf,shapes=[QTM1_G,FP_N])
    return qbuf

@pl.jit.inline
def qkv_bias_fixpipe(qbuf:pl.Tensor[[vTpE,KO],pl.BF16],bq:pl.Tensor[[1,KO],pl.BF16],out:pl.Tensor[[vTpE,KO],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # Bias epilogue pass: out = qbuf + bias in FP32, one rint cast — same add
    # nesting as the old fused epilogue (acc-value first, bias second).
    # N-chunk = RES_N (Vec-wall discipline; the [80,RES_N] tile set is the
    # os-only M=80 case the probe prices).
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg1,name_hint='v1b'):
                qg=b*vTp+tg*QTM1
                for qb_nb in pl.range(on_res1):
                    qb_n0=qb_nb*RES_N
                    qb_pv=pl.cast(pl.slice(qbuf,[QTM1,RES_N],[qg,qb_n0]),target_type=pl.FP32)
                    qb_bt=pl.col_expand_mul(pl.full([QTM1,RES_N],dtype=pl.FP32,value=1.0),pl.cast(pl.slice(bq,[1,RES_N],[0,qb_n0]),target_type=pl.FP32))
                    qb_rs=pl.add(qb_pv,qb_bt)
                    out=pl.assemble(out,pl.cast(qb_rs,target_type=pl.BF16,mode='rint'),[qg,qb_n0])
    return out

@pl.jit.inline
def mlp_f1_gemm_fixpipe(ln:pl.Tensor[[vTpE,D],pl.BF16],wf1:pl.Tensor[[D,ILP],pl.BF16],p1buf:pl.Tensor[[vTpE,ILP],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # FixPipe split: p1buf = ln@wf1, cube only. N-chunk QKN1
    # 64 -> FP_N; K-chunk QN2 256 -> FP_KC (Right [FP_KC,FP_N]x2Bx2stage =
    # 64KB cap-exact; the fused form's [256,64] was cap-exact too). acc drains
    # L0C->GM via PIPE_FIX (zero Vec); the bias+gelu epilogue moved to
    # mlp_f1_gelu_fixpipe. K-trips 6->12->24 at FP_KC=64, per-element FMA
    # order unchanged (rounding-transparent).
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg_G,name_hint='v4ag'):
                qg=b*vTp+tg*QTM_G
                for za_nb in pl.range(on_fpnm):
                    za_n0=za_nb*FP_N
                    za_a=pl.load(ln,[qg,0],[QTM_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    za_b=pl.load(wf1,[0,za_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    za_acc=pl.matmul(pl.move(za_a,target_memory=pl.MemorySpace.Left),pl.move(za_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                    for za_kb in pl.pipeline(on2f-1,stage=2):
                        za_k0=(za_kb+1)*FP_KC
                        za_a=pl.load(ln,[qg,za_k0],[QTM_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                        za_b=pl.load(wf1,[za_k0,za_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                        za_acc=pl.matmul_acc(za_acc,pl.move(za_a,target_memory=pl.MemorySpace.Left),pl.move(za_b,target_memory=pl.MemorySpace.Right))
                    p1buf=pl.store(za_acc,[qg,za_n0],p1buf,shapes=[QTM_G,FP_N])
    return p1buf

@pl.jit.inline
def mlp_f1_gelu_fixpipe(p1buf:pl.Tensor[[vTpE,ILP],pl.BF16],bf1:pl.Tensor[[1,ILP],pl.BF16],g:pl.Tensor[[vTpE,ILP],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # gelu epilogue pass: p = p1buf + bias in FP32 (acc-value first, bias
    # second — the fused form's add nesting), t = p*_QG, sig = 1/(1+exp(-t)),
    # out = cast(p*sig, rint). Only delta vs the fused math: p1buf was rounded
    # to BF16 by the FixPipe BEFORE the bias add and the gelu nonlinearity
    # (~1 ULP input drift, device-gate-proven safe). N-chunk = RES_N.
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg,name_hint='v4ae'):
                qg=b*vTp+tg*QTM
                for zb_nb in pl.range(on_resm):
                    zb_n0=zb_nb*RES_N
                    zb_pv=pl.add(pl.cast(pl.slice(p1buf,[QTM,RES_N],[qg,zb_n0]),target_type=pl.FP32),pl.col_expand_mul(pl.full([QTM,RES_N],dtype=pl.FP32,value=1.0),pl.cast(pl.slice(bf1,[1,RES_N],[0,zb_n0]),target_type=pl.FP32)))
                    zb_t=pl.mul(zb_pv,_QG);zb_sig=pl.recip(pl.add(pl.exp(pl.neg(zb_t)),1.0))
                    g=pl.assemble(g,pl.cast(pl.mul(zb_pv,zb_sig),target_type=pl.BF16,mode='rint'),[qg,zb_n0])
    return g

@pl.jit.inline
def mlp_f2(g:pl.Tensor[[vTpE,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # M-tile promotion: [QT,QN2]=[16,256] -> [48,64] (K chunks stay QN=128, pl.pipeline form); out (p2buf) gets the vTp stride (pad garbage confined; r2's read base moves to b*vTp). spmd 169 -> 57.
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg,name_hint='v4b'):
                qg=b*vTp+tg*QTM;lr=pl.reshape(ls,[1,D])
                for v_nb in pl.range(dn1):
                    v_n0=v_nb*QKN1
                    v_f2=pl.matmul(pl.slice(g,[QTM,QN],[qg,0]),wf2[0:QN,v_n0:v_n0+QKN1],out_dtype=pl.FP32)
                    for v_ib in pl.pipeline(inn-1,stage=2):
                        v_i0=(v_ib+1)*QN
                        v_f2=pl.matmul_acc(v_f2,pl.slice(g,[QTM,QN],[qg,v_i0]),wf2[v_i0:v_i0+QN,v_n0:v_n0+QKN1])
                    v_lc=pl.slice(lr,[1,QKN1],[0,v_n0]);v_r=pl.col_expand_mul(v_f2,v_lc)
                    out=pl.assemble(out,pl.cast(v_r,target_type=pl.BF16,mode='rint'),[qg,v_n0])
    return out

vb2_in=pl.inline(vb2_all_os._func)
ln_fwd_in=pl.inline(ln_fwd._func);qkv_gemm_in=pl.inline(qkv_gemm_fixpipe._func);qkv_bias_in=pl.inline(qkv_bias_fixpipe._func)
mlp_f1_gemm_in=pl.inline(mlp_f1_gemm_fixpipe._func);mlp_f1_gelu_in=pl.inline(mlp_f1_gelu_fixpipe._func);mlp_f2_in=pl.inline(mlp_f2._func)

# ── vb3_out_proj FULL — HL-generic K-tile accumulation (verbatim) ──
@pl.jit.inline
def vb3_partial(ctx:pl.Tensor[[vTpE,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # M-tile promotion: [QT,QN2]=[16,256] -> [48,64] (per-head K chunks stay HDP=128, pl.pipeline form); ctx (attn_ctx) and out (pbuf) carry the vTp stride (v2f writes ctx at b*vTp; block 57's pad rows are stale-GM garbage confined to never-read pbuf pad). r1's pbuf read base moves to b*vTp. spmd 169 -> 57.
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg,name_hint='v3'):
                qg=b*vTp+tg*QTM;lr=pl.reshape(ls,[1,D])
                for nb in pl.range(dn1):
                    n0=nb*QKN1
                    acc=pl.matmul(pl.slice(ctx,[QTM,HDP],[qg,0]),pl.slice(wo,[HDP,QKN1],[0,n0]),out_dtype=pl.FP32)
                    for hb in pl.pipeline(HL-1,stage=2):
                        ho=(hb+1)*HDP
                        acc=pl.matmul_acc(acc,pl.slice(ctx,[QTM,HDP],[qg,ho]),pl.slice(wo,[HDP,QKN1],[ho,n0]))
                    lc=pl.slice(lr,[1,QKN1],[0,n0]);r=pl.col_expand_mul(acc,lc)
                    out=pl.assemble(out,pl.cast(r,target_type=pl.BF16,mode='rint'),[qg,n0])
    return out
vb3f_in=pl.inline(vb3_partial._func)

# ── FIXPIPE split kernels: v3g+v3e / v4g+v4e / v1g+v1b ──
# The tower is a strictly serial residual chain, but the fused epilogues
# (Phase-A retry at [48,64], replaced here) pinned N at 64: the last op
# before the store was vector-domain (scale/bias/residual), so acc had to
# materialize in UB -> aic+aiv kernels at ~144KB Vec (78%). Splitting moves
# the store INTO the cube: acc drains L0C->GM via PIPE_FIX (zero Vec), the
# epilogue becomes a standalone elementwise pass, and N doubles 64->128
# (serial N-chain per block 24->12; Right [128,128]x2Bx2stage = 64KB
# cap-exact). The un-fuse alone is ~neutral (patch-tower fixpipe@64
# measured so) — the win is N=128 SPECIFICALLY. Numeric delta: FixPipe
# pre-rounds acc to BF16 before the *ls/bias adds (~1 ULP, patch-tower
# L0-tight-gate-proven safe). First attempt (archived below)
# fused at the old [16,256] GEMM geometry and LOST +3.2%: the ~5 extra live
# [16,256] FP32 tiles blew the 188,416B Vec wall at QN2=256 and forced
# QN=128 N-chunks — narrower cube tiles cost more than the dispatch saved.
# At the [48,64] geometry the epilogue tiles fit, but the Vec budget still
# capped N at 64 — the wall this split removes.
@pl.jit.inline
def o_proj_gemm_fixpipe(ctx:pl.Tensor[[vTpE,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],pbuf:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # FixPipe split: pbuf = ctx@wo, cube only. N-chunk
    # 64->FP_N, K-chunk HDP=128 -> FP_KC (Right [FP_KC,FP_N]x2Bx2stage =
    # 64KB cap-exact; at FP_KC=64 the head dim splits into two 64-halves,
    # per-column FMA order unchanged). acc drains L0C->GM via PIPE_FIX
    # (zero Vec); the *ls/bias/residual epilogue moved to o_proj_resid_fixpipe.
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg_G,name_hint='v3g'):
                qg=b*vTp+tg*QTM_G
                for fg_nb in pl.range(on_fpn):
                    fg_n0=fg_nb*FP_N
                    fg_a=pl.load(ctx,[qg,0],[QTM_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    fg_b=pl.load(wo,[0,fg_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    fg_acc=pl.matmul(pl.move(fg_a,target_memory=pl.MemorySpace.Left),pl.move(fg_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                    for fg_hb in pl.pipeline(v3g_kk-1,stage=2):
                        fg_ho=(fg_hb+1)*FP_KC
                        fg_a=pl.load(ctx,[qg,fg_ho],[QTM_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                        fg_b=pl.load(wo,[fg_ho,fg_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                        fg_acc=pl.matmul_acc(fg_acc,pl.move(fg_a,target_memory=pl.MemorySpace.Left),pl.move(fg_b,target_memory=pl.MemorySpace.Right))
                    pbuf=pl.store(fg_acc,[qg,fg_n0],pbuf,shapes=[QTM_G,FP_N])
    return pbuf

@pl.jit.inline
def o_proj_resid_fixpipe(pbuf:pl.Tensor[[vTpE,D],pl.BF16],rbb:pl.Tensor[[vTpE,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],bo:pl.Tensor[[1,D],pl.BF16],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # out = rbb + pbuf*ls + bo*ls — the r1 residual pass, add nesting
    # verbatim from the fused form (rbb + (pbuf + bt) in FP32, one rint cast
    # at the end). Only delta vs the fused math: pbuf was rounded to BF16 by
    # the FixPipe BEFORE the *ls scale (fused rounded after) — ~1 ULP.
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg,name_hint='v3e'):
                qg=b*vTp+tg*QTM;lr=pl.reshape(ls,[1,D])
                for fe_nb in pl.range(on_res):
                    fe_n0=fe_nb*RES_N
                    fe_lc=pl.slice(lr,[1,RES_N],[0,fe_n0]);fe_bc=pl.slice(bo,[1,RES_N],[0,fe_n0])
                    fe_pv=pl.col_expand_mul(pl.cast(pl.slice(pbuf,[QTM,RES_N],[qg,fe_n0]),target_type=pl.FP32),fe_lc)
                    fe_bt=pl.col_expand_mul(pl.full([QTM,RES_N],dtype=pl.FP32,value=1.0),pl.mul(pl.cast(fe_bc,target_type=pl.FP32),fe_lc))
                    fe_rs=pl.add(pl.cast(pl.slice(rbb,[QTM,RES_N],[qg,fe_n0]),target_type=pl.FP32),pl.add(fe_pv,fe_bt))
                    out=pl.assemble(out,pl.cast(fe_rs,target_type=pl.BF16,mode='rint'),[qg,fe_n0])
    return out
o_proj_gemm_in=pl.inline(o_proj_gemm_fixpipe._func)
o_proj_resid_in=pl.inline(o_proj_resid_fixpipe._func)

@pl.jit.inline
def mlp_f2_gemm_fixpipe(g:pl.Tensor[[vTpE,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],p2buf:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # FixPipe split: p2buf = g@wf2, cube only. N-chunk
    # 64->FP_N; K-chunk QN3 256 -> QN 128 -> FP_KC (Right [FP_KC,FP_N]x2Bx
    # 2stage = 64KB cap-exact). K-trips 35->70->140 at FP_KC=64, per-element
    # FMA order unchanged (rounding-transparent).
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg_G,name_hint='v4g'):
                qg=b*vTp+tg*QTM_G
                for hg_nb in pl.range(on_fpn):
                    hg_n0=hg_nb*FP_N
                    hg_a=pl.load(g,[qg,0],[QTM_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    hg_b=pl.load(wf2,[0,hg_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    hg_acc=pl.matmul(pl.move(hg_a,target_memory=pl.MemorySpace.Left),pl.move(hg_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                    for hg_ib in pl.pipeline(v4g_kk-1,stage=2):
                        hg_i0=(hg_ib+1)*FP_KC
                        hg_a=pl.load(g,[qg,hg_i0],[QTM_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                        hg_b=pl.load(wf2,[hg_i0,hg_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                        hg_acc=pl.matmul_acc(hg_acc,pl.move(hg_a,target_memory=pl.MemorySpace.Left),pl.move(hg_b,target_memory=pl.MemorySpace.Right))
                    p2buf=pl.store(hg_acc,[qg,hg_n0],p2buf,shapes=[QTM_G,FP_N])
    return p2buf

@pl.jit.inline
def mlp_f2_resid_fixpipe(p2buf:pl.Tensor[[vTpE,D],pl.BF16],x1buf:pl.Tensor[[vTpE,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],bf2:pl.Tensor[[1,D],pl.BF16],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # out = x1buf + p2buf*ls + bf2*ls — the r2 residual pass, same shape as
    # o_proj_resid_fixpipe (FixPipe pre-rounds p2buf before the *ls scale).
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(nqg,name_hint='v4e'):
                qg=b*vTp+tg*QTM;lr=pl.reshape(ls,[1,D])
                for he_nb in pl.range(on_res):
                    he_n0=he_nb*RES_N
                    he_lc=pl.slice(lr,[1,RES_N],[0,he_n0]);he_bc=pl.slice(bf2,[1,RES_N],[0,he_n0])
                    he_pv=pl.col_expand_mul(pl.cast(pl.slice(p2buf,[QTM,RES_N],[qg,he_n0]),target_type=pl.FP32),he_lc)
                    he_bt=pl.col_expand_mul(pl.full([QTM,RES_N],dtype=pl.FP32,value=1.0),pl.mul(pl.cast(he_bc,target_type=pl.FP32),he_lc))
                    he_rs=pl.add(pl.cast(pl.slice(x1buf,[QTM,RES_N],[qg,he_n0]),target_type=pl.FP32),pl.add(he_pv,he_bt))
                    out=pl.assemble(out,pl.cast(he_rs,target_type=pl.BF16,mode='rint'),[qg,he_n0])
    return out
mlp_f2_gemm_in=pl.inline(mlp_f2_gemm_fixpipe._func)
mlp_f2_resid_in=pl.inline(mlp_f2_resid_fixpipe._func)

# ARCHIVE (superseded by the fused kernels, which the
# FixPipe split in turn replaced): a fused-epilogue
# variant (vb3r1/mlp_f2_r2, ported from the patch tower) was measured on real
# cards at the THEN-current [16,256] GEMM geometry and REVERTED: golden PASS
# but bench 1004.1ms vs 973.0ms baseline (+3.2%). The fused kernel kept ~5
# extra [QT,QN] FP32 tiles live, which at QN2=256 blew the 188KB Vec wall
# (195072 > 188416 bytes) and forced QN=128 N-chunks — narrower cube tiles
# cost more than the 2-stages/layer dispatch saved. The os wall is per-stage
# dispatch gaps, but stage fusion must not shrink the GEMM tile geometry —
# the retry ran at [48,64] where the epilogue tiles fit (though
# their Vec budget still capped N at 64 until the FixPipe split removed it).

RCH=47
RNCH=(V_LAYERS+RCH-1)//RCH


def _build_repl(tp_size:int=8, chunk:int=0):
    @pl.program
    class Step3p7VisionReplOS:
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x_in:pl.Tensor[[vTe,D],pl.BF16],
            rq:pl.Tensor[[vWQ,KO],pl.BF16],ro:pl.Tensor[[V_LAYERS*DLP,V_WIDTH],pl.BF16],
            rbq:pl.Tensor[[V_LAYERS,KO],pl.BF16],rbo:pl.Tensor[[V_LAYERS,V_WIDTH],pl.BF16],
            rf1:pl.Tensor[[vWF1,ILP],pl.BF16],rf2:pl.Tensor[[vWF2,V_WIDTH],pl.BF16],
            rbf1:pl.Tensor[[V_LAYERS,ILP],pl.BF16],rbf2:pl.Tensor[[V_LAYERS,V_WIDTH],pl.BF16],
            rl1g:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],rl1b:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],
            rl2g:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],rl2b:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],
            rl1:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],rl2:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],
            rcos:pl.Tensor[[vTp,HD],pl.FP32],rsin:pl.Tensor[[vTp,HD],pl.FP32],
            rto:pl.Out[pl.Tensor[[vTe,D],pl.BF16]],
            lo:pl.Scalar[pl.INT32],
            nl:pl.Scalar[pl.INT32],
            active:pl.Scalar[pl.INDEX],
        )->pl.Tensor[[vTe,D],pl.BF16]:
            nq=vT//QT
            # rbb must be vTp-strided for the fused residual epilogues (their
            # block 57 writes pad rows [2704,2736); a vT-strided out would
            # clobber image b+1's real rows). x_in is a [vTe] input param, so
            # copy the real rows once per tower into a vTp scratch (cp0 — one
            # wrto-shaped dispatch for the whole tower, ~35µs).
            rbb=pl.create_tensor([vTpE,D],dtype=pl.BF16)
            for b in pl.parallel(VB):
                if b<active:
                    for c in pl.spmd(nq,name_hint='cp0'):
                        cq=b*vT+c*QT;cw=b*vTp+c*QT  # read vT (x_in), write vTp (rbb scratch)
                        rbb=pl.assemble(rbb,pl.slice(x_in,[QT,D],[cq,0]),[cw,0])
            x1buf=pl.create_tensor([vTpE,D],dtype=pl.BF16)
            lnbuf=pl.create_tensor([vTpE,D],dtype=pl.BF16);gbuf=pl.create_tensor([vTpE,ILP],dtype=pl.BF16)  # gbuf: vTp-stride (v4a/v4b/v3 M-tile promotion); lnbuf: v1g stride-padded; rbb/x1buf: vTp (fused residual path)
            # pbuf/p2buf/qbuf are the FixPipe direct-store targets:
            # created once, vTp-strided; each layer's GEMM pass overwrites every
            # row it owns before the resid/bias pass reads it (same discipline
            # as lnbuf). qbuf [vTpE,KO] ~= 50MB resident.
            pbuf=pl.create_tensor([vTpE,D],dtype=pl.BF16);p2buf=pl.create_tensor([vTpE,D],dtype=pl.BF16);qbuf=pl.create_tensor([vTpE,KO],dtype=pl.BF16);p1buf=pl.create_tensor([vTpE,ILP],dtype=pl.BF16)
            for L in pl.range(nl):
                gi=lo+L;qo=gi*V_WIDTH;oo=gi*DLP;f1o=gi*V_WIDTH;f2o=gi*ILP
                lq=pl.slice(rq,[V_WIDTH,KO],[qo,0]);lwo=pl.slice(ro,[DLP,V_WIDTH],[oo,0])
                lbq=pl.slice(rbq,[1,KO],[gi,0]);lbo=pl.slice(rbo,[1,V_WIDTH],[gi,0])
                lf1=pl.slice(rf1,[V_WIDTH,ILP],[f1o,0]);lf2=pl.slice(rf2,[ILP,V_WIDTH],[f2o,0])
                lbf1=pl.slice(rbf1,[1,ILP],[gi,0]);lbf2=pl.slice(rbf2,[1,V_WIDTH],[gi,0])
                ll1=pl.slice(rl1,[1,V_WIDTH],[gi,0]);ll2=pl.slice(rl2,[1,V_WIDTH],[gi,0])
                ll1g=pl.slice(rl1g,[1,V_WIDTH],[gi,0]);ll1b=pl.slice(rl1b,[1,V_WIDTH],[gi,0])
                ll2g=pl.slice(rl2g,[1,V_WIDTH],[gi,0]);ll2b=pl.slice(rl2b,[1,V_WIDTH],[gi,0])
                qf=pl.create_tensor([vTpE,KO],dtype=pl.BF16);ac=pl.create_tensor([vTpE,DLP],dtype=pl.BF16)  # qf/ac: vTp-stride (v1g / v3 M-tile promotion)
                lnbuf=ln_fwd_in(rbb,ll1g,ll1b,lnbuf,active)
                qbuf=qkv_gemm_in(lnbuf,lq,qbuf,active)
                qf=qkv_bias_in(qbuf,lbq,qf,active)
                ac=vb2_in(qf,rcos,rsin,ac,active)
                x1buf=pl.create_tensor([vTpE,D],dtype=pl.BF16)
                pbuf=o_proj_gemm_in(ac,lwo,pbuf,active)
                x1buf=o_proj_resid_in(pbuf,rbb,ll1,lbo,x1buf,active)  # FixPipe split: v3g + v3e replace the fused v3r
                lnbuf=ln_fwd_in(x1buf,ll2g,ll2b,lnbuf,active)
                p1buf=mlp_f1_gemm_in(lnbuf,lf1,p1buf,active)
                gbuf=mlp_f1_gelu_in(p1buf,lbf1,gbuf,active)  # FixPipe split: v4ag + v4ae replace the fused v4a
                rbb=pl.create_tensor([vTpE,D],dtype=pl.BF16)
                p2buf=mlp_f2_gemm_in(gbuf,lf2,p2buf,active)
                rbb=mlp_f2_resid_in(p2buf,x1buf,ll2,lbf2,rbb,active)  # FixPipe split: v4g + v4e replace the fused v4r
            for b in pl.parallel(VB):
                if b<active:
                    for _wr in pl.spmd(nq,name_hint='wrto'):
                        _wrr=b*vT+_wr*QT;_wrw=b*vTp+_wr*QT  # read vTp (rbb, fused residual path); write vT (rto)
                        rto=pl.assemble(rto,pl.slice(rbb,[QT,D],[_wrw,0]),[_wrr,0])
            return rto
        @pl.function(level=pl.Level.HOST,role=pl.Role.Orchestrator)
        def host_orch(self,
            hx_in:pl.Tensor[[tp_size,vTe,V_WIDTH],pl.BF16],
            hq:pl.Tensor[[tp_size,vWQ,KO],pl.BF16],ho:pl.Tensor[[tp_size,V_LAYERS*DLP,V_WIDTH],pl.BF16],
            hbq:pl.Tensor[[tp_size,V_LAYERS,KO],pl.BF16],hbo:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.BF16],
            hf1:pl.Tensor[[tp_size,vWF1,ILP],pl.BF16],hf2:pl.Tensor[[tp_size,vWF2,V_WIDTH],pl.BF16],
            hbf1:pl.Tensor[[tp_size,V_LAYERS,ILP],pl.BF16],hbf2:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.BF16],
            hl1g:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],hl1b:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],
            hl2g:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],hl2b:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],
            hl1:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],hl2:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],
            hcos:pl.Tensor[[tp_size,vTp,HD],pl.FP32],hsin:pl.Tensor[[tp_size,vTp,HD],pl.FP32],
            hmid:pl.Tensor[[tp_size,vTe,V_WIDTH],pl.BF16],
            hto:pl.Out[pl.Tensor[[tp_size,vTe,V_WIDTH],pl.BF16]],
            hnl:pl.Scalar[pl.INT32],
            hactive:pl.Scalar[pl.INDEX],
        ):
            if chunk==0:
                for r in pl.range(pld.world_size()):
                    self.per_rank(hx_in[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hto[r],0,hnl,hactive,device=r)
            else:
                for r in pl.range(pld.world_size()):
                    self.per_rank(hx_in[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hmid[r],0,RCH,hactive,device=r)
                    for c in pl.range(1,RNCH-1):
                        self.per_rank(hmid[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hmid[r],c*RCH,RCH,hactive,device=r)
                    self.per_rank(hmid[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hto[r],(RNCH-1)*RCH,V_LAYERS-(RNCH-1)*RCH,hactive,device=r)
    return Step3p7VisionReplOS

Step3p7VisionReplOS=_build_repl(8)
# RCH>=V_LAYERS (RNCH==1): the chained first/middle/last dispatch loop would
# run the tower twice ((0,RCH) head and (0,V_LAYERS) tail overlap), so collapse
# Chunked to the single full-depth dispatch (identical to Single). RCH=47 is
# the perf default: bind has a ~208ms fixed cost per dispatch (12x at RCH=4
# cost ~2.5s of pure host tax); device wall is flat vs RCH (919 vs 920ms).
Step3p7VisionReplOSChunked=_build_repl(8,chunk=RCH if RNCH>1 else 0)
