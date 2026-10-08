# Copyright (c) PyPTO Contributors. SPDX-License-Identifier: Apache-2.0
# Step3.7 PATCH REPL tower — VB=6 ACTIVE-ONLY specialization of the flat
# online-softmax draft (vision_fwd_patch_repl_os_flat.py).
#
# ── N256QKV (PROMOTED after same-session device A/B) ──
# The last non-256 GEMM is now on the ladder: v1g's N-chunk goes 128 ->
# QKN_Q=256 with the derived K-chunk QKK_Q=64 (Right/L0B = [64,256]x2Bx2
# stage = 64KB cap-exact, the L2-line-rule treatment the os tower won
# -6.8% with). v1b (bias pass) chunks at RES_NQ=256 (its FP32 [48,256]
# tiles = 97.5KB=53.0% Vec). Same-session interleaved 5-run A/B (8x910B,
# harness device_wall fallback_flattened): 757.2ms vs flat6 donor 768.5ms
# = -1.5% (small but consistent — every new run beat its adjacent donor;
# attention is ~51% of this tower so qkv is a small slice). L0 tight
# gates (5e-3/5e-3/2%, layers 0+46) + full-tower loose PASS.
#
# ── N256F1 (PROMOTED after same-session device A/B) ──
# Stacks everything: the v3g/v4g N-ladder at FP_N=256 (pinned below, was
# the PYPTO_FP_N probe knob), the v1g128 qkv split (QKN_Q/QKK_Q=128/128),
# and the mlp_f1 FixPipe split (v4ag pure-cube direct store to p1buf
# [vTe,ILP] + v4ae gelu pass, chunked at RES_N). Same-session harness
# device_wall (active=6): 773.45/774.39ms vs the n256 donor 872.18ms =
# -11.3%; @128 anchor 825.89ms (that point also carries the qkv split, so
# it is not a flat6 replica). Gates: L0 tight 5e-3 layers 0+46 PASS,
# full-tower loose PASS x3. Whole network 392.1+773.9 ~= 1166ms = 2.41x
# vLLM 484.5 (was 1266/2.61x). The n256/v1g128/f1/fixpipe drafts are now
# all subsumed (driver marks them retired).
#
# ── N256 PROBE: can FP_N go past 128? ──
# v3g/v4g N-chunk generalized to env PYPTO_FP_N with a DERIVED K-chunk:
#   FP_KC = 128 if FP_N <= 128 else 64
# (Right/L0B = [FP_KC,FP_N]x2Bx2stage must stay <= 64KB; FP_N=256/FP_KC=64
# gives 64KB exactly). FP_KC=64 breaks the L2-line x-rule (128B A rows —
# the ds1 precedent says this is a tax), so the expectation is neutral-
# to-negative; this draft exists to price that tax with one measured point.
# v1g keeps QKN_Q=128 (its N=256 would need the same KC=64 treatment and
# is orthogonal). Everything else is a verbatim copy of the v1g128 draft.
#
# ── v1g128 DRAFT: qkv gets the SAME FixPipe split v3r/v4r got ──
# qkv_gemm (bias-add fused epilogue, QKN=64 / QKK=256) is split into
# qkv_gemm_fixpipe (pure cube, acc -> GM through the FixPipe, zero Vec) +
# qkv_bias_fixpipe (bias add in FP32 + rint cast) with the N-chunk doubled
# QKN 64->128 and the K-chunk halved QKK 256->128 (Right/L0B =
# [128,128]x2Bx2 stages = 64KB = exactly the wall; x rows 256B = one L2
# line, the proven KC=128 pattern). qkv feeds ATTENTION, the tightest
# module — the FixPipe's pre-bias BF16 rounding (~1 ULP) must pass the
# per-layer 5e-3 gate, like it did for v3r/v4r (L0 PASS).
#
# QKK=256 is shared with mlp_f1's K-loop, so this draft introduces
# qkv-private constants QKN_Q/QKK_Q (128/128) and leaves QKK untouched.
# Everything else is a verbatim copy of the promoted flat6 form.
# v3r/v4r (o_proj / mlp_f2, both "scale+bias+residual" fused epilogues) are
# split into a pure-cube GEMM whose FP32 accumulator drains STRAIGHT to GM
# through the A2/A3 FixPipe (tstore_acc2gm, PIPE_FIX — no Vec epilogue, zero
# UB occupancy: pl.store(acc) codegens to a single-AIC kernel) plus a
# separate elementwise residual pass (v3e/v4e) carrying the *ls scale + bias
# + residual add the fusion used to do on-chip.
#
# Why: the fused epilogues forced L0C->Vec->GM and pinned N at 64 (their
# N=128 tile set blew the 184KB Vec wall: 190.1KB rejected in the m64 probe).
# Un-fused, the GEMM's Vec usage drops to ZERO and N=128 becomes expressible
# (Right/L0B = [128,128]x2Bx2 stages = 64KB = exactly the wall). FP_N=128
# fixed after the device A/B (harness run_l3_e2e_repl, active=6,
# strace device_wall): flat6 fused 1027.1ms -> fixpipe@128 918.6ms (-10.6%);
# fixpipe@64 was neutral (1035.9 median) — the win is N=128 itself, the
# un-fuse alone does not pay. Numerics: L0 single-layer tight gate
# (5e-3/5e-3/2%) PASS and full-tower loose gate (1.2e-1/8e-2/1%) PASS —
# the FixPipe's pre-scale BF16 rounding (~1 ULP drift on the projection
# term) survives both gates.
#
# Provenance: this module is a verbatim copy of the flat6 form, with the four
# kernels below replacing the fused vb3r1_fused / mlp_f2_r2_fused epilogues.
#
# The flat draft carries the static crop CAPACITY VB=V_PATCH_BATCH=16, but the
# common 6-crop patch path runs active=6: the non-attention phases already run
# active-only (parallel(VB) + `if b<active` guards), while the attention
# phases (v2r/v2o/v2f) are FLAT unguarded spmd(VB*SLG) over all capacity
# crops — (VB-6)/VB of the attention work, pe_s GM traffic and kt/vt
# re-reads are spent on crops that are never compared. The vLLM reference
# processes exactly the active crops (varlen unpad).
#
# This module is a verbatim copy of the flat draft with ONE constant change:
# VB=6 (vTe=7776, spmd lanes 6*SLG=2592, pe_s2/mb_s2/lb_s2 sized by VB).
# Per-lane math is unchanged — for the 6 active crops the outputs are
# bit-identical to the flat draft; only the inactive-crop lanes disappear.
# Expected: attention 48.5ms -> ~36.4ms (x6/8), tower 130.4 -> ~118.3ms
# (0.955x vLLM 124.0ms).
#
# Constraint: only valid with --active 6 (the driver enforces this). For any
# other crop count the drivers use the flat tower (VB=V_PATCH_BATCH=16
# capacity + active scalar) — the crop count is NEVER hardcoded (see
# vision_config.py V_PATCH_BATCH); this file is only the tuned fast path
# for exactly-6-crop images.
#
# ── B-FLATTENED guarded phases (orch/sched dispatch-count rework) ──
# The 7 per-b guarded phases (lnf x2, v1g, v3r, v4a, v4r, wrto) previously ran
# `for b in pl.parallel(VB): if b<active: spmd(n)` — with active=6=VB that is 6
# task submits per phase per layer, 39 tasks/layer, 1839/tower. The patch tower
# is scheduler-drain-bound (sched 1481ms ~= 0.81ms/task; orch window 1306ms =
# ring-buffer backpressure pacing the orchestrator), so task COUNT is the wall.
# All 7 phases are now FLAT spmd(VB*n) with b=tg//n lane decode + block-level
# `if b<active` guard — the exact pattern the attention phases (v2r/v2o/v2f)
# and the os tower's B-axis attention already use. 10 tasks/layer, 471/tower
# (-74%). Per-lane math, tile shapes and accumulation order are unchanged —
# each (b,tg) lane computes the identical blocks — so outputs stay bit-identical.
#
# Everything else (v4 split bodies, vb3, fused epilogues, residual adds,
# chunked host_orch, pe_s ping-pong WAR protection) is copied verbatim from
# vision_fwd_patch_repl_os_flat.py; see that file for the full
# provenance of the flat design (ported from vision_fwd_repl_os.py,
# 795.1->194.0->130.4ms lineage).
import pypto.language as pl; import pypto.language.distributed as pld
import os
from .vision_config import (V_LAYERS,V_TOKENS_PATCH,V_WIDTH,V_EPS,V_WIDTH_INV,V_ATTN_SCALE,V_HEAD_DIM_PAD,V_HEADS,V_MLP_HIDDEN,V_MLP_HIDDEN_PAD,V_WIDTH_PAD)

# ── FULL-dim constants (same as the retired three-phase patch tower) ──
D=V_WIDTH;DT=128;QT=16;QN=128;QKN=64;QKK=256;RQT=8
DL=V_WIDTH;HD=96;HL=V_HEADS;IL=V_MLP_HIDDEN;ILP=V_MLP_HIDDEN_PAD;_QG=1.702
HDP=V_HEAD_DIM_PAD;DLP=V_WIDTH_PAD;KO=3*DL
KT=48;KTT=48;NKW=V_TOKENS_PATCH//KT;NKB=NKW;NQM=V_TOKENS_PATCH//QT;SL=NQM*HL;OI=SL*QT;HA=HD//2;G=3;MQ=G*QT;NGQM=NQM//G;SLG=NGQM*HL  # 1296=27*48 (KT=48 flash block: [MQ,48] tiles keep v2o under the 188KB Vec wall at MQ=48); G=3 groups/task -> MQ=48 (1296=48*27 exact, no pad)
# MQ is WALL-PINNED at 48 (measured, "MQ 48->144" experiment): a
# decoupled attention-only MQ_ATT=144 variant (v2o/v2f widened, GEMMs/v2r kept
# at 48) fails ptoa at COMPILE time — v2o_aiv Vec buffer 323136B / v2f_aiv
# 407232B vs the 188416B platform limit. Linear backoff: at MQ=48 the two
# phases sit at ~108/136KB = 57%/72% of the wall, so ~66 rows is the absolute
# ceiling — and no intermediate legal step exists (1296=2^4*3^4: the 16-aligned
# divisors are exactly {16,48,144,432,1296}; 96 does not divide 1296). GEMM
# M=144 is equally walled (~11 live [144,64] FP32 epilogue tiles = 396KB; the
# N=32 escape already measured +17% cube loss, 176.3->206.2ms).
vT=V_TOKENS_PATCH;VB=6;vTe=VB*vT  # VB=6 active-only: the ONLY delta vs vision_fwd_patch_repl_os_flat.py (VB=V_PATCH_BATCH=16 capacity there)
MQ_G=128  # GEMM M-chunk PINNED at 128 (promoted, M-decoupling): the four FixPipe projection GEMMs (v1g/v3g/v4ag/v4g) raise M 48->128 (Acc [128,256] fp32 = 128KB = L0C cap-exact); elementwise (v1b/v3e/v4ae/v4e) + attention keep MQ=48
nqg=vT//MQ;nqg_G=(vT+MQ_G-1)//MQ_G  # 27 / 11 blocks per image
vTp=max(nqg*MQ,nqg_G*MQ_G)  # shared per-image stride 1408 (=11*128): 1296=10*128+16 -> last block 16 real + 112 pad (7.9% pad-FLOP vs os 4.0%)
vTpE=VB*vTp
FP_N=256;FP_KC=64  # N-chunk PINNED at the ladder optimum (promoted after same-session device A/B: 256 = 773.9ms vs the n256 donor 872.2ms = -11.3%; @128 = 825.9ms, n384 regressed per the ladder); Right=[FP_KC,FP_N]x2Bx2stage = 64KB cap-exact; FP_KC=64 means 128B A rows = exactly the L2-line rule (the n384 KC=32 sub-line tax was the regression). N ceiling is the Acc wall: [48,FP_N]x4B<=128KB -> FP_N<=512 (768 breaks).
on_fpn=V_WIDTH//FP_N
RES_N=FP_N if FP_N<=256 else 256  # resid-pass N-chunk: the v3e/v4e FP32 [48,FP_N] tiles overflow the 184KB Vec wall at FP_N=384 (device compile verify-fail), so the elementwise passes chunk at <=256 independently of the GEMM N (slice boundaries need not align with store boundaries)
on_res=V_WIDTH//RES_N
v3g_kk=DLP//FP_KC;v4g_kk=ILP//FP_KC  # v3g/v4g K-chunk counts for the generalized pipeline (16/70 at FP_KC=128; 32/140 at 64; 64/280 at 32)
QKN_Q=256;QKK_Q=64  # v1g chunk sizes PINNED at 256/64 (promoted after same-session device A/B: 757.2ms vs the 768.5ms flat6 donor = -1.5%); Right/L0B [QKK_Q,QKN_Q]x2Bx2stage = 64KB cap-exact; QKK stays 256 for mlp_f1 (its K-loop shares the qkk count)
qkn_q=KO//QKN_Q;qkk_q=D//QKK_Q  # v1g chunk counts: 18/24 at 256 (4608%QKN_Q==0, 1536%QKK_Q==0)
RES_NQ=QKN_Q if QKN_Q<=256 else 256  # v1b pass N-chunk: the FP32 [MQ,QKN_Q] tiles would overflow the 184KB Vec wall past 256 (n384 lesson), so the bias pass chunks at <=256 independently of the GEMM N
on_fpn_f1=ILP//FP_N;f1_kk=D//FP_KC;on_res_f1=ILP//RES_N  # mlp_f1 split (promoted flat6 recipe, generalized to the probe knobs): N-chunk 64 -> FP_N, K-chunk QKK 256 -> FP_KC (Right [FP_KC,FP_N]x2Bx2stage = 64KB cap-exact), gelu pass chunks at RES_N (its FP32 [MQ,RES_N] tile set stays off the 184KB Vec wall, same discipline as v3e/v4e)
vWQ=V_LAYERS*V_WIDTH;vWF1=V_LAYERS*V_WIDTH;vWF2=V_LAYERS*ILP
qkn=KO//QKN;qkk=D//QKK;dn=D//DT;on=D//QN;inn=ILP//QN;on64=D//64;in64=ILP//64  # N-chunk counts: v1g now uses qkn_q=36 chunks of QKN_Q=128 (was qkn=72); v4a N=64 (in64=140) — unchanged; v3g/v4g loop on_fpn=12 chunks of FP_N=128 (was on64=24); inn=70 is the v4g K-loop chunk count

# ── v4 split bodies (B axis) — verbatim from the retired three-phase patch tower ──
@pl.jit.inline
def ln_fwd(x:pl.Tensor[[vTpE,D],pl.BF16],g1:pl.Tensor[[1,D],pl.FP32],b1:pl.Tensor[[1,D],pl.FP32],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # QT=16 whole-D tile (the t24d512 dispatch-granularity rework was measured
    # NEUTRAL on the tower — patch ln is not dispatch-dominated (486 tasks vs
    # the front/back's 972 pre-sweet-spot) and LN is a ~1-2ms fraction of the
    # 134.7ms tower, so the granularity lever does not transfer from the B6
    # kernel; reverted to keep the proven form).
    n=vT//QT
    for tg in pl.spmd(VB*n,name_hint='lnf'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*QT;xb=pl.cast(pl.slice(x,[QT,D],[qg,0]),target_type=pl.FP32)
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
                out=pl.assemble(out,v_l,[qg,v_c0])
    return out

@pl.jit.inline
def qkv_gemm_fixpipe(ln:pl.Tensor[[vTpE,D],pl.BF16],wq:pl.Tensor[[D,KO],pl.BF16],qbuf:pl.Tensor[[vTpE,KO],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # FixPipe split (same shape as v3g/v4g): pure cube, acc drains L0C->GM
    # through PIPE_FIX (FP32->BF16 in hardware, zero Vec), N-chunk QKN_Q=128
    # with K-chunk QKK_Q=128 ([128,128]x2Bx2stage = 64KB = Right wall). The
    # bias epilogue moved to qkv_bias_fixpipe (FixPipe pre-rounds acc to BF16
    # BEFORE the bias add — ~1 ULP drift vs the fused form, same as v3e/v4e).
    n=nqg_G
    for tg in pl.spmd(VB*n,name_hint='v1g'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ_G
            for qq_nb in pl.range(qkn_q):
                qq_n0=qq_nb*QKN_Q
                qq_a=pl.load(ln,[qg,0],[MQ_G,QKK_Q],target_memory=pl.MemorySpace.Mat)
                qq_b=pl.load(wq,[0,qq_n0],[QKK_Q,QKN_Q],target_memory=pl.MemorySpace.Mat)
                qq_acc=pl.matmul(pl.move(qq_a,target_memory=pl.MemorySpace.Left),pl.move(qq_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                for qq_kb in pl.pipeline(qkk_q-1, stage=2):
                    qq_k0=(qq_kb+1)*QKK_Q
                    qq_a=pl.load(ln,[qg,qq_k0],[MQ_G,QKK_Q],target_memory=pl.MemorySpace.Mat)
                    qq_b=pl.load(wq,[qq_k0,qq_n0],[QKK_Q,QKN_Q],target_memory=pl.MemorySpace.Mat)
                    qq_acc=pl.matmul_acc(qq_acc,pl.move(qq_a,target_memory=pl.MemorySpace.Left),pl.move(qq_b,target_memory=pl.MemorySpace.Right))
                qbuf=pl.store(qq_acc,[qg,qq_n0],qbuf,shapes=[MQ_G,QKN_Q])
    return qbuf

@pl.jit.inline
def qkv_bias_fixpipe(qbuf:pl.Tensor[[vTpE,KO],pl.BF16],bq:pl.Tensor[[1,KO],pl.BF16],out:pl.Tensor[[vTpE,KO],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # Bias epilogue pass: out = qbuf + bias in FP32, one rint cast — same add
    # nesting as the old fused epilogue (acc-value first, bias second).
    # N-chunk = RES_NQ (Vec-wall discipline, n384 lesson).
    n=vT//MQ
    for tg in pl.spmd(VB*n,name_hint='v1b'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ
            for qb_nb in pl.range(KO//RES_NQ):
                qb_n0=qb_nb*RES_NQ
                qb_pv=pl.cast(pl.slice(qbuf,[MQ,RES_NQ],[qg,qb_n0]),target_type=pl.FP32)
                qb_bt=pl.col_expand_mul(pl.full([MQ,RES_NQ],dtype=pl.FP32,value=1.0),pl.cast(pl.slice(bq,[1,RES_NQ],[0,qb_n0]),target_type=pl.FP32))
                qb_rs=pl.add(qb_pv,qb_bt)
                out=pl.assemble(out,pl.cast(qb_rs,target_type=pl.BF16,mode='rint'),[qg,qb_n0])
    return out

@pl.jit.inline
def mlp_f1_gemm_fixpipe(ln:pl.Tensor[[vTpE,D],pl.BF16],wf1:pl.Tensor[[D,ILP],pl.BF16],p1buf:pl.Tensor[[vTpE,ILP],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # FixPipe split (promoted flat6 recipe, generalized to the
    # N256 probe knobs): p1buf = ln@wf1, cube only. N-chunk 64 -> FP_N
    # (128/256), K-chunk QKK 256 -> FP_KC (128 at FP_N=128, 64 at 256;
    # Right [FP_KC,FP_N]x2Bx2stage = 64KB cap-exact). acc drains L0C->GM
    # via PIPE_FIX (zero Vec); the bias+gelu epilogue moved to
    # mlp_f1_gelu_fixpipe. K-trips 6->12 (128) / ->24 (256), per-element
    # FMA order unchanged (rounding-transparent).
    n=nqg_G
    for tg in pl.spmd(VB*n,name_hint='v4ag'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ_G
            for za_nb in pl.range(on_fpn_f1):
                za_n0=za_nb*FP_N
                za_a=pl.load(ln,[qg,0],[MQ_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                za_b=pl.load(wf1,[0,za_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                za_acc=pl.matmul(pl.move(za_a,target_memory=pl.MemorySpace.Left),pl.move(za_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                for za_kb in pl.pipeline(f1_kk-1,stage=2):
                    za_k0=(za_kb+1)*FP_KC
                    za_a=pl.load(ln,[qg,za_k0],[MQ_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    za_b=pl.load(wf1,[za_k0,za_n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    za_acc=pl.matmul_acc(za_acc,pl.move(za_a,target_memory=pl.MemorySpace.Left),pl.move(za_b,target_memory=pl.MemorySpace.Right))
                p1buf=pl.store(za_acc,[qg,za_n0],p1buf,shapes=[MQ_G,FP_N])
    return p1buf

@pl.jit.inline
def mlp_f1_gelu_fixpipe(p1buf:pl.Tensor[[vTpE,ILP],pl.BF16],bf1:pl.Tensor[[1,ILP],pl.BF16],g:pl.Tensor[[vTpE,ILP],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # gelu epilogue pass: p = p1buf + bias in FP32 (acc-value first, bias
    # second — the fused form's add nesting), t = p*_QG, sig = 1/(1+exp(-t)),
    # out = cast(p*sig, rint). Only delta vs the fused math: p1buf was
    # rounded to BF16 by the FixPipe BEFORE the bias add and the gelu
    # nonlinearity (gate-safe: L0 tight + loose PASS on both towers
    # ). N-chunk = RES_N so the FP32 [MQ,RES_N] tile set stays
    # off the 184KB Vec wall at any FP_N.
    n=vT//MQ
    for tg in pl.spmd(VB*n,name_hint='v4ae'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ
            for zb_nb in pl.range(on_res_f1):
                zb_n0=zb_nb*RES_N
                zb_pv=pl.add(pl.cast(pl.slice(p1buf,[MQ,RES_N],[qg,zb_n0]),target_type=pl.FP32),pl.col_expand_mul(pl.full([MQ,RES_N],dtype=pl.FP32,value=1.0),pl.cast(pl.slice(bf1,[1,RES_N],[0,zb_n0]),target_type=pl.FP32)))
                zb_t=pl.mul(zb_pv,_QG);zb_sig=pl.recip(pl.add(pl.exp(pl.neg(zb_t)),1.0))
                g=pl.assemble(g,pl.cast(pl.mul(zb_pv,zb_sig),target_type=pl.BF16,mode='rint'),[qg,zb_n0])
    return g

@pl.jit.inline
def mlp_f2(g:pl.Tensor[[vTe,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//MQ
    for tg in pl.spmd(VB*n,name_hint='v4b'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vT+rr*MQ;lr=pl.reshape(ls,[1,D])
            for v_nb in pl.range(on64):
                v_n0=v_nb*64
                v_f2=pl.matmul(pl.slice(g,[MQ,QN],[qg,0]),wf2[0:QN,v_n0:v_n0+64],out_dtype=pl.FP32)
                # K-loop pipelining (mlp_f1 recipe): the inn=68
                # K-chunk loop (K=8704, QN=128) amortizes the double-buffer
                # much better than mlp_f1's 6 chunks — the biggest K-pipeline
                # win of the four GEMMs. Body pure matmul_acc; store (out
                # assemble) stays in the N-loop.
                for v_ib in pl.pipeline(inn-1, stage=2):
                    v_i0=(v_ib+1)*QN
                    v_f2=pl.matmul_acc(v_f2,pl.slice(g,[MQ,QN],[qg,v_i0]),wf2[v_i0:v_i0+QN,v_n0:v_n0+64])
                v_lc=pl.slice(lr,[1,64],[0,v_n0]);v_r=pl.col_expand_mul(v_f2,v_lc)
                out=pl.assemble(out,pl.cast(v_r,target_type=pl.BF16,mode='rint'),[qg,v_n0])
    return out

# ── ONLINE-SOFTMAX B-axis attention body (Phase 1) — FLAT M=48 ──
# pe_s/mb_s/lb_s are CALLER-OWNED shared scratch (created once in per_rank,
# ping-pong by layer parity — WAR protection for the v2o->v2f GM round-trip).
# The full-fused qk_pv variant is BLOCKED on the toolchain: its AIV->AIC
# feedback pipe (tpush_to_aic, the only kernel in the codebase using it) is
# broken (83.9%/81.6% wrong at KTF=48/16 with byte-exact ptoas math), so the
# 3-phase split stays the only working design.
@pl.jit.inline
def vb2_all_os(qkv_full:pl.Tensor[[vTpE,KO],pl.BF16],cos:pl.Tensor[[vTp,HD],pl.FP32],sin:pl.Tensor[[vTp,HD],pl.FP32],attn_ctx:pl.Tensor[[vTpE,DLP],pl.BF16],pe_s:pl.Tensor[[VB*OI,vT],pl.BF16],mb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],lb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],active:pl.Scalar[pl.INDEX]):
    nq=vT//MQ;nk=vT//KT
    qr=pl.create_tensor([vTe,DL],dtype=pl.BF16);kr=pl.create_tensor([vTe,DL],dtype=pl.BF16);vp=pl.create_tensor([vTe,DLP],dtype=pl.BF16)
    # v2r: RoPE — flat spmd(VB*SLG), MQ=48 query rows per task (G=3 groups,
    # consecutive positions pos0=g*MQ); per-row cos/sin apply unchanged.
    # BUG-0002 zero pad kept.
    # a crop-block merge variant (spmd(SLG)+serial inner b loop,
    # same as the os tower's WIN there) was measured here and REVERTED: golden
    # PASS + bit-identical, but bench 1070.5/1091.7ms vs 1078.2ms baseline =
    # neutral. Mechanism: the os win is idle-block drain (per-block AIV
    # dispatch ~0.3us; at active=1 the os v2r drained 18928 idle blocks/layer);
    # the patch tower runs active=6=VB so every block is real — nothing idle to
    # harvest, and 6x-serialized block work trades against the dispatch saving.
    for rh in pl.spmd(VB*SLG,name_hint='v2r'):
        b=rh//SLG;rr=rh%SLG;g=rr//HL;h=rr%HL;qg=b*vT+g*MQ;qgf=b*vTp+g*MQ;pos0=g*MQ;h0=h*HD;hv0=h*HDP
        v2o=pl.full([MQ,HD],dtype=pl.FP32,value=1.0)
        v2col=pl.col_expand_mul(v2o,pl.cast(pl.arange(0,[1,HD],dtype=pl.INT32),target_type=pl.FP32))
        v2dup=pl.cast(pl.cast(pl.mul(v2col,0.5),target_type=pl.INT32,mode="trunc"),target_type=pl.FP32)
        v2lane=pl.sub(v2col,pl.mul(v2dup,2.0))
        v2swap=pl.cast(pl.sub(pl.add(v2col,1.0),pl.mul(v2lane,2.0)),target_type=pl.INT32)
        v2qf=pl.cast(pl.slice(qkv_full,[MQ,HD],[qgf,h0]),target_type=pl.FP32)
        v2qs=pl.gather(v2qf,dim=-1,index=v2swap)
        v2kf=pl.cast(pl.slice(qkv_full,[MQ,HD],[qgf,DL+h0]),target_type=pl.FP32)
        v2ks=pl.gather(v2kf,dim=-1,index=v2swap)
        # RoPE tile-computed, PER-ROW stored: the pure
        # elementwise q_orig*cos + q_swap*sin is computed on the whole [MQ,HD]
        # tile in one pass (48x fewer vector dispatches + 1 cos/sin [48,96]
        # load instead of 48 row loads), but qr/kr are written row-by-row.
        # The full-tile tstore variant measured -4.9% but FAILED golden (62%
        # mismatch, max diff 183): the single wide store breaks the v2r->v2o
        # qr/kr RAW ordering the runtime honors for the small per-row stores.
        # Hybrid keeps the safe write pattern: 134.6->131.4ms (-2.4%, PASS).
        ct=pl.slice(cos,[MQ,HD],[pos0,0]);st=pl.slice(sin,[MQ,HD],[pos0,0])
        v2qt=pl.add(pl.mul(v2qf,ct),pl.mul(v2qs,st));v2kt=pl.add(pl.mul(v2kf,ct),pl.mul(v2ks,st))
        for qi in pl.range(MQ):
            qr=pl.assemble(qr,pl.cast(pl.slice(v2qt,[1,HD],[qi,0]),target_type=pl.BF16,mode='rint'),[qg+qi,h0])
            kr=pl.assemble(kr,pl.cast(pl.slice(v2kt,[1,HD],[qi,0]),target_type=pl.BF16,mode='rint'),[qg+qi,h0])
        vr=pl.slice(qkv_full,[MQ,HD],[qgf,2*DL+h0]);vp=pl.assemble(vp,vr,[qg,hv0])
        vp=pl.assemble(vp,pl.full([MQ,HDP-HD],dtype=pl.BF16,value=0.0),[qg,hv0+HD])
    # v2o: SINGLE QK sweep, FLAT unguarded spmd(VB*SLG) — per-KV-block LOCAL
    # row max, local-max exp (<=1, no overflow) BF16 materialized to pe_s,
    # per-block (m_kb, l_kb) stats. NO loop-carried accumulator. KT=48 tiles
    # ([MQ,48] FP32 = 9KB) stay under the 188KB Vec wall at MQ=48.
    # With VB=6 and active=6 every lane is a real crop (no garbage rows).
    for fa in pl.spmd(VB*SLG,name_hint='v2o'):
        b=fa//SLG;rr=fa%SLG;qt=rr//HL;h=rr%HL;qg=b*vT+qt*MQ;h0=h*HD;ois=fa*MQ;sb=fa*NKB
        vo_qt=pl.slice(qr,[MQ,HD],[qg,h0])
        for kb in pl.range(nk):
            vo_kt=pl.slice(kr,[KT,HD],[b*vT+kb*KT,h0])
            vo_raw=pl.mul(pl.matmul(vo_qt,vo_kt,b_trans=True,out_dtype=pl.FP32),V_ATTN_SCALE)
            vo_m=pl.reshape(pl.row_max(vo_raw),[1,MQ])
            vo_e=pl.exp(pl.row_expand_sub(vo_raw,pl.reshape(vo_m,[MQ,1])))
            pe_s=pl.assemble(pe_s,pl.cast(vo_e,target_type=pl.BF16,mode="rint"),[ois,kb*KT])
            mb_s=pl.assemble(mb_s,vo_m,[sb+kb,0])
            lb_s=pl.assemble(lb_s,pl.reshape(pl.row_sum(vo_e),[1,MQ]),[sb+kb,0])
    # v2f: FLAT unguarded spmd(VB*SLG) — global max over per-block stats, then
    # PV with per-block rescale exp(m_kb-mi) on the matmul OUTPUT (vector
    # domain; pe_l stays a pure GM->cube slice), corrected denominators. PV N
    # (HDP=128) split into two 64-wide acc halves + direct half-writes: the
    # [MQ,128] ctx tile blows the 188KB Vec wall / ptoas tmov shape at MQ=48.
    for fa in pl.spmd(VB*SLG,name_hint='v2f'):
        b=fa//SLG;rr=fa%SLG;qt=rr//HL;h=rr%HL;qg=b*vTp+qt*MQ;hv0=h*HDP;ois=fa*MQ;sb=fa*NKB
        vf_mi=pl.full([1,MQ],dtype=pl.FP32,value=-3.0e38)
        for kb in pl.range(nk):
            vf_mi=pl.maximum(vf_mi,pl.slice(mb_s,[1,MQ],[sb+kb,0]))
        acc_o0=pl.full([MQ,HDP//2],dtype=pl.FP32,value=0.0);acc_o1=pl.full([MQ,HDP//2],dtype=pl.FP32,value=0.0)
        acc_l=pl.full([1,MQ],dtype=pl.FP32,value=0.0)
        for kb in pl.range(nk):
            vf_c=pl.exp(pl.sub(pl.slice(mb_s,[1,MQ],[sb+kb,0]),vf_mi))  # [1,MQ] in (0,1]
            pe_l=pl.slice(pe_s,[MQ,KT],[ois,kb*KT]);vt0=pl.slice(vp,[KT,HDP//2],[b*vT+kb*KT,hv0]);vt1=pl.slice(vp,[KT,HDP//2],[b*vT+kb*KT,hv0+HDP//2])
            acc_o0=pl.add(acc_o0,pl.row_expand_mul(pl.matmul(pe_l,vt0,out_dtype=pl.FP32),pl.reshape(vf_c,[MQ,1])))
            acc_o1=pl.add(acc_o1,pl.row_expand_mul(pl.matmul(pe_l,vt1,out_dtype=pl.FP32),pl.reshape(vf_c,[MQ,1])))
            acc_l=pl.add(acc_l,pl.mul(pl.slice(lb_s,[1,MQ],[sb+kb,0]),vf_c))
        ctx0=pl.row_expand_div(acc_o0,pl.reshape(acc_l,[MQ,1]))
        ctx1=pl.row_expand_div(acc_o1,pl.reshape(acc_l,[MQ,1]))
        attn_ctx=pl.assemble(attn_ctx,pl.cast(ctx0,target_type=pl.BF16,mode="rint"),[qg,hv0])
        attn_ctx=pl.assemble(attn_ctx,pl.cast(ctx1,target_type=pl.BF16,mode="rint"),[qg,hv0+HDP//2])
    return attn_ctx

vb2_in=pl.inline(vb2_all_os._func)
ln_fwd_in=pl.inline(ln_fwd._func);qkv_gemm_in=pl.inline(qkv_gemm_fixpipe._func);qkv_bias_in=pl.inline(qkv_bias_fixpipe._func)
mlp_f1_gemm_in=pl.inline(mlp_f1_gemm_fixpipe._func);mlp_f1_gelu_in=pl.inline(mlp_f1_gelu_fixpipe._func);mlp_f2_in=pl.inline(mlp_f2._func)

# ── vb3_out_proj FULL (B axis) — verbatim from the retired three-phase patch tower ──
@pl.jit.inline
def vb3_partial(ctx:pl.Tensor[[vTe,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//MQ
    for tg in pl.spmd(VB*n,name_hint='v3'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vT+rr*MQ;lr=pl.reshape(ls,[1,D])
            for nb in pl.range(on64):
                n0=nb*64
                acc=pl.matmul(pl.slice(ctx,[MQ,HDP],[qg,0]),pl.slice(wo,[HDP,64],[0,n0]),out_dtype=pl.FP32)
                # K-loop pipelining over the HL=16 head chunks (HDP=128
                # each) — same recipe as qkv/mlp_f1/mlp_f2.
                for hb in pl.pipeline(HL-1, stage=2):
                    ho=(hb+1)*HDP
                    acc=pl.matmul_acc(acc,pl.slice(ctx,[MQ,HDP],[qg,ho]),pl.slice(wo,[HDP,64],[ho,n0]))
                lc=pl.slice(lr,[1,64],[0,n0]);r=pl.col_expand_mul(acc,lc)
                out=pl.assemble(out,pl.cast(r,target_type=pl.BF16,mode='rint'),[qg,n0])
    return out
vb3f_in=pl.inline(vb3_partial._func)

# ── FUSED epilogue kernels ────────────────────────────────────────────────
# The tower is a strictly serial residual chain (removing any module saves its
# full time), so the remaining wins are module-to-module GM round-trips. vb3+r1
# and mlp_f2+r2 each remove a pbuf/p2buf write+read (47.8MB/layer) and a kernel
# launch. The intermediate value is reproduced bit-exactly via an on-chip BF16
# rint round-trip (the two-kernel path rounded it through GM BF16); the add
# nesting matches the original r1/r2 so FP32 association is unchanged.

# ── FIXPIPE SPLIT kernels ───────────────────────────────
# vb3r1_fused / mlp_f2_r2_fused are replaced by GEMM-only phases whose FP32
# accumulator drains straight to GM through the FixPipe (pl.store(acc) ->
# tstore_acc2gm on PIPE_FIX, BF16 conversion in hardware, zero Vec) plus an
# elementwise residual pass carrying the *ls scale + bias + residual add the
# fusion used to do on-chip. pbuf/p2buf are caller-owned (created once in
# per_rank, fully overwritten by the GEMM phase before the residual pass
# reads them — same reuse discipline as lnbuf).

@pl.jit.inline
def o_proj_gemm_fixpipe(ctx:pl.Tensor[[vTpE,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],pbuf:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # pbuf = ctx@wo — cube only. K-chunk generalized to FP_KC (128 at
    # FP_N<=128, 64 above — see header); the store moved OUT of the vector
    # domain entirely: FixPipe converts FP32->BF16 during the drain.
    n=nqg_G
    for tg in pl.spmd(VB*n,name_hint='v3g'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ_G
            for nb in pl.range(on_fpn):
                n0=nb*FP_N
                fg_a=pl.load(ctx,[qg,0],[MQ_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                fg_b=pl.load(wo,[0,n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                fg_acc=pl.matmul(pl.move(fg_a,target_memory=pl.MemorySpace.Left),pl.move(fg_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                for hb in pl.pipeline(v3g_kk-1, stage=2):
                    ho=(hb+1)*FP_KC
                    fg_a=pl.load(ctx,[qg,ho],[MQ_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    fg_b=pl.load(wo,[ho,n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    fg_acc=pl.matmul_acc(fg_acc,pl.move(fg_a,target_memory=pl.MemorySpace.Left),pl.move(fg_b,target_memory=pl.MemorySpace.Right))
                pbuf=pl.store(fg_acc,[qg,n0],pbuf,shapes=[MQ_G,FP_N])
    return pbuf

@pl.jit.inline
def o_proj_resid_fixpipe(pbuf:pl.Tensor[[vTpE,D],pl.BF16],rbb:pl.Tensor[[vTpE,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],bo:pl.Tensor[[1,D],pl.BF16],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # out = rbb + pbuf*ls + bo*ls — the r1 residual pass, add nesting
    # verbatim from the fused form (rbb + (pbuf + bt) in FP32, one rint
    # cast at the end). Only delta vs the fused math: pbuf was rounded to
    # BF16 by the FixPipe BEFORE the *ls scale (fused rounds after).
    n=vT//MQ
    for tg in pl.spmd(VB*n,name_hint='v3e'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ;lr=pl.reshape(ls,[1,D])
            for nb in pl.range(on_res):
                n0=nb*RES_N
                lc=pl.slice(lr,[1,RES_N],[0,n0]);bc=pl.slice(bo,[1,RES_N],[0,n0])
                fe_pv=pl.col_expand_mul(pl.cast(pl.slice(pbuf,[MQ,RES_N],[qg,n0]),target_type=pl.FP32),lc)
                fe_bt=pl.col_expand_mul(pl.full([MQ,RES_N],dtype=pl.FP32,value=1.0),pl.mul(pl.cast(bc,target_type=pl.FP32),lc))
                fe_rs=pl.add(pl.cast(pl.slice(rbb,[MQ,RES_N],[qg,n0]),target_type=pl.FP32),pl.add(fe_pv,fe_bt))
                out=pl.assemble(out,pl.cast(fe_rs,target_type=pl.BF16,mode='rint'),[qg,n0])
    return out

@pl.jit.inline
def mlp_f2_gemm_fixpipe(g:pl.Tensor[[vTpE,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],p2buf:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # p2buf = g@wf2 — cube only, FixPipe drain. K-chunk generalized to
    # FP_KC (QN=128 at FP_N<=128, 64 above — see header).
    n=nqg_G
    for tg in pl.spmd(VB*n,name_hint='v4g'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ_G
            for nb in pl.range(on_fpn):
                n0=nb*FP_N
                hg_a=pl.load(g,[qg,0],[MQ_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                hg_b=pl.load(wf2,[0,n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                hg_acc=pl.matmul(pl.move(hg_a,target_memory=pl.MemorySpace.Left),pl.move(hg_b,target_memory=pl.MemorySpace.Right),out_dtype=pl.FP32)
                for ib in pl.pipeline(v4g_kk-1, stage=2):
                    io=(ib+1)*FP_KC
                    hg_a=pl.load(g,[qg,io],[MQ_G,FP_KC],target_memory=pl.MemorySpace.Mat)
                    hg_b=pl.load(wf2,[io,n0],[FP_KC,FP_N],target_memory=pl.MemorySpace.Mat)
                    hg_acc=pl.matmul_acc(hg_acc,pl.move(hg_a,target_memory=pl.MemorySpace.Left),pl.move(hg_b,target_memory=pl.MemorySpace.Right))
                p2buf=pl.store(hg_acc,[qg,n0],p2buf,shapes=[MQ_G,FP_N])
    return p2buf

@pl.jit.inline
def mlp_f2_resid_fixpipe(p2buf:pl.Tensor[[vTpE,D],pl.BF16],x1buf:pl.Tensor[[vTpE,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],bf2:pl.Tensor[[1,D],pl.BF16],out:pl.Tensor[[vTpE,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # out = x1buf + p2buf*ls + bf2*ls — the r2 residual pass, same shape
    # as o_proj_resid_fixpipe.
    n=vT//MQ
    for tg in pl.spmd(VB*n,name_hint='v4e'):
        b=tg//n
        if b<active:
            rr=tg%n;qg=b*vTp+rr*MQ;lr=pl.reshape(ls,[1,D])
            for nb in pl.range(on_res):
                n0=nb*RES_N
                lc=pl.slice(lr,[1,RES_N],[0,n0]);bc=pl.slice(bf2,[1,RES_N],[0,n0])
                he_pv=pl.col_expand_mul(pl.cast(pl.slice(p2buf,[MQ,RES_N],[qg,n0]),target_type=pl.FP32),lc)
                he_bt=pl.col_expand_mul(pl.full([MQ,RES_N],dtype=pl.FP32,value=1.0),pl.mul(pl.cast(bc,target_type=pl.FP32),lc))
                he_rs=pl.add(pl.cast(pl.slice(x1buf,[MQ,RES_N],[qg,n0]),target_type=pl.FP32),pl.add(he_pv,he_bt))
                out=pl.assemble(out,pl.cast(he_rs,target_type=pl.BF16,mode='rint'),[qg,n0])
    return out

o_proj_gemm_in=pl.inline(o_proj_gemm_fixpipe._func)
o_proj_resid_in=pl.inline(o_proj_resid_fixpipe._func)
mlp_f2_gemm_in=pl.inline(mlp_f2_gemm_fixpipe._func)
mlp_f2_resid_in=pl.inline(mlp_f2_resid_fixpipe._func)

# RCH=V_LAYERS: one dispatch per tower (mirrors the global os tower). The
# earlier RCH=6 chunking was retired: single-dispatch (47 layers in
# one per_rank) is verified PASS on real hardware, with or without per-layer
# pl.scope() (1461.7/1465.5ms vs 1459.0/1462.9ms device/host) — the sweep-era
# "RCH>=8 heap wall" was non-resident host overhead, not heap backpressure.
RCH=V_LAYERS
RNCH=(V_LAYERS+RCH-1)//RCH


def _build_repl(tp_size:int=8, chunk:int=0):
    @pl.program
    class Step3p7VisionPatchReplOSFlat6:
        @pl.function(type=pl.FunctionType.Orchestration, attrs={"inline_orchestration": True})
        def v_attn(self,qf:pl.Tensor[[vTpE,KO],pl.BF16],cos:pl.Tensor[[vTp,HD],pl.FP32],sin:pl.Tensor[[vTp,HD],pl.FP32],ac:pl.Out[pl.Tensor[[vTpE,DLP],pl.BF16]],pe_s:pl.Tensor[[VB*OI,vT],pl.BF16],mb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],lb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],active:pl.Scalar[pl.INDEX])->pl.Tensor[[vTpE,DLP],pl.BF16]:ac=vb2_in(qf,cos,sin,ac,pe_s,mb_s,lb_s,active);return ac
        # auto_scope=False + explicit per-layer pl.scope(): mirrors the DSv4
        # prefill/decode discipline (models/deepseek_v4_flash_mtp/*.py). Each
        # runtime scope gets its own HeapRing level, so a scope exit reclaims
        # that layer's allocations (qf/ac/x1buf/rbb creates + the
        # InjectGMPipeBuffer workspaces inside the inlined GEMM pipelines)
        # instead of accumulating across all nl layers of one per_rank call.
        # Measured performance-neutral on the single-dispatch tower (A/B:
        # 1461.7/1465.5ms vs 1459.0/1462.9ms device/host) — kept as
        # defensive heap discipline, matching DSv4/step3.5-style programs.
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
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
            nq=vT//MQ
            # rbb must be vTp-strided (GEMM M-decoupling): the 4 GEMM
            # phases write 11 blocks of MQ_G=128 (1408 rows/image); a vT-strided out
            # would clobber image b+1's real rows. x_in is compact [vTe], so copy the
            # real rows once per tower into a vTp scratch (cp0).
            rbb=pl.create_tensor([vTpE,D],dtype=pl.BF16)
            for b in pl.parallel(VB):
                if b<active:
                    for c in pl.spmd(nq,name_hint='cp0'):
                        cq=b*vT+c*MQ;cw=b*vTp+c*MQ
                        rbb=pl.assemble(rbb,pl.slice(x_in,[MQ,D],[cq,0]),[cw,0])
            x1buf=pl.create_tensor([vTpE,D],dtype=pl.BF16)
            lnbuf=pl.create_tensor([vTpE,D],dtype=pl.BF16);gbuf=pl.create_tensor([vTpE,ILP],dtype=pl.BF16)
            # pbuf/p2buf are the FixPipe direct-store targets (GEMM acc -> GM).
            # Created once: each layer's GEMM phase overwrites every row it
            # owns before the residual pass reads it (same discipline as lnbuf).
            pbuf=pl.create_tensor([vTpE,D],dtype=pl.BF16);p2buf=pl.create_tensor([vTpE,D],dtype=pl.BF16);p1buf=pl.create_tensor([vTpE,ILP],dtype=pl.BF16)
            # qbuf is the v1g FixPipe direct-store target ([vTe,KO] = 71.7MB).
            # Created once: the GEMM phase overwrites every row it owns before
            # qkv_bias_fixpipe reads it (same discipline as pbuf -> v3e).
            qbuf=pl.create_tensor([vTpE,KO],dtype=pl.BF16)
            # pe_s/mb_s/lb_s are shared flash scratch, ping-pong by layer parity
            # (v2o of layer L writes the buffer v2f of layer L reads; layer L+1
            # writes the other half — WAR protection for the v2o->v2f round-trip).
            pe_s2=pl.create_tensor([2*VB*OI,vT],dtype=pl.BF16);mb_s2=pl.create_tensor([2*VB*SLG*NKB,MQ],dtype=pl.FP32);lb_s2=pl.create_tensor([2*VB*SLG*NKB,MQ],dtype=pl.FP32)
            for L in pl.range(nl):
                with pl.scope():
                    gi=lo+L;qo=gi*V_WIDTH;oo=gi*DLP;f1o=gi*V_WIDTH;f2o=gi*ILP
                    lq=pl.slice(rq,[V_WIDTH,KO],[qo,0]);lwo=pl.slice(ro,[DLP,V_WIDTH],[oo,0])
                    lbq=pl.slice(rbq,[1,KO],[gi,0]);lbo=pl.slice(rbo,[1,V_WIDTH],[gi,0])
                    lf1=pl.slice(rf1,[V_WIDTH,ILP],[f1o,0]);lf2=pl.slice(rf2,[ILP,V_WIDTH],[f2o,0])
                    lbf1=pl.slice(rbf1,[1,ILP],[gi,0]);lbf2=pl.slice(rbf2,[1,V_WIDTH],[gi,0])
                    ll1=pl.slice(rl1,[1,V_WIDTH],[gi,0]);ll2=pl.slice(rl2,[1,V_WIDTH],[gi,0])
                    ll1g=pl.slice(rl1g,[1,V_WIDTH],[gi,0]);ll1b=pl.slice(rl1b,[1,V_WIDTH],[gi,0])
                    ll2g=pl.slice(rl2g,[1,V_WIDTH],[gi,0]);ll2b=pl.slice(rl2b,[1,V_WIDTH],[gi,0])
                    qf=pl.create_tensor([vTpE,KO],dtype=pl.BF16);ac=pl.create_tensor([vTpE,DLP],dtype=pl.BF16)
                    lnbuf=ln_fwd_in(rbb,ll1g,ll1b,lnbuf,active)
                    qbuf=qkv_gemm_in(lnbuf,lq,qbuf,active)
                    qf=qkv_bias_in(qbuf,lbq,qf,active)
                    par=(L%2)*VB*OI;par2=(L%2)*VB*SLG*NKB
                    pe_s=pl.slice(pe_s2,[VB*OI,vT],[par,0]);mb_s=pl.slice(mb_s2,[VB*SLG*NKB,MQ],[par2,0]);lb_s=pl.slice(lb_s2,[VB*SLG*NKB,MQ],[par2,0])
                    ac=self.v_attn(qf,rcos,rsin,ac,pe_s,mb_s,lb_s,active)
                    x1buf=pl.create_tensor([vTpE,D],dtype=pl.BF16)
                    pbuf=o_proj_gemm_in(ac,lwo,pbuf,active)
                    x1buf=o_proj_resid_in(pbuf,rbb,ll1,lbo,x1buf,active)
                    lnbuf=ln_fwd_in(x1buf,ll2g,ll2b,lnbuf,active)
                    gbuf=mlp_f1_gemm_in(lnbuf,lf1,p1buf,active)
                    gbuf=mlp_f1_gelu_in(p1buf,lbf1,gbuf,active)
                    rbb=pl.create_tensor([vTpE,D],dtype=pl.BF16)
                    p2buf=mlp_f2_gemm_in(gbuf,lf2,p2buf,active)
                    rbb=mlp_f2_resid_in(p2buf,x1buf,ll2,lbf2,rbb,active)
            for _wr in pl.spmd(VB*nq,name_hint='wrto'):
                _wb=_wr//nq
                if _wb<active:
                    _wrr=_wb*vT+(_wr%nq)*MQ;_wrw=_wb*vTp+(_wr%nq)*MQ
                    rto=pl.assemble(rto,pl.slice(rbb,[MQ,D],[_wrw,0]),[_wrr,0])
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
            elif RNCH==1:
                # RCH>=V_LAYERS: the chunked ladder degenerates to one full
                # tower — collapse to a single per_rank. Without this guard
                # the first/middle/last ladder below would run the tower
                # TWICE (first(0,47) into hmid + last(0,47) into hto).
                for r in pl.range(pld.world_size()):
                    self.per_rank(hx_in[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hto[r],0,hnl,hactive,device=r)
            else:
                for r in pl.range(pld.world_size()):
                    self.per_rank(hx_in[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hmid[r],0,RCH,hactive,device=r)
                    for c in pl.range(1,RNCH-1):
                        self.per_rank(hmid[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hmid[r],c*RCH,RCH,hactive,device=r)
                    self.per_rank(hmid[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],hto[r],(RNCH-1)*RCH,V_LAYERS-(RNCH-1)*RCH,hactive,device=r)
    return Step3p7VisionPatchReplOSFlat6

Step3p7VisionPatchReplOSFlat6=_build_repl(8)
Step3p7VisionPatchReplOSFlat6Chunked=_build_repl(8,chunk=RCH)
