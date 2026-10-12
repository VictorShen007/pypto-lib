# Copyright (c) PyPTO Contributors. SPDX-License-Identifier: Apache-2.0
# Step3.7 PATCH REPL tower — ONLINE-SOFTMAX single-pass QK attention, FULL-FLAT experiment.
#
# Phase 1 of the DSV4-style attention rework, ported from
# vision_fwd_repl_os.py to the B-axis patch tower: the v2g (global
# row-max sweep) + v2p (QK recompute + exp) phases merge into ONE v2o pass
# with per-KV-block LOCAL max / local exp / local sum stats (mb_s/lb_s);
# v2f rescales each block's PV matmul OUTPUT by exp(m_kb - mi) in the vector
# domain (A operands stay pure GM->cube slices). Exact softmax math, QK
# sweeps 2 -> 1, and the loop-carried serial reduction chains (gmax/acc_l)
# that serialized the old K loops are gone (per-iteration independent stats).
#
# Barrier structure: v2r/v2o/v2f are ALL flat unguarded spmd (the full-flat
# port, 795.1->194.0ms 4.1x); query tile is M-AUGMENTED to MQ=48 (G=3 query
# groups per task, 1296=48*27 exact) so the QK/PV cube M jumps 16->48.
# Inactive-crop garbage rows are discarded by the guarded r1/r2/wrto.
#
# Everything else (v4 split bodies, vb3, residual adds, chunked host_orch,
# pe_s ping-pong WAR protection) is copied verbatim from
# the retired three-phase patch tower. The attention body is defined LOCALLY (not
# shared from vision_fwd_patch) so free vars resolve against this module's
# full-dim globals without ambiguity.
import pypto.language as pl; import pypto.language.distributed as pld
from .vision_config import (V_LAYERS,V_TOKENS_PATCH,V_PATCH_BATCH,V_WIDTH,V_EPS,V_WIDTH_INV,V_ATTN_SCALE,V_HEAD_DIM_PAD,V_HEADS,V_MLP_HIDDEN,V_MLP_HIDDEN_PAD,V_WIDTH_PAD)

# ── FULL-dim constants (same as the retired three-phase patch tower) ──
D=V_WIDTH;DT=128;QT=16;QN=128;QKN=64;QKK=256;RQT=8
DL=V_WIDTH;HD=96;HL=V_HEADS;IL=V_MLP_HIDDEN;ILP=V_MLP_HIDDEN_PAD;_QG=1.702
HDP=V_HEAD_DIM_PAD;DLP=V_WIDTH_PAD;KO=3*DL
KT=48;KTT=48;NKW=V_TOKENS_PATCH//KT;NKB=NKW;NQM=V_TOKENS_PATCH//QT;SL=NQM*HL;OI=SL*QT;HA=HD//2;G=3;MQ=G*QT;NGQM=NQM//G;SLG=NGQM*HL  # 1296=27*48 (KT=48 flash block: [MQ,48] tiles keep v2o under the 188KB Vec wall at MQ=48); G=3 groups/task -> MQ=48 (1296=48*27 exact, no pad)
vT=V_TOKENS_PATCH;VB=V_PATCH_BATCH;vTe=VB*vT  # VB=16 capacity + runtime `active` scalar (vision_config): any crop count <= 16 in one launch; > 16 chunked by the driver
vWQ=V_LAYERS*V_WIDTH;vWF1=V_LAYERS*V_WIDTH;vWF2=V_LAYERS*ILP
qkn=KO//QKN;qkk=D//QKK;dn=D//DT;on=D//QN;inn=ILP//QN;on64=D//64;in64=ILP//64  # N=64 output chunks for the epilogue-heavy GEMMs: at MQ=48 their N=128 tiles (24KB each, ~11 live) blow the 188KB Vec wall; N=64 halves every tile to 12KB so ~11 live fit (~132KB). N=32 was tried first (fits easily) but killed cube efficiency: 176.3->206.2ms

# ── v4 split bodies (B axis) — verbatim from the retired three-phase patch tower ──
@pl.jit.inline
def ln_fwd(x:pl.Tensor[[vTe,D],pl.BF16],g1:pl.Tensor[[1,D],pl.FP32],b1:pl.Tensor[[1,D],pl.FP32],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # QT=16 whole-D tile (the t24d512 dispatch-granularity rework was measured
    # NEUTRAL on the tower — patch ln is not dispatch-dominated (486 tasks vs
    # the front/back's 972 pre-sweet-spot) and LN is a ~1-2ms fraction of the
    # 134.7ms tower, so the granularity lever does not transfer from the B6
    # kernel; reverted to keep the proven form).
    n=vT//QT
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='lnf'):
                qg=b*vT+tg*QT;xb=pl.cast(pl.slice(x,[QT,D],[qg,0]),target_type=pl.FP32)
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
def qkv_gemm(ln:pl.Tensor[[vTe,D],pl.BF16],wq:pl.Tensor[[D,KO],pl.BF16],bq:pl.Tensor[[1,KO],pl.BF16],out:pl.Tensor[[vTe,KO],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//MQ
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='v1g'):
                qg=b*vT+tg*MQ
                for v_nb in pl.range(qkn):
                    v_n0=v_nb*QKN
                    v_acc=pl.matmul(pl.slice(ln,[MQ,QKK],[qg,0]),wq[0:QKK,v_n0:v_n0+QKN],out_dtype=pl.FP32)
                    for v_kb in pl.pipeline(qkk-1, stage=2):
                        v_k0=(v_kb+1)*QKK
                        v_acc=pl.matmul_acc(v_acc,pl.slice(ln,[MQ,QKK],[qg,v_k0]),wq[v_k0:v_k0+QKK,v_n0:v_n0+QKN])
                    v_bc2=pl.slice(bq,[1,QKN],[0,v_n0])
                    v_acc=pl.add(v_acc,pl.col_expand_mul(pl.full([MQ,QKN],dtype=pl.FP32,value=1.0),pl.cast(v_bc2,target_type=pl.FP32)))
                    out=pl.assemble(out,pl.cast(v_acc,target_type=pl.BF16,mode='rint'),[qg,v_n0])
    return out

@pl.jit.inline
def mlp_f1(ln:pl.Tensor[[vTe,D],pl.BF16],wf1:pl.Tensor[[D,ILP],pl.BF16],bf1:pl.Tensor[[1,ILP],pl.BF16],g:pl.Tensor[[vTe,ILP],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//MQ
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='v4a'):
                qg=b*vT+tg*MQ
                for v_ib in pl.range(in64):
                    v_i0=v_ib*64
                    v_f1=pl.matmul(pl.slice(ln,[MQ,QKK],[qg,0]),wf1[0:QKK,v_i0:v_i0+64],out_dtype=pl.FP32)
                    # K-loop pipelining (front/back ds1/ds2 recipe): peel the
                    # K=0 init chunk (pl.matmul), then
                    # pl.pipeline(stage=2) over chunks 1..qkk-1 — double-buffers
                    # the next (ln, wf1) K-chunk loads under cube compute. Body
                    # is pure matmul_acc (no store: the g assemble stays in the
                    # N-loop, which holds the store so stays pl.range per the
                    # rule "loops with stores can't pl.pipeline").
                    # Same chunks, same order -> K accumulation order unchanged
                    # (bit-exact).
                    for v_kb in pl.pipeline(qkk-1, stage=2):
                        v_k0=(v_kb+1)*QKK
                        v_f1=pl.matmul_acc(v_f1,pl.slice(ln,[MQ,QKK],[qg,v_k0]),wf1[v_k0:v_k0+QKK,v_i0:v_i0+64])
                    v_bc1=pl.slice(bf1,[1,64],[0,v_i0])
                    v_f1=pl.add(v_f1,pl.col_expand_mul(pl.full([MQ,64],dtype=pl.FP32,value=1.0),pl.cast(v_bc1,target_type=pl.FP32)))
                    v_t=pl.mul(v_f1,_QG);v_sig=pl.recip(pl.add(pl.exp(pl.neg(v_t)),1.0))
                    g=pl.assemble(g,pl.cast(pl.mul(v_f1,v_sig),target_type=pl.BF16,mode='rint'),[qg,v_i0])
    return g

@pl.jit.inline
def mlp_f2(g:pl.Tensor[[vTe,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//MQ
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='v4b'):
                qg=b*vT+tg*MQ;lr=pl.reshape(ls,[1,D])
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
def vb2_all_os(qkv_full:pl.Tensor[[vTe,KO],pl.BF16],cos:pl.Tensor[[V_TOKENS_PATCH,HD],pl.FP32],sin:pl.Tensor[[V_TOKENS_PATCH,HD],pl.FP32],attn_ctx:pl.Tensor[[vTe,DLP],pl.BF16],pe_s:pl.Tensor[[VB*OI,vT],pl.BF16],mb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],lb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],active:pl.Scalar[pl.INDEX]):
    nq=vT//MQ;nk=vT//KT
    qr=pl.create_tensor([vTe,DL],dtype=pl.BF16);kr=pl.create_tensor([vTe,DL],dtype=pl.BF16);vp=pl.create_tensor([vTe,DLP],dtype=pl.BF16)
    # v2r: RoPE — flat spmd(VB*SLG), MQ=48 query rows per task (G=3 groups,
    # consecutive positions pos0=g*MQ); per-row cos/sin apply unchanged.
    # BUG-0002 zero pad kept.
    for rh in pl.spmd(VB*SLG,name_hint='v2r'):
        b=rh//SLG;rr=rh%SLG;g=rr//HL;h=rr%HL;qg=b*vT+g*MQ;pos0=g*MQ;h0=h*HD;hv0=h*HDP
        v2o=pl.full([MQ,HD],dtype=pl.FP32,value=1.0)
        v2col=pl.col_expand_mul(v2o,pl.cast(pl.arange(0,[1,HD],dtype=pl.INT32),target_type=pl.FP32))
        v2dup=pl.cast(pl.cast(pl.mul(v2col,0.5),target_type=pl.INT32,mode="trunc"),target_type=pl.FP32)
        v2lane=pl.sub(v2col,pl.mul(v2dup,2.0))
        v2swap=pl.cast(pl.sub(pl.add(v2col,1.0),pl.mul(v2lane,2.0)),target_type=pl.INT32)
        v2qf=pl.cast(pl.slice(qkv_full,[MQ,HD],[qg,h0]),target_type=pl.FP32)
        v2qs=pl.gather(v2qf,dim=-1,index=v2swap)
        v2kf=pl.cast(pl.slice(qkv_full,[MQ,HD],[qg,DL+h0]),target_type=pl.FP32)
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
        vr=pl.slice(qkv_full,[MQ,HD],[qg,2*DL+h0]);vp=pl.assemble(vp,vr,[qg,hv0])
        vp=pl.assemble(vp,pl.full([MQ,HDP-HD],dtype=pl.BF16,value=0.0),[qg,hv0+HD])
    # v2o: SINGLE QK sweep, FLAT unguarded spmd(VB*SLG) — per-KV-block LOCAL
    # row max, local-max exp (<=1, no overflow) BF16 materialized to pe_s,
    # per-block (m_kb, l_kb) stats. NO loop-carried accumulator. KT=48 tiles
    # ([MQ,48] FP32 = 9KB) stay under the 188KB Vec wall at MQ=48.
    # Inactive-crop garbage rows are written but only read by the equally
    # unguarded v2f; its inactive rows are discarded by guarded vb3.
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
        b=fa//SLG;rr=fa%SLG;qt=rr//HL;h=rr%HL;qg=b*vT+qt*MQ;hv0=h*HDP;ois=fa*MQ;sb=fa*NKB
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
ln_fwd_in=pl.inline(ln_fwd._func);qkv_gemm_in=pl.inline(qkv_gemm._func)
mlp_f1_in=pl.inline(mlp_f1._func);mlp_f2_in=pl.inline(mlp_f2._func)

# ── vb3_out_proj FULL (B axis) — verbatim from the retired three-phase patch tower ──
@pl.jit.inline
def vb3_partial(ctx:pl.Tensor[[vTe,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    n=vT//MQ
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='v3'):
                qg=b*vT+tg*MQ;lr=pl.reshape(ls,[1,D])
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

@pl.jit.inline
def vb3r1_fused(ctx:pl.Tensor[[vTe,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],rbb:pl.Tensor[[vTe,D],pl.BF16],bo:pl.Tensor[[1,D],pl.BF16],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # out = rbb + (ctx@wo)*ls + bo*ls (vb3 cube part + r1 residual add in one).
    n=vT//MQ
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='v3r'):
                fxqg=b*vT+tg*MQ;fxlr=pl.reshape(ls,[1,D])
                for fxnb in pl.range(on64):
                    fxn0=fxnb*64
                    fxacc=pl.matmul(pl.slice(ctx,[MQ,HDP],[fxqg,0]),pl.slice(wo,[HDP,64],[0,fxn0]),out_dtype=pl.FP32)
                    for fxhb in pl.pipeline(HL-1, stage=2):
                        fxho=(fxhb+1)*HDP
                        fxacc=pl.matmul_acc(fxacc,pl.slice(ctx,[MQ,HDP],[fxqg,fxho]),pl.slice(wo,[HDP,64],[fxho,fxn0]))
                    fxlcl=pl.slice(fxlr,[1,64],[0,fxn0]);fxbc=pl.slice(bo,[1,64],[0,fxn0])
                    fxp=pl.cast(pl.col_expand_mul(fxacc,fxlcl),target_type=pl.BF16,mode='rint')  # = pbuf value
                    fxbt=pl.col_expand_mul(pl.full([MQ,64],dtype=pl.FP32,value=1.0),pl.mul(pl.cast(fxbc,target_type=pl.FP32),fxlcl))
                    fxrs=pl.add(pl.cast(pl.slice(rbb,[MQ,64],[fxqg,fxn0]),target_type=pl.FP32),pl.add(pl.cast(fxp,target_type=pl.FP32),fxbt))
                    out=pl.assemble(out,pl.cast(fxrs,target_type=pl.BF16,mode='rint'),[fxqg,fxn0])
    return out
vb3r1_in=pl.inline(vb3r1_fused._func)

@pl.jit.inline
def mlp_f2_r2_fused(g:pl.Tensor[[vTe,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],x1buf:pl.Tensor[[vTe,D],pl.BF16],bf2:pl.Tensor[[1,D],pl.BF16],out:pl.Tensor[[vTe,D],pl.BF16],active:pl.Scalar[pl.INDEX]):
    # out = x1buf + (g@wf2)*ls + bf2*ls (mlp_f2 cube part + r2 residual add in one).
    n=vT//MQ
    for b in pl.parallel(VB):
        if b<active:
            for tg in pl.spmd(n,name_hint='v4r'):
                fyqg=b*vT+tg*MQ;fylr=pl.reshape(ls,[1,D])
                for fynb in pl.range(on64):
                    fyn0=fynb*64
                    fyacc=pl.matmul(pl.slice(g,[MQ,QN],[fyqg,0]),wf2[0:QN,fyn0:fyn0+64],out_dtype=pl.FP32)
                    for fyib in pl.pipeline(inn-1, stage=2):
                        fyio=(fyib+1)*QN
                        fyacc=pl.matmul_acc(fyacc,pl.slice(g,[MQ,QN],[fyqg,fyio]),wf2[fyio:fyio+QN,fyn0:fyn0+64])
                    fylc=pl.slice(fylr,[1,64],[0,fyn0]);fybc=pl.slice(bf2,[1,64],[0,fyn0])
                    fyp=pl.cast(pl.col_expand_mul(fyacc,fylc),target_type=pl.BF16,mode='rint')  # = p2buf value
                    fybt=pl.col_expand_mul(pl.full([MQ,64],dtype=pl.FP32,value=1.0),pl.mul(pl.cast(fybc,target_type=pl.FP32),fylc))
                    fyrs=pl.add(pl.cast(pl.slice(x1buf,[MQ,64],[fyqg,fyn0]),target_type=pl.FP32),pl.add(pl.cast(fyp,target_type=pl.FP32),fybt))
                    out=pl.assemble(out,pl.cast(fyrs,target_type=pl.BF16,mode='rint'),[fyqg,fyn0])
    return out
mlp_f2_r2_in=pl.inline(mlp_f2_r2_fused._func)

RCH=4
RNCH=(V_LAYERS+RCH-1)//RCH


def _build_repl(tp_size:int=8, chunk:int=0):
    @pl.program
    class Step3p7VisionPatchReplOSFlat:
        @pl.function(type=pl.FunctionType.Orchestration, attrs={"inline_orchestration": True})
        def v_attn(self,qf:pl.Tensor[[vTe,KO],pl.BF16],cos:pl.Tensor[[V_TOKENS_PATCH,HD],pl.FP32],sin:pl.Tensor[[V_TOKENS_PATCH,HD],pl.FP32],ac:pl.Out[pl.Tensor[[vTe,DLP],pl.BF16]],pe_s:pl.Tensor[[VB*OI,vT],pl.BF16],mb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],lb_s:pl.Tensor[[VB*SLG*NKB,MQ],pl.FP32],active:pl.Scalar[pl.INDEX])->pl.Tensor[[vTe,DLP],pl.BF16]:ac=vb2_in(qf,cos,sin,ac,pe_s,mb_s,lb_s,active);return ac
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
            rcos:pl.Tensor[[V_TOKENS_PATCH,HD],pl.FP32],rsin:pl.Tensor[[V_TOKENS_PATCH,HD],pl.FP32],
            rto:pl.Out[pl.Tensor[[vTe,D],pl.BF16]],
            lo:pl.Scalar[pl.INT32],
            nl:pl.Scalar[pl.INT32],
            active:pl.Scalar[pl.INDEX],
        )->pl.Tensor[[vTe,D],pl.BF16]:
            nq=vT//MQ;rbb=x_in;x1buf=pl.create_tensor([vTe,D],dtype=pl.BF16)
            lnbuf=pl.create_tensor([vTe,D],dtype=pl.BF16);gbuf=pl.create_tensor([vTe,ILP],dtype=pl.BF16)
            # pe_s/mb_s/lb_s are shared flash scratch, ping-pong by layer parity
            # (v2o of layer L writes the buffer v2f of layer L reads; layer L+1
            # writes the other half — WAR protection for the v2o->v2f round-trip).
            pe_s2=pl.create_tensor([2*VB*OI,vT],dtype=pl.BF16);mb_s2=pl.create_tensor([2*VB*SLG*NKB,MQ],dtype=pl.FP32);lb_s2=pl.create_tensor([2*VB*SLG*NKB,MQ],dtype=pl.FP32)
            for L in pl.range(nl):
                gi=lo+L;qo=gi*V_WIDTH;oo=gi*DLP;f1o=gi*V_WIDTH;f2o=gi*ILP
                lq=pl.slice(rq,[V_WIDTH,KO],[qo,0]);lwo=pl.slice(ro,[DLP,V_WIDTH],[oo,0])
                lbq=pl.slice(rbq,[1,KO],[gi,0]);lbo=pl.slice(rbo,[1,V_WIDTH],[gi,0])
                lf1=pl.slice(rf1,[V_WIDTH,ILP],[f1o,0]);lf2=pl.slice(rf2,[ILP,V_WIDTH],[f2o,0])
                lbf1=pl.slice(rbf1,[1,ILP],[gi,0]);lbf2=pl.slice(rbf2,[1,V_WIDTH],[gi,0])
                ll1=pl.slice(rl1,[1,V_WIDTH],[gi,0]);ll2=pl.slice(rl2,[1,V_WIDTH],[gi,0])
                ll1g=pl.slice(rl1g,[1,V_WIDTH],[gi,0]);ll1b=pl.slice(rl1b,[1,V_WIDTH],[gi,0])
                ll2g=pl.slice(rl2g,[1,V_WIDTH],[gi,0]);ll2b=pl.slice(rl2b,[1,V_WIDTH],[gi,0])
                qf=pl.create_tensor([vTe,KO],dtype=pl.BF16);ac=pl.create_tensor([vTe,DLP],dtype=pl.BF16)
                lnbuf=ln_fwd_in(rbb,ll1g,ll1b,lnbuf,active)
                qf=qkv_gemm_in(lnbuf,lq,lbq,qf,active)
                par=(L%2)*VB*OI;par2=(L%2)*VB*SLG*NKB
                pe_s=pl.slice(pe_s2,[VB*OI,vT],[par,0]);mb_s=pl.slice(mb_s2,[VB*SLG*NKB,MQ],[par2,0]);lb_s=pl.slice(lb_s2,[VB*SLG*NKB,MQ],[par2,0])
                ac=self.v_attn(qf,rcos,rsin,ac,pe_s,mb_s,lb_s,active)
                x1buf=pl.create_tensor([vTe,D],dtype=pl.BF16)
                x1buf=vb3r1_in(ac,lwo,ll1,rbb,lbo,x1buf,active)
                lnbuf=ln_fwd_in(x1buf,ll2g,ll2b,lnbuf,active)
                gbuf=mlp_f1_in(lnbuf,lf1,lbf1,gbuf,active)
                rbb=pl.create_tensor([vTe,D],dtype=pl.BF16)
                rbb=mlp_f2_r2_in(gbuf,lf2,ll2,x1buf,lbf2,rbb,active)
            for b in pl.parallel(VB):
                if b<active:
                    for _wr in pl.spmd(nq,name_hint='wrto'):
                        _wr0=b*vT+_wr*MQ;rto=pl.assemble(rto,pl.slice(rbb,[MQ,D],[_wr0,0]),[_wr0,0])
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
            hcos:pl.Tensor[[tp_size,V_TOKENS_PATCH,HD],pl.FP32],hsin:pl.Tensor[[tp_size,V_TOKENS_PATCH,HD],pl.FP32],
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
    return Step3p7VisionPatchReplOSFlat

Step3p7VisionPatchReplOSFlat=_build_repl(8)
Step3p7VisionPatchReplOSFlatChunked=_build_repl(8,chunk=RCH)
