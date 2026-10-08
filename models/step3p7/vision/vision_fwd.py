# Copyright (c) PyPTO Contributors. SPDX-License-Identifier: Apache-2.0
# Step3.7 vision transformer (47 layers only). No patch_embed/ln_pre/downsampler/projector
# (those are vLLM's responsibility). Input = layer_00_layer_input, output = tower_out.
import pypto.language as pl; import pypto.language.distributed as pld
from .vision_config import (TP_WORLD_SIZE,V_LAYERS,V_TOKENS,V_WIDTH,V_WIDTH_LOCAL,V_EPS,V_WIDTH_INV,V_ATTN_SCALE,V_HEAD_DIM_PAD,V_WIDTH_LOCAL_PAD_V5,V_MLP_HIDDEN_LOCAL,V_MLP_HIDDEN_LOCAL_PAD)
D=V_WIDTH;DT=128;QT=16;QN=128;QKN=64;QKK=48;RQT=8  # RQT=8: RoPE tile (pl.gather bug at QT=16, use 8)
DL=V_WIDTH_LOCAL;HD=96;HL=2;IL=V_MLP_HIDDEN_LOCAL;ILP=V_MLP_HIDDEN_LOCAL_PAD;_QG=1.702
HDP=V_HEAD_DIM_PAD;DLP=V_WIDTH_LOCAL_PAD_V5;KO=3*DL
KT=208;NQM=V_TOKENS//QT;SL=NQM*HL;OI=SL*QT;HA=HD//2
vT=V_TOKENS;vWQ=V_LAYERS*V_WIDTH;vWF1=V_LAYERS*V_WIDTH;vWF2=V_LAYERS*ILP
qkn=KO//QKN;qkk=D//QKK;dn=D//DT;on=D//QN;inn=ILP//QN

# ── vb1: LN1+qkv+bias ──
@pl.jit.inline
def vb1_all(bb:pl.Tensor[[vT,D],pl.BF16],wq:pl.Tensor[[D,KO],pl.BF16],bq:pl.Tensor[[1,KO],pl.BF16],g1:pl.Tensor[[1,D],pl.FP32],b1:pl.Tensor[[1,D],pl.FP32],out:pl.Tensor[[vT,KO],pl.BF16]):
    n=vT//QT
    for tg in pl.spmd(n,name_hint='vb1'):
        qg=tg*QT;xb=pl.cast(pl.slice(bb,[QT,D],[qg,0]),target_type=pl.FP32)
        v_s=pl.full([1,QT],dtype=pl.FP32,value=0.0);v_s2=pl.full([1,QT],dtype=pl.FP32,value=0.0)
        for v_db in pl.range(dn):
            v_d0=v_db*DT;v_zc=pl.slice(xb,[QT,DT],[0,v_d0])
            v_s=pl.add(v_s,pl.reshape(pl.row_sum(v_zc),[1,QT]));v_s2=pl.add(v_s2,pl.reshape(pl.row_sum(pl.mul(v_zc,v_zc)),[1,QT]))
        v_m=pl.mul(v_s,V_WIDTH_INV);v_v=pl.sub(pl.mul(v_s2,V_WIDTH_INV),pl.mul(v_m,v_m))
        v_iv=pl.rsqrt(pl.add(v_v,V_EPS),high_precision=True);v_mt=pl.reshape(v_m,[QT,1]);v_it=pl.reshape(v_iv,[QT,1])
        for v_nb in pl.range(qkn):
            v_n0=v_nb*QKN;v_xk0=pl.slice(xb,[QT,QKK],[0,0])
            v_o0=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
            v_cs0=pl.sub(pl.row_expand_mul(v_xk0,v_it),pl.row_expand_mul(v_o0,pl.mul(v_mt,v_it)))
            v_gc=pl.slice(g1,[1,QKK],[0,0]);v_bc=pl.slice(b1,[1,QKK],[0,0])
            v_l0=pl.cast(pl.add(pl.col_expand_mul(v_cs0,v_gc),pl.col_expand_mul(v_o0,v_bc)),target_type=pl.BF16,mode='rint')
            v_acc=pl.matmul(v_l0,wq[0:QKK,v_n0:v_n0+QKN],out_dtype=pl.FP32)
            for v_kb in pl.range(1,qkk):
                v_k0=v_kb*QKK;v_xk=pl.slice(xb,[QT,QKK],[0,v_k0])
                v_ok=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
                v_ck=pl.sub(pl.row_expand_mul(v_xk,v_it),pl.row_expand_mul(v_ok,pl.mul(v_mt,v_it)))
                v_gk=pl.slice(g1,[1,QKK],[0,v_k0]);v_bk=pl.slice(b1,[1,QKK],[0,v_k0])
                v_lk=pl.cast(pl.add(pl.col_expand_mul(v_ck,v_gk),pl.col_expand_mul(v_ok,v_bk)),target_type=pl.BF16,mode='rint')
                v_acc=pl.matmul_acc(v_acc,v_lk,wq[v_k0:v_k0+QKK,v_n0:v_n0+QKN])
            v_bc2=pl.slice(bq,[1,QKN],[0,v_n0])
            v_acc=pl.add(v_acc,pl.col_expand_mul(pl.full([QT,QKN],dtype=pl.FP32,value=1.0),pl.cast(v_bc2,target_type=pl.FP32)))
            out=pl.assemble(out,pl.cast(v_acc,target_type=pl.BF16,mode='rint'),[qg,v_n0])
    return out

# ── vb2_attn ──
# Three-phase flash at KT=208. A vec-op-produced A operand fed DIRECTLY to a
# wide (K>128) cube matmul silently corrupts; GM-slice A operands are clean.
# So pe is materialized to GM between the softmax and PV phases. The online-merge
# am/al/ao scratch is replaced by a GLOBAL row max (v2g) + plain sum (exact
# softmax; per-block max with plain o/l sum is NOT softmax).
@pl.jit.inline
def vb2_all(qkv_full:pl.Tensor[[vT,KO],pl.BF16],cos:pl.Tensor[[vT,HD],pl.FP32],sin:pl.Tensor[[vT,HD],pl.FP32],attn_ctx:pl.Tensor[[vT,DLP],pl.BF16]):
    nq=vT//QT;nk=vT//KT
    qr=pl.create_tensor([vT,DL],dtype=pl.BF16);kr=pl.create_tensor([vT,DL],dtype=pl.BF16);vp=pl.create_tensor([vT,DLP],dtype=pl.BF16)
    pe_s=pl.create_tensor([OI,vT],dtype=pl.BF16);gm_s=pl.create_tensor([SL,QT],dtype=pl.FP32);ls_s=pl.create_tensor([SL,QT],dtype=pl.FP32)
    # RoPE: pl.spmd + inner pl.range per-position, col_expand_mul (DSv4 pattern).
    # col_expand_mul requires [1,N] 2nd arg → process 1 position per inner iter.
    # Inner pl.range iteration has ODD op count → PTOAS SyncSolver skips Loop
    # split (childNum % 2 != 0 → return) → all pairs isUseless=false → barriers
    # generated correctly. Also col_expand_mul = TCOLEXPANDMUL (fused, DSv4 same).
    for rh in pl.spmd((vT//QT)*HL,name_hint='v2r'):
        qg=(rh//HL)*QT;h=rh%HL;h0=h*HD;hv0=h*HDP
        v2o=pl.full([QT,HD],dtype=pl.FP32,value=1.0)
        v2col=pl.col_expand_mul(v2o,pl.cast(pl.arange(0,[1,HD],dtype=pl.INT32),target_type=pl.FP32))
        v2dup=pl.cast(pl.cast(pl.mul(v2col,0.5),target_type=pl.INT32,mode="trunc"),target_type=pl.FP32)
        v2lane=pl.sub(v2col,pl.mul(v2dup,2.0))
        v2swap=pl.cast(pl.sub(pl.add(v2col,1.0),pl.mul(v2lane,2.0)),target_type=pl.INT32)
        v2qf=pl.cast(pl.slice(qkv_full,[QT,HD],[qg,h0]),target_type=pl.FP32)
        v2qs=pl.gather(v2qf,dim=-1,index=v2swap)
        v2kf=pl.cast(pl.slice(qkv_full,[QT,HD],[qg,DL+h0]),target_type=pl.FP32)
        v2ks=pl.gather(v2kf,dim=-1,index=v2swap)
        for qi in pl.range(QT):
            ct1=pl.slice(cos,[1,HD],[qg+qi,0]);st1=pl.slice(sin,[1,HD],[qg+qi,0])
            q1=pl.slice(v2qf,[1,HD],[qi,0]);qs1=pl.slice(v2qs,[1,HD],[qi,0])
            qr=pl.assemble(qr,pl.cast(pl.add(pl.col_expand_mul(q1,ct1),pl.col_expand_mul(qs1,st1)),target_type=pl.BF16,mode='rint'),[qg+qi,h0])
            k1=pl.slice(v2kf,[1,HD],[qi,0]);ks1=pl.slice(v2ks,[1,HD],[qi,0])
            kr=pl.assemble(kr,pl.cast(pl.add(pl.col_expand_mul(k1,ct1),pl.col_expand_mul(ks1,st1)),target_type=pl.BF16,mode='rint'),[qg+qi,h0])
        vr=pl.slice(qkv_full,[QT,HD],[qg,2*DL+h0]);vp=pl.assemble(vp,vr,[qg,hv0])
        # Zero-fill the per-head pad columns (HD..HDP): v2f's PV matmul reads
        # the full [KT,HDP] slice, and unwritten pads read arena garbage —
        # stale NaN bits on repeated dispatches propagate NaN*0=NaN through
        # vb3's K-dim read (BUG-0002; same fix in vision_fwd_patch.py).
        vp=pl.assemble(vp,pl.full([QT,HDP-HD],dtype=pl.BF16,value=0.0),[qg,hv0+HD])
    # v2g: global per-row max of scaled scores (running max over wide QK).
    for gm in pl.spmd(nq*HL,name_hint='v2g'):
        qt=gm//HL;h=gm%HL;qg=qt*QT;h0=h*HD
        v2_qt=pl.slice(qr,[QT,HD],[qg,h0])
        gmax=pl.full([1,QT],dtype=pl.FP32,value=-3.0e38)
        for kb in pl.range(nk):
            kt=pl.slice(kr,[KT,HD],[kb*KT,h0])
            raw=pl.mul(pl.matmul(v2_qt,kt,b_trans=True,out_dtype=pl.FP32),V_ATTN_SCALE)
            gmax=pl.maximum(gmax,pl.reshape(pl.row_max(raw),[1,QT]))
        gm_s=pl.assemble(gm_s,gmax,[gm,0])
    # v2p: recompute QK, pe=exp(raw-gmax) materialized bf16 to GM + row sums.
    for fa in pl.spmd(nq*HL,name_hint='v2p'):
        qt=fa//HL;h=fa%HL;qg=qt*QT;h0=h*HD;ois=fa*QT
        v2_qt=pl.slice(qr,[QT,HD],[qg,h0])
        gm_col=pl.reshape(pl.slice(gm_s,[1,QT],[fa,0]),[QT,1])
        acc_l=pl.full([1,QT],dtype=pl.FP32,value=0.0)
        for kb in pl.range(nk):
            k0=kb*KT;kt=pl.slice(kr,[KT,HD],[k0,h0])
            raw=pl.mul(pl.matmul(v2_qt,kt,b_trans=True,out_dtype=pl.FP32),V_ATTN_SCALE)
            pe=pl.exp(pl.row_expand_sub(raw,gm_col))
            pe_s=pl.assemble(pe_s,pl.cast(pe,target_type=pl.BF16,mode="rint"),[ois,k0])
            acc_l=pl.add(acc_l,pl.reshape(pl.row_sum(pe),[1,QT]))
        ls_s=pl.assemble(ls_s,acc_l,[fa,0])
    # v2f: PV matmuls from the GM pe (pure GM->cube operand, clean at KT=208).
    for fa in pl.spmd(nq*HL,name_hint='v2f'):
        qt=fa//HL;h=fa%HL;qg=qt*QT;hv0=h*HDP;ois=fa*QT
        acc_o=pl.full([QT,HDP],dtype=pl.FP32,value=0.0)
        for kb in pl.range(nk):
            k0=kb*KT
            pe_l=pl.slice(pe_s,[QT,KT],[ois,k0]);vt=pl.slice(vp,[KT,HDP],[k0,hv0])
            acc_o=pl.add(acc_o,pl.matmul(pe_l,vt,out_dtype=pl.FP32))
        ml=pl.reshape(pl.slice(ls_s,[1,QT],[fa,0]),[QT,1]);ctx=pl.row_expand_div(acc_o,ml)
        attn_ctx=pl.assemble(attn_ctx,pl.cast(ctx,target_type=pl.BF16,mode="rint"),[qg,hv0])
    return attn_ctx

# ── vb3_out_proj partial ──
@pl.jit.inline
def vb3_partial(ctx:pl.Tensor[[vT,DLP],pl.BF16],wo:pl.Tensor[[DLP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],out:pl.Tensor[[vT,D],pl.BF16]):
    n=vT//QT
    for tg in pl.spmd(n,name_hint='v3'):
        qg=tg*QT;lr=pl.reshape(ls,[1,D])
        for nb in pl.range(on):
            n0=nb*QN
            c0=pl.matmul(pl.slice(ctx,[QT,HDP],[qg,0]),pl.slice(wo,[HDP,QN],[0,n0]),out_dtype=pl.FP32)
            c1=pl.matmul(pl.slice(ctx,[QT,HDP],[qg,HDP]),pl.slice(wo,[HDP,QN],[HDP,n0]),out_dtype=pl.FP32)
            lc=pl.slice(lr,[1,QN],[0,n0]);r=pl.col_expand_mul(pl.add(c0,c1),lc)
            out=pl.assemble(out,pl.cast(r,target_type=pl.BF16,mode='rint'),[qg,n0])
    return out

# ── vb4_mlp partial ──
@pl.jit.inline
def vb4_partial(x2:pl.Tensor[[vT,D],pl.BF16],wf1:pl.Tensor[[D,ILP],pl.BF16],bf1:pl.Tensor[[1,ILP],pl.BF16],wf2:pl.Tensor[[ILP,D],pl.BF16],ls:pl.Tensor[[D],pl.FP32],g2:pl.Tensor[[1,D],pl.FP32],b2:pl.Tensor[[1,D],pl.FP32],out:pl.Tensor[[vT,D],pl.BF16]):
    n=vT//QT
    for tg in pl.spmd(n,name_hint='v4'):
        qg=tg*QT;z0=pl.cast(pl.slice(x2,[QT,D],[qg,0]),target_type=pl.FP32)
        s=pl.full([1,QT],dtype=pl.FP32,value=0.0);s2=pl.full([1,QT],dtype=pl.FP32,value=0.0)
        for db in pl.range(dn):
            d0=db*DT;zc=pl.cast(pl.slice(x2,[QT,DT],[qg,d0]),target_type=pl.FP32)
            s=pl.add(s,pl.reshape(pl.row_sum(zc),[1,QT]));s2=pl.add(s2,pl.reshape(pl.row_sum(pl.mul(zc,zc)),[1,QT]))
        m=pl.mul(s,V_WIDTH_INV);v=pl.sub(pl.mul(s2,V_WIDTH_INV),pl.mul(m,m));iv=pl.rsqrt(pl.add(v,V_EPS),high_precision=True)
        mt=pl.reshape(m,[QT,1]);it=pl.reshape(iv,[QT,1]);lr=pl.reshape(ls,[1,D])
        for nb in pl.range(on):
            n0=nb*QN
            xk0=pl.cast(pl.slice(x2,[QT,QKK],[qg,0]),target_type=pl.FP32);o0=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
            cs0=pl.sub(pl.row_expand_mul(xk0,it),pl.row_expand_mul(o0,pl.mul(mt,it)))
            g0=pl.slice(g2,[1,QKK],[0,0]);b0=pl.slice(b2,[1,QKK],[0,0])
            l0=pl.cast(pl.add(pl.col_expand_mul(cs0,g0),pl.col_expand_mul(o0,b0)),target_type=pl.BF16,mode='rint')
            f1a=pl.matmul(l0,wf1[0:QKK,0:QN],out_dtype=pl.FP32)
            for kb in pl.range(1,qkk):
                k0=kb*QKK;xk=pl.cast(pl.slice(x2,[QT,QKK],[qg,k0]),target_type=pl.FP32);o=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
                cs=pl.sub(pl.row_expand_mul(xk,it),pl.row_expand_mul(o,pl.mul(mt,it)))
                gk=pl.slice(g2,[1,QKK],[0,k0]);bk=pl.slice(b2,[1,QKK],[0,k0])
                l=pl.cast(pl.add(pl.col_expand_mul(cs,gk),pl.col_expand_mul(o,bk)),target_type=pl.BF16,mode='rint')
                f1a=pl.matmul_acc(f1a,l,wf1[k0:k0+QKK,0:QN])
            bc=pl.slice(bf1,[1,QN],[0,0])
            f1a=pl.add(f1a,pl.col_expand_mul(pl.full([QT,QN],dtype=pl.FP32,value=1.0),pl.cast(bc,target_type=pl.FP32)))
            t=pl.mul(f1a,_QG);sig=pl.recip(pl.add(pl.exp(pl.neg(t)),1.0));g=pl.cast(pl.mul(f1a,sig),target_type=pl.BF16,mode='rint')
            f2a=pl.matmul(g,wf2[0:QN,n0:n0+QN],out_dtype=pl.FP32)
            for ib in pl.range(1,inn):
                i0=ib*QN
                xkb=pl.cast(pl.slice(x2,[QT,QKK],[qg,0]),target_type=pl.FP32);ob2=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
                csb=pl.sub(pl.row_expand_mul(xkb,it),pl.row_expand_mul(ob2,pl.mul(mt,it)))
                gb2=pl.slice(g2,[1,QKK],[0,0]);bb2=pl.slice(b2,[1,QKK],[0,0])
                lb=pl.cast(pl.add(pl.col_expand_mul(csb,gb2),pl.col_expand_mul(ob2,bb2)),target_type=pl.BF16,mode='rint')
                f1b=pl.matmul(lb,wf1[0:QKK,i0:i0+QN],out_dtype=pl.FP32)
                for kb2 in pl.range(1,qkk):
                    k02=kb2*QKK;xk2=pl.cast(pl.slice(x2,[QT,QKK],[qg,k02]),target_type=pl.FP32);o2=pl.full([QT,QKK],dtype=pl.FP32,value=1.0)
                    cs2=pl.sub(pl.row_expand_mul(xk2,it),pl.row_expand_mul(o2,pl.mul(mt,it)))
                    g2k=pl.slice(g2,[1,QKK],[0,k02]);b2k=pl.slice(b2,[1,QKK],[0,k02])
                    l2=pl.cast(pl.add(pl.col_expand_mul(cs2,g2k),pl.col_expand_mul(o2,b2k)),target_type=pl.BF16,mode='rint')
                    f1b=pl.matmul_acc(f1b,l2,wf1[k02:k02+QKK,i0:i0+QN])
                bcb=pl.slice(bf1,[1,QN],[0,i0])
                f1b=pl.add(f1b,pl.col_expand_mul(pl.full([QT,QN],dtype=pl.FP32,value=1.0),pl.cast(bcb,target_type=pl.FP32)))
                tb=pl.mul(f1b,_QG);sigb=pl.recip(pl.add(pl.exp(pl.neg(tb)),1.0))
                gb=pl.cast(pl.mul(f1b,sigb),target_type=pl.BF16,mode='rint')
                f2a=pl.matmul_acc(f2a,gb,wf2[i0:i0+QN,n0:n0+QN])
            lc=pl.slice(lr,[1,QN],[0,n0]);r=pl.col_expand_mul(f2a,lc)
            out=pl.assemble(out,pl.cast(r,target_type=pl.BF16,mode='rint'),[qg,n0])
    return out

vb1_in=pl.inline(vb1_all._func);vb2_in=pl.inline(vb2_all._func)
vb3f_in=pl.inline(vb3_partial._func);vb4f_in=pl.inline(vb4_partial._func)

def _build(tp_size:int=TP_WORLD_SIZE):
    @pl.program
    class Step3p7Vision:
        @pl.function(type=pl.FunctionType.Orchestration, attrs={"inline_orchestration": True})
        def v_qkv(self,bb:pl.Tensor[[vT,D],pl.BF16],wq:pl.Tensor[[D,KO],pl.BF16],bq:pl.Tensor[[1,KO],pl.BF16],g:pl.Tensor[[1,D],pl.FP32],b:pl.Tensor[[1,D],pl.FP32],qf:pl.Out[pl.Tensor[[vT,KO],pl.BF16]],mr:pl.Scalar[pl.INT32])->pl.Tensor[[vT,KO],pl.BF16]:qf=vb1_in(bb,wq,bq,g,b,qf);return qf
        @pl.function(type=pl.FunctionType.Orchestration, attrs={"inline_orchestration": True})
        def v_attn(self,qf:pl.Tensor[[vT,KO],pl.BF16],cos:pl.Tensor[[V_TOKENS,HD],pl.FP32],sin:pl.Tensor[[V_TOKENS,HD],pl.FP32],ac:pl.Out[pl.Tensor[[vT,DLP],pl.BF16]],mr:pl.Scalar[pl.INT32])->pl.Tensor[[vT,DLP],pl.BF16]:ac=vb2_in(qf,cos,sin,ac);return ac
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(self,local:pl.Tensor[[vT,D],pl.BF16],tw:pld.DistributedTensor[[vT,D],pl.BF16],sw:pld.DistributedTensor[[tp_size,2],pl.INT32],mr:pl.Scalar[pl.INT32],sc:pl.Scalar[pl.INT32])->pl.Tensor[[vT,D],pl.BF16]:
            gs=tp_size;shard=D//gs;n=vT//QT
            for s in pl.parallel(gs):
                if s!=mr:
                    so=s*shard
                    for aq in pl.range(n):
                        a0=aq*QT;t=pl.load(local,[a0,so],[QT,shard]);pl.store(t,[a0,so],tw)
            for peer in pl.parallel(gs):
                if peer!=mr:pld.system.notify(target=sw,peer=peer,offsets=[mr,sc],value=1,op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(gs):
                if src!=mr:pld.system.wait(signal=sw,offsets=[src,sc],expected=1,cmp=pld.WaitCmp.Ge)
            for owner in pl.range(gs):
                if owner==mr:
                    base=owner*shard
                    for aq in pl.range(n):
                        a0=aq*QT;own=pl.load(local,[a0,base],[QT,shard])
                        acc=pl.mul(pl.cast(own,target_type=pl.FP32),0.0)
                        for peer in pl.range(gs):
                            if peer==mr:acc=pl.add(acc,pl.cast(own,target_type=pl.FP32))
                            else:
                                rt=pld.tile.remote_load(tw,peer=peer,offsets=[a0,base],shape=[QT,shard])
                                acc=pl.add(acc,pl.cast(rt,target_type=pl.FP32))
                        red=pl.cast(acc,target_type=pl.BF16)
                        pl.store(red,[a0,base],local)
                        for dst in pl.range(gs):
                            if dst!=mr:pld.tile.remote_store(red,tw,dst,[a0,base])
            for peer in pl.parallel(gs):
                if peer!=mr:pld.system.notify(target=sw,peer=peer,offsets=[mr,sc],value=1,op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(gs):
                if src!=mr:pld.system.wait(signal=sw,offsets=[src,sc],expected=2,cmp=pld.WaitCmp.Ge)
            for r in pl.parallel(gs):
                off=r*shard
                if r!=mr:
                    for aq in pl.range(n):
                        a0=aq*QT;ps=pl.load(tw,[a0,off],[QT,shard]);pl.store(ps,[a0,off],local)
            for peer in pl.parallel(gs):
                if peer!=mr:pld.system.notify(target=sw,peer=peer,offsets=[mr,sc],value=1,op=pld.NotifyOp.AtomicAdd)
            for src in pl.parallel(gs):
                if src!=mr:pld.system.wait(signal=sw,offsets=[src,sc],expected=3,cmp=pld.WaitCmp.Ge)
            _zero=pl.cast(0,pl.INT32)
            for _clr in pl.range(gs):
                pl.write(sw,[_clr,sc],_zero)
            return local
        @pl.function(type=pl.FunctionType.Orchestration)
        def per_rank(self,
            x_in:pl.Tensor[[vT,D],pl.BF16],
            rq:pl.Tensor[[vWQ,KO],pl.BF16],ro:pl.Tensor[[V_LAYERS*DLP,V_WIDTH],pl.BF16],
            rbq:pl.Tensor[[V_LAYERS,KO],pl.BF16],rbo:pl.Tensor[[V_LAYERS,V_WIDTH],pl.BF16],
            rf1:pl.Tensor[[vWF1,ILP],pl.BF16],rf2:pl.Tensor[[vWF2,V_WIDTH],pl.BF16],
            rbf1:pl.Tensor[[V_LAYERS,ILP],pl.BF16],rbf2:pl.Tensor[[V_LAYERS,V_WIDTH],pl.BF16],
            rl1g:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],rl1b:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],
            rl2g:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],rl2b:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],
            rl1:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],rl2:pl.Tensor[[V_LAYERS,V_WIDTH],pl.FP32],
            rcos:pl.Tensor[[V_TOKENS,HD],pl.FP32],rsin:pl.Tensor[[V_TOKENS,HD],pl.FP32],
            tw:pld.DistributedTensor[[vT,D],pl.BF16],sw:pld.DistributedTensor[[tp_size,2],pl.INT32],
            rto:pl.Out[pl.Tensor[[vT,D],pl.BF16]],
            mr:pl.Scalar[pl.INT32],
            nl:pl.Scalar[pl.INT32],
        )->pl.Tensor[[vT,D],pl.BF16]:
            nq=vT//QT;rbb=x_in;pbuf=pl.create_tensor([vT,D],dtype=pl.BF16);x1buf=pl.create_tensor([vT,D],dtype=pl.BF16);p2buf=pl.create_tensor([vT,D],dtype=pl.BF16)
            for L in pl.range(nl):
                qo=L*V_WIDTH;oo=L*DLP;f1o=L*V_WIDTH;f2o=L*ILP
                lq=pl.slice(rq,[V_WIDTH,KO],[qo,0]);lwo=pl.slice(ro,[DLP,V_WIDTH],[oo,0])
                lbq=pl.slice(rbq,[1,KO],[L,0]);lbo=pl.slice(rbo,[1,V_WIDTH],[L,0])
                lf1=pl.slice(rf1,[V_WIDTH,ILP],[f1o,0]);lf2=pl.slice(rf2,[ILP,V_WIDTH],[f2o,0])
                lbf1=pl.slice(rbf1,[1,ILP],[L,0]);lbf2=pl.slice(rbf2,[1,V_WIDTH],[L,0])
                ll1=pl.slice(rl1,[1,V_WIDTH],[L,0]);ll2=pl.slice(rl2,[1,V_WIDTH],[L,0])
                ll1g=pl.slice(rl1g,[1,V_WIDTH],[L,0]);ll1b=pl.slice(rl1b,[1,V_WIDTH],[L,0])
                ll2g=pl.slice(rl2g,[1,V_WIDTH],[L,0]);ll2b=pl.slice(rl2b,[1,V_WIDTH],[L,0])
                qf=pl.create_tensor([vT,KO],dtype=pl.BF16);ac=pl.create_tensor([vT,DLP],dtype=pl.BF16)
                qf=self.v_qkv(rbb,lq,lbq,ll1g,ll1b,qf,mr)
                ac=self.v_attn(qf,rcos,rsin,ac,mr)
                pbuf=vb3f_in(ac,lwo,ll1,pbuf)
                if TP_WORLD_SIZE>1:pbuf=self.tp_all_reduce(pbuf,tw,sw,mr,0)
                x1buf=pl.create_tensor([vT,D],dtype=pl.BF16)
                for r1 in pl.spmd(nq,name_hint='r1'):
                    rq1=r1*QT
                    for rb_n in pl.range(on):
                        rn=rb_n*QN;r1o=pl.full([QT,QN],dtype=pl.FP32,value=1.0)
                        r1xc=pl.cast(pl.slice(rbb,[QT,QN],[rq1,rn]),target_type=pl.FP32)
                        r1po=pl.cast(pl.slice(pbuf,[QT,QN],[rq1,rn]),target_type=pl.FP32)
                        r1bc=pl.slice(lbo,[1,QN],[0,rn]);r1lc=pl.slice(ll1,[1,QN],[0,rn])
                        r1rs=pl.add(r1xc,pl.add(r1po,pl.col_expand_mul(r1o,pl.mul(pl.cast(r1bc,target_type=pl.FP32),r1lc))))
                        x1buf=pl.assemble(x1buf,pl.cast(r1rs,target_type=pl.BF16,mode='rint'),[rq1,rn])
                p2buf=vb4f_in(x1buf,lf1,lbf1,lf2,ll2,ll2g,ll2b,p2buf)
                if TP_WORLD_SIZE>1:p2buf=self.tp_all_reduce(p2buf,tw,sw,mr,1)
                rbb=pl.create_tensor([vT,D],dtype=pl.BF16)
                for r2 in pl.spmd(nq,name_hint='r2'):
                    rq2=r2*QT
                    for rb2_n in pl.range(on):
                        rn2=rb2_n*QN;r2o=pl.full([QT,QN],dtype=pl.FP32,value=1.0)
                        r2xc=pl.cast(pl.slice(x1buf,[QT,QN],[rq2,rn2]),target_type=pl.FP32)
                        r2po=pl.cast(pl.slice(p2buf,[QT,QN],[rq2,rn2]),target_type=pl.FP32)
                        r2bc=pl.slice(lbf2,[1,QN],[0,rn2]);r2lc=pl.slice(ll2,[1,QN],[0,rn2])
                        r2rs=pl.add(r2xc,pl.add(r2po,pl.col_expand_mul(r2o,pl.mul(pl.cast(r2bc,target_type=pl.FP32),r2lc))))
                        rbb=pl.assemble(rbb,pl.cast(r2rs,target_type=pl.BF16,mode='rint'),[rq2,rn2])
            for _wr in pl.spmd(nq,name_hint='wrto'):
                _wr0=_wr*QT;rto=pl.assemble(rto,pl.slice(rbb,[QT,D],[_wr0,0]),[_wr0,0])
            return rto
        @pl.function(level=pl.Level.HOST,role=pl.Role.Orchestrator)
        def host_orch(self,
            hx_in:pl.Tensor[[tp_size,vT,V_WIDTH],pl.BF16],
            hq:pl.Tensor[[tp_size,vWQ,KO],pl.BF16],ho:pl.Tensor[[tp_size,V_LAYERS*DLP,V_WIDTH],pl.BF16],
            hbq:pl.Tensor[[tp_size,V_LAYERS,KO],pl.BF16],hbo:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.BF16],
            hf1:pl.Tensor[[tp_size,vWF1,ILP],pl.BF16],hf2:pl.Tensor[[tp_size,vWF2,V_WIDTH],pl.BF16],
            hbf1:pl.Tensor[[tp_size,V_LAYERS,ILP],pl.BF16],hbf2:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.BF16],
            hl1g:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],hl1b:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],
            hl2g:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],hl2b:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],
            hl1:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],hl2:pl.Tensor[[tp_size,V_LAYERS,V_WIDTH],pl.FP32],
            hcos:pl.Tensor[[tp_size,V_TOKENS,HD],pl.FP32],hsin:pl.Tensor[[tp_size,V_TOKENS,HD],pl.FP32],
            hto:pl.Out[pl.Tensor[[tp_size,vT,V_WIDTH],pl.BF16]],
            hnl:pl.Scalar[pl.INT32],
        ):
            tb=pld.alloc_window_buffer(vT*V_WIDTH*2);sb=pld.alloc_window_buffer(tp_size*2*4)
            for r in pl.range(pld.world_size()):
                t=pld.window(tb,[vT,V_WIDTH],dtype=pl.BF16);s=pld.window(sb,[tp_size,2],dtype=pl.INT32)
                self.per_rank(hx_in[r],hq[r],ho[r],hbq[r],hbo[r],hf1[r],hf2[r],hbf1[r],hbf2[r],hl1g[r],hl1b[r],hl2g[r],hl2b[r],hl1[r],hl2[r],hcos[r],hsin[r],t,s,hto[r],r,hnl,device=r)
    return Step3p7Vision
Step3p7Vision=_build(TP_WORLD_SIZE)
