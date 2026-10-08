#!/usr/bin/env python3
"""Whole-network prefill precision at PREFILL_T=331.

Feeds VLLM golden layer-0 input hidden, runs all 45 layers in one program via
WholePrefillHolder, compares the final hidden_states against VLLM golden
layer-44 ffn_out.

Weight residency reuses the _stage_main_prefill exporter path (8 per-rank
exporters publish the IPC weight pool), then WholePrefillHolder imports it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # pypto-lib 根目录

CKPT = "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
VLLM_DUMP = "/mnt/hw910test-jfs/shenwx/pypto_jy_backup/golden_step3p5/prefill_golden_aime_P1P2P3"
HIDDEN = 4096
TP = 8
RAW_T = 331
PAD_T = ((RAW_T + 31) // 32) * 32  # 352


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0,1,2,3,4,5,6,7")
    p.add_argument("--out", default="/tmp/whole_net_align_331")
    p.add_argument("--rtol", type=float, default=5e-3)
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--mlp-rtol", type=float, default=8e-2)
    p.add_argument("--mlp-atol", type=float, default=2e-1)
    p.add_argument("--pass-rate", type=float, default=0.999)
    p.add_argument("--layer-hi", type=int, default=None,
                   help="run layers [0, layer_hi); default None -> 45")
    p.add_argument("--golden-layer", type=int, default=None,
                   help="golden ffn_out layer to compare; default layer_hi-1")
    p.add_argument("--pad-noise", type=float, default=0.0,
                   help="if >0, fill padding rows with randn*scale instead of zeros")
    p.add_argument("--save-actual", default=None,
                   help="save the RAW_T valid-row next_hidden to this .pt path")
    return p.parse_args()


def _start_exporters(out: Path, devices: list[int], ckpt: str):
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("ready.rank*"):
        path.unlink(missing_ok=True)
    for path in out.glob("*.done"):
        path.unlink(missing_ok=True)
    for pattern in ("pypto_weight.*", "ipc_heartbeat.*"):
        for path in out.glob(pattern):
            path.unlink(missing_ok=True)
    (out / "STOP").unlink(missing_ok=True)

    root = Path(__file__).resolve().parents[3]  # pypto-lib repo root (this clone)
    procs = []

    def launch(rank, dev):
        handle = open(out / f"export_rank{rank}.log", "w", encoding="utf-8")
        command = [
            sys.executable, "-m", "tests.step3p5.harnesses._stage_main_prefill_only",
            "--export-rank", str(rank),
            "--dev", str(dev),
            "--out", str(out),
            "--ckpt", ckpt,
            # Keep routed experts INT8 (W8A8). The default dequantizes to BF16,
            # which doubles the routed-expert weight pool to ~47 GiB and OOMs the
            # 64 GB HBM on a whole-network [0,45) run (see bf16bf16-prefill-oom).
            "--int8-routed",
        ]
        proc = subprocess.Popen(command, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT)
        setattr(proc, "_log_handle", handle)
        procs.append(proc)

    def ready(rank):
        return (out / f"ready.rank{rank}").exists() and (out / f"pypto_weight_map.rank{rank}.json.done").exists()

    def wait_ready(ranks, deadline):
        while time.time() < deadline:
            if all(ready(r) for r in ranks):
                return
            if any(p.poll() not in (None, 0) for p in procs):
                _stop_exporters(out, procs)
                raise RuntimeError("prefill exporter exited before readiness")
            time.sleep(3.0)
        _stop_exporters(out, procs)
        raise TimeoutError("prefill exporters not ready within 40 min")

    # BF16@BF16 dequantizes the routed experts (2x INT8 bytes) in host RAM, so
    # 8 concurrent exporters peak near the machine RAM ceiling and trip the OOM
    # killer. Load in two waves of TP//2 so the per-wave host-RAM peak stays
    # bounded; a ready exporter drops its host bundle and only holds the device
    # pool, so the wave-1 exporters no longer count against the wave-2 peak.
    deadline = time.time() + 2400.0
    wave = 2
    launch(0, devices[0])
    for rank in range(1, TP):
        launch(rank, devices[rank])
        if (rank + 1) % wave == 0 and rank + 1 < TP:
            wait_ready(range(rank + 1 - wave, rank + 1), deadline)
    wait_ready(range(TP), deadline)
    return procs


def _stop_exporters(out: Path, procs):
    try:
        (out / "STOP").write_text("1", encoding="utf-8")
    except OSError:
        pass
    for proc in procs:
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
        h = getattr(proc, "_log_handle", None)
        if h is not None:
            h.close()


def main():
    args = _parse_args()
    devices = [int(x) for x in args.device.split(",")]

    # PREFILL_T=331 env (before any config import).
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = "384"
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = "384"
    os.environ["PYPTO_STEP3P5_KV_NUM_LAYERS"] = "45"
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(45 * 3 * 128)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = "48"  # 3 blocks x storage batch 16
    # Opt-in weight cache: exporters reuse the per-layer test cache instead of
    # re-loading + slicing 26.2 GiB/rank from safetensors on every fresh prepare.
    os.environ["PYPTO_STEP3P5_WEIGHT_CACHE_DIR"] = "/mnt/hw910test-jfs/shenwx/step3p5_weights_cache_tp8"
    os.environ["PYPTO_STEP3P5_DUMP"] = "1"  # 精度测试需要 device dump 对比 golden

    import models.step3p5.prefill_qkv_proj_rope as pq
    pq.PREFILL_T = PAD_T
    pq.PREFILL_SEQ = PAD_T

    out = Path(args.out)
    print(f"=== whole-net prefill precision (PREFILL_T={RAW_T} -> pad {PAD_T}) ===", flush=True)
    print(f"  devices={devices} layer_hi={args.layer_hi or 45} golden_layer={args.golden_layer or (args.layer_hi or 45) - 1}", flush=True)

    layer_hi = args.layer_hi or 45
    golden_layer = args.golden_layer if args.golden_layer is not None else (layer_hi - 1)

    print("\n--- starting weight exporters ---", flush=True)
    procs = _start_exporters(out, devices, CKPT)
    print("  all ranks ready", flush=True)

    try:
        from tools.step3p5.whole_prefill_holder import WholePrefillHolder
        holder = WholePrefillHolder(
            device_ids=devices, out_dir=str(out), ckpt=CKPT, layer_hi=layer_hi,
        ).build()
        with holder:
            golden_input = torch.load(
                f"{VLLM_DUMP}/1_rank0_layer_00_layer_input.pt", map_location="cpu", weights_only=True,
            )["hidden_states"][:RAW_T].to(torch.bfloat16)
            if args.pad_noise > 0:
                g = torch.Generator().manual_seed(1234)
                pad_rows = (torch.randn(
                    PAD_T - RAW_T, HIDDEN, generator=g,
                    dtype=torch.float32,
                ) * args.pad_noise).to(torch.bfloat16)
            else:
                pad_rows = torch.zeros(PAD_T - RAW_T, HIDDEN, dtype=golden_input.dtype)
            golden_input_pad = torch.cat([golden_input, pad_rows])
            holder.set_prefill_input(golden_input_pad)

            t0 = time.time()
            result = holder.run()
            elapsed = time.time() - t0

            hidden = result["next_hidden"]  # [tp, PAD_T, HIDDEN]
            actual = hidden[0, :RAW_T, :].to(torch.bfloat16).float()
            if args.save_actual:
                torch.save(actual.to(torch.bfloat16), args.save_actual)
            golden_output = torch.load(
                f"{VLLM_DUMP}/1_rank0_layer_{golden_layer:02d}_ffn_out.pt", map_location="cpu", weights_only=True,
            )["hidden_states"][:RAW_T].to(torch.bfloat16).float()

            diff = (actual - golden_output).abs()
            max_abs = diff.max().item()
            close_attn = torch.isclose(actual, golden_output, rtol=args.rtol, atol=args.atol)
            close_mlp = torch.isclose(actual, golden_output, rtol=args.mlp_rtol, atol=args.mlp_atol)
            pass_rate_attn = close_attn.float().mean().item()
            pass_rate_mlp = close_mlp.float().mean().item()
            rel = diff / (golden_output.abs() + 1e-12)
            median_rel = rel.median().item()
            mean_rel = rel.mean().item()

            # Cosine similarity of the final hidden vs golden: overall (flattened
            # into one vector) and per-token (each [4096] hidden vector). This is
            # the "direction preservation" metric after 45 layers of accumulation.
            a_flat = actual.reshape(-1)
            g_flat = golden_output.reshape(-1)
            cos_overall = float((a_flat @ g_flat) / (a_flat.norm() * g_flat.norm() + 1e-12))
            cos_overall = max(-1.0, min(1.0, cos_overall))  # clamp fp overshoot

            dot_tok = (actual * golden_output).sum(dim=1)   # [RAW_T]
            na_tok = actual.norm(dim=1)
            ng_tok = golden_output.norm(dim=1)
            cos_per_token = (dot_tok / (na_tok * ng_tok + 1e-12)).clamp(-1.0, 1.0)
            cos_mean = cos_per_token.mean().item()
            cos_min = cos_per_token.min().item()
            cos_median = cos_per_token.median().item()
            cos_argmin = int(cos_per_token.argmin().item())

            # Token-exact judgement: close the last row through the offline CPU
            # tail (final RMSNorm + LM head + greedy argmax) on both device and
            # golden hidden states, then compare the predicted next token.
            from _tmp_prefill_token_exact import _tail_logits
            device_logits = _tail_logits(actual[-1], ckpt=CKPT)
            golden_logits = _tail_logits(golden_output[-1], ckpt=CKPT)
            device_token = int(device_logits.argmax().item())
            golden_token = int(golden_logits.argmax().item())
            token_exact = device_token == golden_token
            dv_vals, dv_idx = torch.topk(device_logits.float(), 5)
            gd_vals, gd_idx = torch.topk(golden_logits.float(), 5)
            device_margin = float(dv_vals[0] - dv_vals[1])
            golden_margin = float(gd_vals[0] - gd_vals[1])
            top5_jaccard = (
                len(set(dv_idx.tolist()) & set(gd_idx.tolist()))
                / len(set(dv_idx.tolist()) | set(gd_idx.tolist()))
            )
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(CKPT, fix_mistral_regex=True)
            device_text = tokenizer.decode([device_token])
            golden_text = tokenizer.decode([golden_token])

            status = "PASS" if token_exact else "FAIL"
            print(f"  whole-net: {status} | max_abs={max_abs:.4f} attn={pass_rate_attn:.6f} "
                  f"mlp={pass_rate_mlp:.6f} median_rel={median_rel:.6f} mean_rel={mean_rel:.6f} "
                  f"cos={cos_overall:.6f} cos_tok(mean/min/med)={cos_mean:.6f}/{cos_min:.6f}/{cos_median:.6f} "
                  f"cos_argmin_tok={cos_argmin} ({elapsed:.1f}s)", flush=True)
            print(f"  token-exact: device={device_token}({device_text!r}) golden={golden_token}({golden_text!r}) "
                  f"exact={token_exact} margin={device_margin:.4f}/{golden_margin:.4f} top5_jaccard={top5_jaccard:.4f}", flush=True)
    finally:
        _stop_exporters(out, procs)


if __name__ == "__main__":
    main()
