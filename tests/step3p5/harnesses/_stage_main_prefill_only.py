# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resident prefill performance diagnostic — TTFT scaling vs sequence length.

Mirrors ``_stage_main_hidden_only.py``'s perf-only ITL: for each length in
``--ttft-seq-lens``, re-launch a per-length subprocess (``PREFILL_T`` is a
compile-time constant, so each length needs its own compilation), rebuild the
resident ``WholePrefillHolder``, warm up, then time ``holder.run()`` over
``--ttft-iters``. Aggregates per-length stats into ``ttft_report.json``.

Reuses the CACHE_DIR weight cache (``PYPTO_STEP3P5_WEIGHT_CACHE_DIR``) so the
per-length exporter weight load is seconds, not minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CKPT = "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
TP = 8
BLOCK_SIZE = 128
SEED_TOKEN = 1
HIDDEN = 4096


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="8,9,10,11,12,13,14,15")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--out", required=True)
    p.add_argument("--export-rank", type=int, default=-1)
    p.add_argument("--dev", type=int, default=8)
    p.add_argument("--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    p.add_argument(
        "--int8-routed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "default True (W8A8): matches the vLLM golden and hits the weight "
            "cache; pass --no-int8-routed to dequantize routed experts to BF16 "
            "(doubles the routed-expert pool, may OOM)."
        ),
    )
    p.add_argument(
        "--layer-lo",
        type=int,
        default=0,
        help="inclusive lower layer index (exporter pre-slices MoE stack; default 0)",
    )
    p.add_argument(
        "--layer-hi",
        type=int,
        default=None,
        help="exclusive upper layer index (default None = all layers)",
    )
    p.add_argument(
        "--ttft-seq-lens",
        default="",
        help="perf-only: comma list of prefill seq lengths (e.g. 128,512,1024,2048)",
    )
    p.add_argument(
        "--ttft-seq-len",
        type=int,
        default=None,
        help="internal: single-length measurement (subprocess self-invocation)",
    )
    p.add_argument("--ttft-iters", type=int, default=10)
    p.add_argument("--ttft-warmup", type=int, default=2)
    p.add_argument("--seed-token", type=int, default=SEED_TOKEN)
    return p.parse_args()


def _devices(text: str) -> list[int]:
    devices = [int(item) for item in str(text).split(",") if item.strip()]
    if len(devices) != TP or len(set(devices)) != TP:
        raise ValueError(f"expected {TP} distinct devices, got {devices}")
    return devices


# ---------------------------------------------------------------------------
# Exporter (mirrors _stage_main_prefill.py _export_rank / _start_exporters).
# ---------------------------------------------------------------------------
def _export_rank(args: argparse.Namespace) -> int:
    if not 0 <= args.export_rank < TP:
        raise ValueError(f"export rank must be 0..{TP - 1}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from tools.step3p5.pypto_weight_ipc import export_from_checkpoint_resident

    weight_owner, weight_summary, _ = export_from_checkpoint_resident(
        args.ckpt,
        rank=args.export_rank,
        tp_world_size=TP,
        out_dir=str(out),
        dev=args.dev,
        int8_routed=args.int8_routed,
        kv_ipc=False,
        production_hidden_only=False,
        layer_lo=args.layer_lo,
        layer_hi=args.layer_hi,
    )
    (out / f"ready.rank{args.export_rank}").write_text(
        json.dumps({"rank": args.export_rank, "weight": weight_summary}, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[prefill-only-export rank={args.export_rank}] ready "
        f"weight_bytes={weight_summary['pool_bytes']} dev={args.dev}",
        flush=True,
    )
    try:
        stop = out / "STOP"
        while not stop.exists():
            time.sleep(2.0)
    finally:
        weight_owner.teardown()
    return 0


def _ready(out: Path, rank: int) -> bool:
    return (out / f"ready.rank{rank}").exists()


def _stop_exporters(out: Path, procs: list[subprocess.Popen]) -> None:
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
        handle = getattr(proc, "_prefill_only_log_handle", None)
        if handle is not None:
            handle.close()


def _start_exporters(args: argparse.Namespace, devices: list[int]) -> list[subprocess.Popen]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in out.glob("ready.rank*"):
        path.unlink(missing_ok=True)
    for pattern in ("pypto_weight.*", "ipc_heartbeat.*"):
        for path in out.glob(pattern):
            path.unlink(missing_ok=True)
    (out / "STOP").unlink(missing_ok=True)

    root = Path(__file__).resolve().parents[3]
    procs: list[subprocess.Popen] = []
    for rank, dev in enumerate(devices):
        handle = open(out / f"export_rank{rank}.log", "w", encoding="utf-8")
        command = [
            sys.executable, "-m", "tests.step3p5.harnesses._stage_main_prefill_only",
            "--export-rank", str(rank), "--dev", str(dev),
            "--out", str(out), "--ckpt", args.ckpt,
        ]
        if args.int8_routed:
            command += ["--int8-routed"]
        proc = subprocess.Popen(
            command, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT,
        )
        setattr(proc, "_prefill_only_log_handle", handle)
        procs.append(proc)

    deadline = time.time() + 2400.0
    while time.time() < deadline:
        if all(_ready(out, rank) for rank in range(TP)):
            return procs
        if any(proc.poll() not in (None, 0) for proc in procs):
            _stop_exporters(out, procs)
            raise RuntimeError("prefill-only exporter exited before readiness")
        time.sleep(3.0)
    _stop_exporters(out, procs)
    raise TimeoutError("prefill-only exporters not ready within 40 minutes")


# ---------------------------------------------------------------------------
# Per-length measurement (runs inside a subprocess).
# ---------------------------------------------------------------------------
def _load_embedding_rows(ckpt: str, tokens: list[int]) -> Any:
    import torch
    import safetensors.torch as st
    from models.step3p5.weight_loader import _read_index

    weight_map = _read_index(ckpt)
    shard = weight_map["model.embed_tokens.weight"]
    with st.safe_open(str(Path(ckpt) / shard), framework="pt") as handle:
        rows = torch.stack(
            [handle.get_slice("model.embed_tokens.weight")[int(t), :] for t in tokens],
            dim=0,
        )
    if tuple(rows.shape) != (len(tokens), HIDDEN):
        raise ValueError(
            f"embedding rows shape={tuple(rows.shape)}, expected {(len(tokens), HIDDEN)}"
        )
    return rows.to(torch.bfloat16).contiguous()


def _measure_one_length(args: argparse.Namespace, length: int) -> dict[str, Any]:
    """Rebuild the resident holder at PREFILL_T=pad32(length) and time run()."""
    import torch

    pad32 = ((length + 31) // 32) * 32
    pad128 = ((length + 127) // 128) * 128
    blocks = pad128 // BLOCK_SIZE

    # Compile-time PREFILL_T: set env BEFORE importing config / compiling.
    # MAX_SEQ/ROPE_SEQ must be 128-aligned (config enforces % 128 == 0), while
    # PREFILL_T is only 32-aligned (the actual padded token count). For a
    # non-128-multiple length the two differ (pad32 <= pad128).
    os.environ["PYPTO_STEP3P5_MAX_SEQ"] = str(pad128)
    os.environ["PYPTO_STEP3P5_ROPE_SEQ"] = str(pad128)
    os.environ["PYPTO_STEP3P5_KV_NUM_LAYERS"] = "45"
    os.environ["PYPTO_STEP3P5_KV_CACHE_ROWS"] = str(45 * pad128)
    os.environ["PYPTO_STEP3P5_BLOCK_TABLE_FLAT"] = str(blocks * 16)
    os.environ.setdefault("PYPTO_PROG_BUILD_DIR", str(Path(args.out) / "build_output"))
    # Opt into the weight cache (per-length rebuild must not reload from safetensors).
    os.environ.setdefault("PYPTO_STEP3P5_WEIGHT_CACHE_DIR", "/tmp/step3p5_weights_cache")

    import models.step3p5.prefill_qkv_proj_rope as pq

    # Must run BEFORE prefill_fwd's first import (WholePrefillHolder.build()
    # lazy-imports it); otherwise prefill_fwd captures the stale default 128.
    pq.PREFILL_T = pad32
    pq.PREFILL_SEQ = pad32

    devices = _devices(args.device)
    procs = _start_exporters(args, devices)
    try:
        from tools.step3p5.whole_prefill_holder import WholePrefillHolder

        holder = WholePrefillHolder(
            device_ids=devices, out_dir=args.out, ckpt=args.ckpt,
            platform=args.platform,
        ).build()

        prompt = [args.seed_token] * pad32
        embedding = _load_embedding_rows(args.ckpt, prompt)

        with holder:
            holder.set_prefill_input(embedding)
            for _ in range(max(0, args.ttft_warmup)):
                holder.run()
            samples: list[float] = []
            for _ in range(max(1, args.ttft_iters)):
                started = time.time()
                holder.run()
                samples.append(time.time() - started)
    finally:
        _stop_exporters(Path(args.out), procs)

    ms = sorted(s * 1000.0 for s in samples)
    n = len(ms)
    mean_ms = statistics.fmean(ms)
    return {
        "seq_len": length,
        "pad_len": pad32,
        "iters": n,
        "ttft_ms_min": round(ms[0], 3),
        "ttft_ms_mean": round(mean_ms, 3),
        "ttft_ms_p50": round(ms[(n - 1) // 2], 3),
        "ttft_ms_p99": round(ms[min(n - 1, int(n * 0.99))], 3),
        "ttft_ms_max": round(ms[-1], 3),
        "tokens_per_sec": round(pad32 / (mean_ms / 1000.0), 2),
    }


def main() -> int:
    args = _parse_args()
    if args.export_rank >= 0:
        return _export_rank(args)

    # Subprocess self-invocation: measure a single length, emit JSON to stdout.
    if args.ttft_seq_len is not None:
        result = _measure_one_length(args, args.ttft_seq_len)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0

    if not args.ttft_seq_lens:
        raise SystemExit("--ttft-seq-lens is required (e.g. 128,512,1024,2048)")

    seq_lens = [int(x) for x in args.ttft_seq_lens.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for length in seq_lens:
        print(f"[prefill-only] measuring seq_len={length} ...", flush=True)
        command = [
            sys.executable, "-m", "tests.step3p5.harnesses._stage_main_prefill_only",
            "--ttft-seq-len", str(length),
            "--device", args.device, "--ckpt", args.ckpt, "--out", str(out),
            "--ttft-iters", str(args.ttft_iters),
            "--ttft-warmup", str(args.ttft_warmup),
            "--seed-token", str(args.seed_token),
        ]
        if args.int8_routed:
            command += ["--int8-routed"]
        proc = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"seq_len={length} measurement failed (rc={proc.returncode})")
        # The subprocess stdout mixes the exporter/compile logs with the final
        # JSON line; take the last JSON object.
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and "ttft_ms_mean" in line:
                results.append(json.loads(line))
                print(f"  seq_len={length}: {line}", flush=True)
                break
        else:
            raise SystemExit(f"seq_len={length}: no ttft result JSON in subprocess output")

    report_path = out / "ttft_report.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[prefill-only] TTFT report => {report_path}", flush=True)
    for r in results:
        print(
            f"  seq_len={r['seq_len']:>5}  ttft_ms_mean={r['ttft_ms_mean']:>9}  "
            f"p99={r['ttft_ms_p99']:>9}  tok/s={r['tokens_per_sec']:>8}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
