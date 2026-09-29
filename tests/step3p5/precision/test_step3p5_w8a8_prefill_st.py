from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.step3p5.prefill_precision_suite import DEFAULT_SEQ_LENGTHS, run


DEFAULT_GOLDEN_ROOT = Path(
    "/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_prefill_v001/"
    "golden_step3p5_w8a8_prefill_vllm"
)
DEFAULT_CKPT_DIR = Path(
    "/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp"
)


def _lengths_from_env() -> list[int]:
    value = os.environ.get("STEP3P5_PREFILL_SEQ_LENS")
    if not value:
        return list(DEFAULT_SEQ_LENGTHS)
    return [int(item) for item in value.split(",") if item.strip()]


def test_step3p5_w8a8_prefill_multilen_precision_st(tmp_path: Path) -> None:
    """W8A8 prefill ST: detail + final-logits precision for 1k..128k cases."""


    report_root_env = os.environ.get("STEP3P5_PREFILL_REPORT_ROOT")
    if report_root_env:
        import json
        report_path = Path(report_root_env) / "STEP3P5_W8A8_PREFILL_REPORT.json"
        assert report_path.exists(), f"missing prefill report: {report_path}"
        report = json.load(open(report_path))
        assert report["ok"]
        assert sorted(case["seq_len"] for case in report["cases"]) == sorted(_lengths_from_env())
        return

    golden_root = Path(os.environ.get("STEP3P5_W8A8_PREFILL_GOLDEN_ROOT", str(DEFAULT_GOLDEN_ROOT)))
    ckpt_dir = Path(os.environ.get("STEP3P5_W8A8_CKPT_DIR", str(DEFAULT_CKPT_DIR)))
    if not golden_root.exists():
        pytest.xfail(f"missing W8A8 prefill golden root: {golden_root}")
    if not ckpt_dir.exists():
        pytest.xfail(f"missing W8A8 checkpoint: {ckpt_dir}")

    import argparse

    report = run(argparse.Namespace(
        golden_root=str(golden_root),
        ckpt_dir=str(ckpt_dir),
        output_root=str(tmp_path / "prefill_precision"),
        case=None,
        seq_len=_lengths_from_env(),
        tp_world_size=8,
        rtol=float(os.environ.get("STEP3P5_PREFILL_RTOL", "5e-3")),
        atol=float(os.environ.get("STEP3P5_PREFILL_ATOL", "5e-3")),
        mlp_rtol=float(os.environ.get("STEP3P5_PREFILL_MLP_RTOL", "8e-2")),
        mlp_atol=float(os.environ.get("STEP3P5_PREFILL_MLP_ATOL", "2e-1")),
        pass_rate=float(os.environ.get("STEP3P5_PREFILL_PASS_RATE", "0.997")),
        max_detail_tokens=int(os.environ.get("STEP3P5_PREFILL_MAX_DETAIL_TOKENS", "1024")),
        logits_chunk_size=int(os.environ.get("STEP3P5_PREFILL_LOGITS_CHUNK", "4096")),
        copy_golden_manifest=True,
        make_tar=False,
        tar_path=None,
        include_golden_in_tar=False,
    ))
    assert report["ok"]
    assert sorted(case["seq_len"] for case in report["cases"]) == sorted(_lengths_from_env())
