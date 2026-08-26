import sys

import pytest
import torch

from benchmarks.benchmark_attention import _measure_cuda_memory, main


def test_attention_latency_benchmark(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks every mechanism with the smallest CPU workload."""
    monkeypatch.setattr(
        "benchmarks.benchmark_attention._measure_cpu_memory",
        lambda **_: 1.0,
    )

    main(
        argv=[
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--batch-size",
            "1",
            "--heads",
            "1",
            "--head-dim",
            "2",
            "--sequence-lengths",
            "2",
            "--min-run-time",
            "0.001",
        ]
    )

    output = capsys.readouterr().out

    assert "PyTorch SDPA (auto)" in output
    assert "ADLERS SDPA (auto)" in output
    assert "ADLERS einsum" in output
    assert "ADLERS ProbSparse" in output
    assert "inference" in output
    assert "training" in output
    assert "1x1x2x2" in output
    assert "Peak Process RSS Delta (MiB)" in output


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="resettable peak process RSS is Linux-only",
)
def test_cpu_memory_measurement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks the peak process RSS delta in a fresh subprocess."""
    main(
        argv=[
            "--device",
            "cpu",
            "--mechanism",
            "adlers-einsum",
            "--mode",
            "inference",
            "--batch-size",
            "1",
            "--heads",
            "8",
            "--head-dim",
            "64",
            "--sequence-lengths",
            "512",
            "2",
            "--min-run-time",
            "0.001",
        ]
    )

    output = capsys.readouterr().out
    peak_rss = [
        float(line.split()[-1])
        for line in output.splitlines()
        if line.startswith("ADLERS einsum")
    ]

    assert "Peak Process RSS Delta (MiB)" in output
    assert peak_rss[0] > peak_rss[1] > 0


def test_cuda_memory_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks the peak allocated memory delta without requiring CUDA."""
    calls = 0

    def measured_call() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: None,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: 180,
    )

    peak_memory = _measure_cuda_memory(
        measured_call=measured_call,
        mode="inference",
        device=torch.device("cuda"),
    )

    assert peak_memory == 80
    assert calls == 1
