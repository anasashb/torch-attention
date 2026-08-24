import pytest
import torch

from benchmarks.benchmark_attention import _measure_cuda_memory, main


def test_attention_latency_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks every mechanism with the smallest CPU workload."""
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
    assert "Peak Allocated Memory Delta (MiB)" in output


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
