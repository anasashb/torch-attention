import pytest

from benchmarks.benchmark_attention import main


def test_scaled_dot_product_latency_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks the scaled dot-product benchmark's smallest CPU workload."""
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
    assert "inference" in output
    assert "training" in output
    assert "1x1x2x2" in output
