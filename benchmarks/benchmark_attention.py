"""Benchmark attention implementations."""

import argparse
import json
import platform
import subprocess  # noqa: S404
import sys
from collections.abc import Callable
from itertools import chain, product
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch import Tensor

from adlers import ProbSparseAttention, ScaledDotProductAttention

_DEFAULT_SEQUENCE_LENGTHS = (128, 512, 2048)
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_MECHANISM_LABELS = {
    "sdpa-auto": "PyTorch SDPA (auto)",
    "adlers-sdpa": "ADLERS SDPA (auto)",
    "adlers-einsum": "ADLERS einsum",
    "adlers-probsparse": "ADLERS ProbSparse",
}
_SCHEMA_VERSION = 1
_SEED = 66
_JsonObject = dict[str, Any]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value}"
        )
    return parsed


def _device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid device {value!r}") from error

    if device.type not in {"cpu", "cuda"}:
        raise argparse.ArgumentTypeError(
            f"expected a CPU or CUDA device, got {value!r}"
        )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA is not available")

    return device


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark attention implementations with warm inputs.",
    )
    parser.add_argument(
        "--mechanism",
        choices=tuple(_MECHANISM_LABELS),
        help="Run one mechanism instead of all mechanisms.",
    )
    parser.add_argument(
        "--mode",
        choices=("inference", "training", "all"),
        default="all",
    )
    parser.add_argument(
        "--device",
        type=_device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES))
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--heads", type=_positive_int, default=8)
    parser.add_argument("--head-dim", type=_positive_int, default=64)
    parser.add_argument(
        "--sequence-lengths",
        type=_positive_int,
        nargs="+",
        default=list(_DEFAULT_SEQUENCE_LENGTHS),
        metavar="LENGTH",
    )
    parser.add_argument("--num-threads", type=_positive_int, default=1)
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="Minimum seconds collected for each measurement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write benchmark results to this JSON file.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare results with this JSON benchmark run.",
    )
    parser.add_argument(
        "--cpu-memory-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(args=argv)


def _make_attention_call(
    mechanism: str,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    training: bool,
) -> Callable[[], Tensor]:
    if mechanism == "sdpa-auto":
        # this is here only for us to compare that our SDPA wrapper does not
        # add too much compute overhead compared to torch's native function
        def call_sdpa_auto() -> Tensor:
            return F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        return call_sdpa_auto

    attention = (
        ProbSparseAttention(
            is_causal=is_causal,
            dropout_rate=0.0,
            output_attention_scores=False,
            strict_mode=True,
        )
        if mechanism == "adlers-probsparse"
        else ScaledDotProductAttention(
            is_causal=is_causal,
            dropout_rate=0.0,
            output_attention_scores=False,
            strict_mode=True,
            backend="sdpa" if mechanism == "adlers-sdpa" else "einsum",
        )
    ).to(device=query.device)

    attention.train(mode=training)

    def call_adlers() -> Tensor:
        output, _ = attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )
        return output

    return call_adlers


def _make_execution_call(
    attention_call: Callable[[], Tensor],
    mode: str,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Callable[[], object]:
    if mode != "training":
        return attention_call

    gradient = torch.ones_like(query)

    def run_forward_and_backward() -> None:
        output = attention_call()
        torch.autograd.grad(
            outputs=output,
            inputs=(query, key, value),
            grad_outputs=gradient,
        )

    return run_forward_and_backward


def _measure_cuda_memory(
    measured_call: Callable[[], object],
    mode: str,
    device: torch.device,
) -> int:
    with torch.inference_mode(mode=mode == "inference"):
        torch.cuda.synchronize(device=device)
        baseline = torch.cuda.memory_allocated(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
        measured_call()
        torch.cuda.synchronize(device=device)

    return torch.cuda.max_memory_allocated(device=device) - baseline


def _measure_cpu_memory(
    batch_size: int,
    heads: int,
    head_dim: int,
    num_threads: int,
    is_causal: bool,
    dtype: torch.dtype,
    sequence_length: int,
    mode: str,
    mechanism: str,
) -> float:
    # A direct worker inherits this process's RSS high-water mark through exec.
    # The lightweight supervisor forks it from a clean process.
    options = {
        "--device": "cpu",
        "--dtype": str(dtype).removeprefix("torch."),
        "--mode": mode,
        "--mechanism": mechanism,
        "--batch-size": str(batch_size),
        "--heads": str(heads),
        "--head-dim": str(head_dim),
        "--sequence-lengths": str(sequence_length),
        "--num-threads": str(num_threads),
    }
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys; subprocess.run(args=sys.argv[1:], check=True)",
        sys.executable,
        __file__,
        "--cpu-memory-worker",
        *chain.from_iterable(options.items()),
        *(("--causal",) if is_causal else ()),
    ]

    result = subprocess.run(  # noqa: S603
        args=command,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout)


def _print_benchmark_setup(
    device: torch.device,
    device_name: str,
    dtype: torch.dtype,
    is_causal: bool,
    num_threads: int,
) -> None:
    print("Attention benchmark")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device_name} [{device}]")
    print(f"CUDA: {torch.version.cuda or 'not available'}")
    print(f"Dtype: {str(dtype).removeprefix('torch.')}")
    print(f"Causal: {is_causal}")
    print(f"CPU threads per measurement: {num_threads}")
    print()


def _case_key(result: _JsonObject) -> tuple[object, ...]:
    return (
        result["mechanism"],
        result["mode"],
        tuple(result["shape"]),
        result["dtype"],
        result["device"],
        result["causal"],
    )


def _load_baseline_cases(
    path: Path,
    environment: _JsonObject,
    settings: _JsonObject,
    expected_cases: set[tuple[object, ...]],
    memory_metric: str,
) -> dict[tuple[object, ...], _JsonObject]:
    baseline: _JsonObject = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            "baseline schema is incompatible; regenerate it with this script"
        )

    if baseline.get("environment") != environment:
        raise ValueError(
            "baseline environment differs; rerun both revisions with the "
            "same Python, PyTorch, CUDA, device, and CPU thread count"
        )

    if baseline.get("settings") != settings:
        raise ValueError(
            "baseline settings differ; use the same minimum runtime and seed"
        )

    results: list[_JsonObject] = baseline["results"]
    cases = {_case_key(result): result for result in results}

    if len(cases) != len(results) or cases.keys() != expected_cases:
        raise ValueError(
            "baseline cases differ; use identical mechanisms, modes, shapes, "
            "dtype, device, and causal setting"
        )

    if any(result["memory"]["metric"] != memory_metric for result in results):
        raise ValueError("baseline memory metric differs from current run")

    return cases


def _print_baseline_comparison(
    results: list[_JsonObject],
    baseline_cases: dict[tuple[object, ...], _JsonObject],
) -> None:

    print("\nBaseline comparison (negative is better)")
    header = (
        f"{'Mechanism':<20} {'Mode':<10} {'Shape [B,H,L,D]':<20} "
        f"{'Latency Delta (ms)':>18} {'Latency Delta (%)':>18} "
        f"{'Memory Delta (MiB)':>20} {'Memory Delta (%)':>18}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        baseline = baseline_cases[_case_key(result)]
        shape_label = "x".join(str(dimension) for dimension in result["shape"])

        current_latency = result["latency"]["median_seconds"]
        baseline_latency = baseline["latency"]["median_seconds"]

        latency_delta_ms = (
            f"{(current_latency - baseline_latency) * 1_000:+.4f}"
        )
        latency_delta_percent = (
            f"{(current_latency / baseline_latency - 1) * 100:+.2f}"
        )

        current_memory = result["memory"]["value_mib"]
        baseline_memory = baseline["memory"]["value_mib"]

        memory_delta_mib = (
            "unavailable"
            if current_memory is None or baseline_memory is None
            else f"{current_memory - baseline_memory:+.2f}"
        )
        memory_delta_percent = (
            "unavailable"
            if current_memory is None or not baseline_memory
            else f"{(current_memory / baseline_memory - 1) * 100:+.2f}"
        )

        print(
            f"{_MECHANISM_LABELS[result['mechanism']]:<20} "
            f"{result['mode']:<10} {shape_label:<20} "
            f"{latency_delta_ms:>18} {latency_delta_percent:>18} "
            f"{memory_delta_mib:>20} {memory_delta_percent:>18}"
        )


def main(argv: list[str] | None = None) -> None:
    """Run the attention benchmark."""

    args = _parse_args(argv=argv)
    selected_mechanism: str | None = args.mechanism
    selected_mode: str = args.mode
    device: torch.device = args.device
    selected_dtype: str | None = args.dtype
    is_causal: bool = args.causal
    batch_size: int = args.batch_size
    heads: int = args.heads
    head_dim: int = args.head_dim
    sequence_lengths: list[int] = args.sequence_lengths
    num_threads: int = args.num_threads
    min_run_time: float = args.min_run_time
    output: Path | None = args.output
    baseline_path: Path | None = args.baseline
    cpu_memory_worker: bool = args.cpu_memory_worker

    default_dtype = torch.float16 if device.type == "cuda" else torch.float32
    dtype = (
        _DTYPES[selected_dtype] if selected_dtype is not None else default_dtype
    )

    dtype_name = str(dtype).removeprefix("torch.")
    device_name = (
        torch.cuda.get_device_name(device=device)
        if device.type == "cuda"
        else f"CPU ({platform.machine()})"
    )

    environment: _JsonObject = {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "cpu_threads": num_threads,
    }

    settings: _JsonObject = {
        "minimum_run_time_seconds": min_run_time,
        "seed": _SEED,
    }

    if device.type == "cuda" and device.index is not None:
        # needed for torch.utils.benchmark.Timer
        # if --device arg has index, and its not cuda:0
        torch.cuda.set_device(device=device)

    modes = (
        ("inference", "training")
        if selected_mode == "all"
        else (selected_mode,)
    )

    mechanisms = (
        (selected_mechanism,)
        if selected_mechanism
        else tuple(_MECHANISM_LABELS)
    )

    cases = tuple(product(sequence_lengths, modes, mechanisms))

    memory_header, memory_metric = (
        (
            "Peak Allocated Memory Delta (MiB)",
            "peak_allocated_memory_delta",
        )
        if device.type == "cuda"
        else ("Peak Process RSS Delta (MiB)", "peak_process_rss_delta")
    )

    baseline_cases: dict[tuple[object, ...], _JsonObject] | None = None
    if baseline_path is not None:
        expected_cases: set[tuple[object, ...]] = {
            (
                mechanism,
                mode,
                (
                    batch_size,
                    heads,
                    sequence_length,
                    head_dim,
                ),
                dtype_name,
                str(device),
                is_causal,
            )
            for sequence_length, mode, mechanism in cases
        }

        baseline_cases = _load_baseline_cases(
            path=baseline_path,
            environment=environment,
            settings=settings,
            expected_cases=expected_cases,
            memory_metric=memory_metric,
        )

    if cpu_memory_worker:
        torch.set_num_threads(num_threads)
    else:
        _print_benchmark_setup(
            device=device,
            device_name=device_name,
            dtype=dtype,
            is_causal=is_causal,
            num_threads=num_threads,
        )

        header = (
            f"{'Mechanism':<20} {'Mode':<10} {'Shape [B,H,L,D]':<20} "
            f"{'Median (ms)':>12} {'IQR (ms)':>10} "
            f"{memory_header:>33}"
        )
        print(header)
        print("-" * len(header))

    results: list[_JsonObject] = []

    for sequence_length, mode, mechanism in cases:
        training = mode == "training"
        shape = (
            batch_size,
            heads,
            sequence_length,
            head_dim,
        )

        generator = torch.Generator(device=device).manual_seed(_SEED)
        query, key, value = (
            torch.randn(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
                requires_grad=training,
            )
            for _ in range(3)
        )

        attention_call = _make_attention_call(
            mechanism=mechanism,
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            training=training,
        )

        measured_call = _make_execution_call(
            attention_call=attention_call,
            mode=mode,
            query=query,
            key=key,
            value=value,
        )

        if cpu_memory_worker:
            import resource

            # Linux resets peak RSS to current RSS when clear_refs receives 5.
            Path("/proc/self/clear_refs").write_text(data="5", encoding="ascii")
            baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            with torch.inference_mode(mode=mode == "inference"):
                measured_call()

            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print((peak_rss - baseline_rss) / 1024)
            return

        with torch.inference_mode(mode=mode == "inference"):
            latency_measurement = benchmark.Timer(
                stmt="measured_call()",
                globals={"measured_call": measured_call},
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=min_run_time)

        memory_mib = None

        if device.type == "cuda":
            peak_memory = _measure_cuda_memory(
                measured_call=measured_call,
                mode=mode,
                device=device,
            )
            memory_mib = peak_memory / 1024**2

        elif sys.platform == "linux":
            memory_mib = _measure_cpu_memory(
                batch_size=batch_size,
                heads=heads,
                head_dim=head_dim,
                num_threads=num_threads,
                is_causal=is_causal,
                dtype=dtype,
                sequence_length=sequence_length,
                mode=mode,
                mechanism=mechanism,
            )

        memory_label = (
            "unavailable" if memory_mib is None else f"{memory_mib:.2f}"
        )

        shape_label = f"{batch_size}x{heads}x{sequence_length}x{head_dim}"
        print(
            f"{_MECHANISM_LABELS[mechanism]:<20} "
            f"{mode:<10} {shape_label:<20} "
            f"{latency_measurement.median * 1_000:>12.4f} "
            f"{latency_measurement.iqr * 1_000:>10.4f} "
            f"{memory_label:>33}"
        )

        results.append(
            {
                "mechanism": mechanism,
                "mode": mode,
                "shape": list(shape),
                "dtype": dtype_name,
                "device": str(device),
                "causal": is_causal,
                "latency": {
                    "median_seconds": latency_measurement.median,
                    "iqr_seconds": latency_measurement.iqr,
                    "samples_seconds": latency_measurement.times,
                },
                "memory": {
                    "metric": memory_metric,
                    "value_mib": memory_mib,
                },
            }
        )

    if output is not None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "environment": environment,
            "settings": settings,
            "results": results,
        }
        output.write_text(
            data=json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if baseline_cases is not None:
        _print_baseline_comparison(
            results=results,
            baseline_cases=baseline_cases,
        )


if __name__ == "__main__":
    main()
