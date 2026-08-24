"""Benchmark attention implementations."""

import argparse
import platform
from collections.abc import Callable
from itertools import product

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
_SEED = 66


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


def _measure_latency(
    measured_call: Callable[[], object],
    mode: str,
    num_threads: int,
    min_run_time: float,
) -> benchmark.Measurement:
    with torch.inference_mode(mode=mode == "inference"):
        return benchmark.Timer(
            stmt="measured_call()",
            globals={"measured_call": measured_call},
            num_threads=num_threads,
        ).blocked_autorange(min_run_time=min_run_time)


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


def _print_benchmark_setup(
    device: torch.device,
    dtype: torch.dtype,
    is_causal: bool,
    num_threads: int,
) -> None:

    device_name = (
        torch.cuda.get_device_name(device=device)
        if device.type == "cuda"
        else f"CPU ({platform.machine()})"
    )

    print("Attention benchmark")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device_name} [{device}]")
    print(f"CUDA: {torch.version.cuda or 'not available'}")
    print(f"Dtype: {str(dtype).removeprefix('torch.')}")
    print(f"Causal: {is_causal}")
    print(f"CPU threads per measurement: {num_threads}")
    print()


def main(argv: list[str] | None = None) -> None:
    """Run the attention benchmark."""

    args = _parse_args(argv=argv)
    device: torch.device = args.device
    dtype = _DTYPES.get(args.dtype) or (
        torch.float16 if device.type == "cuda" else torch.float32
    )

    if device.type == "cuda" and device.index is not None:
        # needed for torch.utils.benchmark.Timer
        # if --device arg has index, and its not cuda:0
        torch.cuda.set_device(device=device)

    modes = ("inference", "training") if args.mode == "all" else (args.mode,)
    mechanisms = (
        (args.mechanism,) if args.mechanism else tuple(_MECHANISM_LABELS)
    )

    _print_benchmark_setup(
        device=device,
        dtype=dtype,
        is_causal=args.causal,
        num_threads=args.num_threads,
    )

    header = (
        f"{'Mechanism':<20} {'Mode':<10} {'Shape [B,H,L,D]':<20} "
        f"{'Median (ms)':>12} {'IQR (ms)':>10} "
        f"{'Peak Allocated Memory Delta (MiB)':>33}"
    )
    print(header)
    print("-" * len(header))

    for sequence_length, mode, mechanism in product(
        args.sequence_lengths,
        modes,
        mechanisms,
    ):
        training = mode == "training"
        shape = (
            args.batch_size,
            args.heads,
            sequence_length,
            args.head_dim,
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
            is_causal=args.causal,
            training=training,
        )

        measured_call = _make_execution_call(
            attention_call=attention_call,
            mode=mode,
            query=query,
            key=key,
            value=value,
        )

        latency_measurement = _measure_latency(
            measured_call=measured_call,
            mode=mode,
            num_threads=args.num_threads,
            min_run_time=args.min_run_time,
        )

        memory_label = "-"
        if device.type == "cuda":
            peak_memory = _measure_cuda_memory(
                measured_call=measured_call,
                mode=mode,
                device=device,
            )
            memory_label = f"{peak_memory / 1024**2:.2f}"

        shape_label = (
            f"{args.batch_size}x{args.heads}x{sequence_length}x{args.head_dim}"
        )
        print(
            f"{_MECHANISM_LABELS[mechanism]:<20} "
            f"{mode:<10} {shape_label:<20} "
            f"{latency_measurement.median * 1_000:>12.4f} "
            f"{latency_measurement.iqr * 1_000:>10.4f} "
            f"{memory_label:>33}"
        )


if __name__ == "__main__":
    main()
