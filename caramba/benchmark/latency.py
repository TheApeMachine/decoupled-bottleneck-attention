"""
latency provides latency/throughput measurement for language models.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import nn

from caramba.config.benchmark import LatencyBenchmarkConfig
from caramba.infer.generate import GenerateConfig


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    prompt_len: int
    gen_len: int
    batch_size: int
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    tokens_per_second: float
    time_to_first_token_ms: float


@dataclass
class LatencyResult:
    """Result of a latency benchmark."""

    model_name: str
    measurements: list[LatencyMeasurement] = field(default_factory=list)

    @property
    def avg_tokens_per_second(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(m.tokens_per_second for m in self.measurements) / len(self.measurements)

    @property
    def avg_time_to_first_token_ms(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(m.time_to_first_token_ms for m in self.measurements) / len(self.measurements)


class LatencyBenchmark:
    """Measures latency and throughput of generation."""

    def __init__(self, config: LatencyBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def run(self, model: nn.Module, model_name: str) -> LatencyResult:
        """Run latency benchmark on a model."""
        model.eval()
        result = LatencyResult(model_name=model_name)

        for batch_size in self.config.batch_sizes:
            for prompt_len in self.config.prompt_lengths:
                for gen_len in self.config.generation_lengths:
                    measurement = self._measure(
                        model=model,
                        batch_size=batch_size,
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                    )
                    result.measurements.append(measurement)

        return result

    def _measure(
        self,
        model: nn.Module,
        batch_size: int,
        prompt_len: int,
        gen_len: int,
    ) -> LatencyMeasurement:
        """Measure latency for a specific configuration."""
        # Create random input tokens
        input_ids = torch.randint(
            0, 32000,  # Assume vocab size
            (batch_size, prompt_len),
            device=self.device,
            dtype=torch.long,
        )

        gen_config = GenerateConfig(
            max_new_tokens=gen_len,
            temperature=1.0,
            top_p=1.0,
        )

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                # Simple forward pass for warmup
                _ = model(input_ids)

        # Synchronize before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        # Timed runs
        prefill_times: list[float] = []
        decode_times: list[float] = []
        total_times: list[float] = []

        for _ in range(self.config.timed_runs):
            # Reset input for each run
            current_ids = input_ids.clone()

            # Measure prefill
            start_prefill = time.perf_counter()
            with torch.no_grad():
                logits = model(current_ids)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()

            end_prefill = time.perf_counter()
            prefill_time = (end_prefill - start_prefill) * 1000  # ms

            # Measure decode (autoregressive generation without cache for simplicity)
            start_decode = time.perf_counter()
            with torch.no_grad():
                for _ in range(gen_len):
                    # Get last token logits
                    if logits.dim() == 3:
                        last_logits = logits[:, -1, :]
                    else:
                        last_logits = logits
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=-1)
                    # Forward with growing context (no cache benchmark)
                    logits = model(current_ids)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()

            end_decode = time.perf_counter()
            decode_time = (end_decode - start_decode) * 1000  # ms

            prefill_times.append(prefill_time)
            decode_times.append(decode_time)
            total_times.append(prefill_time + decode_time)

        avg_prefill = sum(prefill_times) / len(prefill_times)
        avg_decode = sum(decode_times) / len(decode_times)
        avg_total = sum(total_times) / len(total_times)

        tokens_generated = gen_len * batch_size
        tokens_per_second = tokens_generated / (avg_total / 1000) if avg_total > 0 else 0

        return LatencyMeasurement(
            prompt_len=prompt_len,
            gen_len=gen_len,
            batch_size=batch_size,
            prefill_time_ms=avg_prefill,
            decode_time_ms=avg_decode,
            total_time_ms=avg_total,
            tokens_per_second=tokens_per_second,
            time_to_first_token_ms=avg_prefill,
        )
