"""
latency provides latency/throughput measurement for language models.

Supports two measurement modes:
- use_cache=False: Re-forward full growing context each step (baseline/worst case)
- use_cache=True: Use KV-cache with Generator for realistic inference throughput
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import nn

from caramba.benchmark.utils import get_model_vocab_size
from caramba.config.benchmark import LatencyBenchmarkConfig
from caramba.infer.generate import Generator, GenerateConfig, sample_next_token


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
    use_cache: bool = False  # Whether KV-cache was used


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
        """Measure latency for a specific configuration.

        Dispatches to cached or uncached measurement based on config.use_cache.
        """
        if self.config.use_cache:
            return self._measure_with_cache(model, batch_size, prompt_len, gen_len)
        else:
            return self._measure_without_cache(model, batch_size, prompt_len, gen_len)

    def _measure_without_cache(
        self,
        model: nn.Module,
        batch_size: int,
        prompt_len: int,
        gen_len: int,
    ) -> LatencyMeasurement:
        """Measure latency without KV-cache (re-forward full context each step).

        This is the baseline/worst-case measurement that re-computes attention
        over the full growing context at each decode step.

        TTFT Semantics: For consistency with cached mode, TTFT is measured as
        prefill + first decode step. This ensures TTFT is comparable across
        cached and uncached benchmarks (both report time until first token
        is actually generated, not just time to complete prefill).
        """
        vocab_size = self._get_vocab_size(model)

        input_ids = torch.randint(
            0, vocab_size,
            (batch_size, prompt_len),
            device=self.device,
            dtype=torch.long,
        )

        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(input_ids)

        self._sync_device()

        prefill_times: list[float] = []
        first_decode_times: list[float] = []
        decode_times: list[float] = []
        ttft_times: list[float] = []
        total_times: list[float] = []

        for _ in range(self.config.timed_runs):
            current_ids = input_ids.clone()

            # Measure prefill
            start_prefill = time.perf_counter()
            with torch.no_grad():
                logits = model(current_ids)
            self._sync_device()
            end_prefill = time.perf_counter()
            prefill_time = (end_prefill - start_prefill) * 1000

            # Measure first decode step separately (for TTFT)
            start_first_decode = time.perf_counter()
            with torch.no_grad():
                if logits.dim() == 3:
                    last_logits = logits[:, -1, :]
                else:
                    last_logits = logits
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                logits = model(current_ids)
            self._sync_device()
            end_first_decode = time.perf_counter()
            first_decode_time = (end_first_decode - start_first_decode) * 1000

            ttft = prefill_time + first_decode_time

            # Measure remaining decode steps
            start_decode = time.perf_counter()
            with torch.no_grad():
                for _ in range(gen_len - 1):
                    if logits.dim() == 3:
                        last_logits = logits[:, -1, :]
                    else:
                        last_logits = logits
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=-1)
                    logits = model(current_ids)
            self._sync_device()
            end_decode = time.perf_counter()
            remaining_decode_time = (end_decode - start_decode) * 1000
            total_decode_time = first_decode_time + remaining_decode_time

            prefill_times.append(prefill_time)
            first_decode_times.append(first_decode_time)
            decode_times.append(total_decode_time)
            ttft_times.append(ttft)
            total_times.append(prefill_time + total_decode_time)

        avg_prefill = sum(prefill_times) / len(prefill_times)
        avg_decode = sum(decode_times) / len(decode_times)
        avg_ttft = sum(ttft_times) / len(ttft_times)
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
            time_to_first_token_ms=avg_ttft,
            use_cache=False,
        )

    def _measure_with_cache(
        self,
        model: nn.Module,
        batch_size: int,
        prompt_len: int,
        gen_len: int,
    ) -> LatencyMeasurement:
        """Measure latency with KV-cache (realistic inference throughput).

        Uses the Generator class to properly handle KV-cache management,
        providing accurate time-to-first-token (prefill + first decode)
        and decode throughput measurements.

        Note: Cache allocation is performed BEFORE the timed region to measure
        steady-state inference cost (not including one-time cache setup).
        """
        vocab_size = self._get_vocab_size(model)

        input_ids = torch.randint(
            0, vocab_size,
            (batch_size, prompt_len),
            device=self.device,
            dtype=torch.long,
        )

        # Configure generator with greedy decoding
        gen_config = GenerateConfig(
            max_new_tokens=gen_len,
            temperature=0.0,  # Greedy
            max_seq_len=prompt_len + gen_len + 1,
        )

        # Warmup with fresh generator (includes cache allocation)
        for _ in range(self.config.warmup_runs):
            g = Generator(model, config=gen_config, device=self.device)
            with torch.no_grad():
                _ = g.prefill(input_ids)
                logits = g.decode_step(torch.zeros(batch_size, dtype=torch.long, device=self.device))

        self._sync_device()

        prefill_times: list[float] = []
        decode_times: list[float] = []
        ttft_times: list[float] = []  # Time to first token
        total_times: list[float] = []

        for _ in range(self.config.timed_runs):
            # Create generator and allocate caches BEFORE timing
            # This measures steady-state inference, not one-time setup
            g = Generator(model, config=gen_config, device=self.device)
            g._ensure_caches(batch_size)  # Pre-allocate caches
            self._sync_device()

            # Measure prefill (now excludes cache allocation)
            start_prefill = time.perf_counter()
            with torch.no_grad():
                logits = g.prefill(input_ids)
            self._sync_device()
            end_prefill = time.perf_counter()
            prefill_time = (end_prefill - start_prefill) * 1000

            # Sample first token (part of TTFT)
            start_first_decode = time.perf_counter()
            with torch.no_grad():
                next_token = sample_next_token(logits, temperature=0.0)
                logits = g.decode_step(next_token)
            self._sync_device()
            end_first_decode = time.perf_counter()
            first_decode_time = (end_first_decode - start_first_decode) * 1000

            ttft = prefill_time + first_decode_time

            # Measure remaining decode steps
            start_decode = time.perf_counter()
            with torch.no_grad():
                for _ in range(gen_len - 1):
                    next_token = sample_next_token(logits, temperature=0.0)
                    logits = g.decode_step(next_token)
            self._sync_device()
            end_decode = time.perf_counter()
            decode_time = first_decode_time + (end_decode - start_decode) * 1000

            prefill_times.append(prefill_time)
            decode_times.append(decode_time)
            ttft_times.append(ttft)
            total_times.append(prefill_time + decode_time)

        avg_prefill = sum(prefill_times) / len(prefill_times)
        avg_decode = sum(decode_times) / len(decode_times)
        avg_ttft = sum(ttft_times) / len(ttft_times)
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
            time_to_first_token_ms=avg_ttft,
            use_cache=True,
        )

    def _sync_device(self) -> None:
        """Synchronize device for accurate timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _get_vocab_size(self, model: nn.Module) -> int:
        """Get vocab size from model, with fallback to default."""
        return get_model_vocab_size(model, default=32000)
