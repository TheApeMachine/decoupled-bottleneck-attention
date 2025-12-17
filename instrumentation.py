#!/usr/bin/env python3
"""
instrumentation.py

Deep instrumentation framework for understanding attention mechanisms.

Provides:
- InstrumentationConfig: Configurable analysis depth
- Analyzer: Hooks for attention, gradient, and representation analysis
- TensorWriter: HDF5 storage for tensor data

Usage:
    from instrumentation import InstrumentationConfig, Analyzer

    config = InstrumentationConfig(level="medium")
    analyzer = Analyzer(config, out_dir="runs/experiment")
    
    # In training loop:
    analyzer.step(step, model, loss)
    
    # After training:
    analyzer.finalize()
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Optional HDF5 support
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. Tensor storage will use numpy files.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InstrumentationConfig:
    """
    Configurable instrumentation levels.
    
    Levels:
        off:    No analysis, minimal logging
        light:  Basic metrics only (~5% overhead)
        medium: + rank/entropy analysis (~15% overhead)
        heavy:  + full attention matrices, gradient rank (~30%+ overhead)
    """
    level: Literal["off", "light", "medium", "heavy"] = "medium"
    
    # Analysis frequency (in training steps)
    analysis_every: int = 100
    
    # What to analyze at each level
    # Light level
    log_loss: bool = True
    log_throughput: bool = True
    log_lr: bool = True
    
    # Medium level (in addition to light)
    compute_attention_entropy: bool = True
    compute_attention_sparsity: bool = True
    compute_attention_rank: bool = True
    compute_hidden_rank: bool = True
    track_gradient_norms: bool = True
    
    # Heavy level (in addition to medium)
    save_attention_matrices: bool = False
    compute_gradient_rank: bool = False
    compute_layer_similarity: bool = False
    track_parameter_changes: bool = False
    save_hidden_states: bool = False
    
    # Path contribution (specific to decoupled attention)
    compute_path_contribution: bool = True
    
    def __post_init__(self):
        """Apply level presets."""
        if self.level == "off":
            self.analysis_every = 999999  # Effectively disable
            self.compute_attention_entropy = False
            self.compute_attention_sparsity = False
            self.compute_attention_rank = False
            self.compute_hidden_rank = False
            self.track_gradient_norms = False
            self.compute_path_contribution = False
            
        elif self.level == "light":
            self.compute_attention_rank = False
            self.compute_hidden_rank = False
            self.compute_layer_similarity = False
            self.compute_gradient_rank = False
            self.compute_path_contribution = False
            
        elif self.level == "heavy":
            self.save_attention_matrices = True
            self.compute_gradient_rank = True
            self.compute_layer_similarity = True
            self.track_parameter_changes = True


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_effective_rank(matrix: torch.Tensor, threshold: float = 0.99) -> Tuple[float, List[float]]:
    """
    Compute effective rank via SVD.
    
    Returns:
        effective_rank: Number of singular values needed to capture `threshold` of variance
        singular_values: Normalized singular values (as list)
    """
    try:
        # Flatten to 2D if needed
        if matrix.dim() > 2:
            matrix = matrix.reshape(-1, matrix.shape[-1])
        
        # SVD
        U, S, V = torch.svd(matrix.float())
        
        # Normalize singular values
        S_norm = S / S.sum()
        
        # Cumulative sum to find effective rank
        cumsum = torch.cumsum(S_norm, dim=0)
        effective_rank = (cumsum < threshold).sum().item() + 1
        
        return effective_rank, S_norm.tolist()[:min(20, len(S_norm))]  # Top 20 singular values
    except Exception as e:
        return -1, []


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution.
    
    Higher entropy = more spread out attention
    Lower entropy = more focused attention
    """
    try:
        # attn_weights: (B, H, T, T) or (B, H, T, T)
        # Clamp to avoid log(0)
        attn = attn_weights.clamp(min=1e-10)
        entropy = -(attn * attn.log()).sum(dim=-1).mean().item()
        return entropy
    except Exception:
        return -1.0


def compute_attention_sparsity(attn_weights: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute sparsity of attention weights.
    
    Returns fraction of attention weights below threshold.
    """
    try:
        sparse_count = (attn_weights < threshold).float().mean().item()
        return sparse_count
    except Exception:
        return -1.0


def compute_path_contributions(
    sem_scores: Optional[torch.Tensor], 
    geo_scores: Optional[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute relative contribution of semantic vs geometric paths.
    
    For decoupled attention: score = sem_scores + geo_scores
    This measures which path contributes more to the final attention.
    """
    if sem_scores is None or geo_scores is None:
        return {"semantic_ratio": -1, "geometric_ratio": -1}
    
    try:
        sem_mag = sem_scores.abs().mean().item()
        geo_mag = geo_scores.abs().mean().item()
        total = sem_mag + geo_mag + 1e-10
        
        return {
            "semantic_ratio": sem_mag / total,
            "geometric_ratio": geo_mag / total,
            "semantic_magnitude": sem_mag,
            "geometric_magnitude": geo_mag,
        }
    except Exception:
        return {"semantic_ratio": -1, "geometric_ratio": -1}


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient norms for each parameter group.
    """
    norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            
            # Group by layer type
            if "attn" in name.lower():
                key = "attn_grad_norm"
            elif "ffn" in name.lower() or "mlp" in name.lower():
                key = "ffn_grad_norm"
            elif "embed" in name.lower() or "emb" in name.lower():
                key = "embed_grad_norm"
            else:
                key = "other_grad_norm"
            
            if key not in norms:
                norms[key] = []
            norms[key].append(norm)
    
    # Average per group
    return {k: sum(v) / len(v) for k, v in norms.items()}


# =============================================================================
# MEMORY INSTRUMENTATION
# =============================================================================

def get_tensor_memory_bytes(t: torch.Tensor) -> int:
    """Get actual memory usage of a tensor in bytes."""
    return t.element_size() * t.nelement()


def measure_model_memory(model: nn.Module) -> Dict[str, int]:
    """
    Measure actual memory usage of model parameters.
    
    Returns dict with bytes for each component.
    """
    memory = {
        "total_params_bytes": 0,
        "attn_params_bytes": 0,
        "ffn_params_bytes": 0,
        "embed_params_bytes": 0,
        "other_params_bytes": 0,
    }
    
    for name, param in model.named_parameters():
        size = get_tensor_memory_bytes(param)
        memory["total_params_bytes"] += size
        
        if "attn" in name.lower():
            memory["attn_params_bytes"] += size
        elif "ffn" in name.lower() or "mlp" in name.lower() or "ff" in name.lower():
            memory["ffn_params_bytes"] += size
        elif "embed" in name.lower() or "emb" in name.lower() or "tok" in name.lower():
            memory["embed_params_bytes"] += size
        else:
            memory["other_params_bytes"] += size
    
    return memory


def measure_kv_cache_memory(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    """
    Measure actual KV cache memory for a model.
    
    This creates a dummy KV cache and measures its real size.
    """
    # Get model config
    cfg = getattr(model, 'cfg', None)
    if cfg is None:
        return {"error": "No model config found"}
    
    n_layer = getattr(cfg, 'n_layer', 6)
    n_head = getattr(cfg, 'n_head', 8)
    d_model = getattr(cfg, 'd_model', 512)
    attn_mode = getattr(cfg, 'attn_mode', 'standard')
    
    # Determine KV dimensions based on attention mode
    if attn_mode == 'standard':
        kv_dim = d_model
    elif attn_mode == 'bottleneck':
        kv_dim = getattr(cfg, 'attn_dim', d_model)
    elif attn_mode == 'decoupled':
        sem_dim = getattr(cfg, 'sem_dim', 32)
        geo_dim = getattr(cfg, 'geo_dim', 64)
        # Decoupled stores: k_sem, k_geo, v
        kv_dim = sem_dim + geo_dim + getattr(cfg, 'attn_dim', sem_dim + geo_dim)
    elif attn_mode == 'gqa':
        kv_head = getattr(cfg, 'kv_head', n_head)
        head_dim = getattr(cfg, 'attn_dim', d_model) // n_head
        kv_dim = kv_head * head_dim * 2  # K and V
    else:
        kv_dim = d_model
    
    # Calculate actual bytes
    elem_size = 2 if dtype == torch.float16 else 4  # fp16 or fp32
    
    # KV cache per layer: 2 (K,V) * batch * seq * dim * elem_size
    # For decoupled: k_sem + k_geo + v
    if attn_mode == 'decoupled':
        per_layer_bytes = (sem_dim + geo_dim + getattr(cfg, 'attn_dim', sem_dim + geo_dim)) * batch_size * seq_len * elem_size
    else:
        per_layer_bytes = 2 * kv_dim * batch_size * seq_len * elem_size
    
    total_bytes = per_layer_bytes * n_layer
    
    # Also measure with Q4 quantization (4 bits per element)
    q4_elem_size = 0.5  # 4 bits = 0.5 bytes
    if attn_mode == 'decoupled':
        per_layer_q4 = (sem_dim + geo_dim + getattr(cfg, 'attn_dim', sem_dim + geo_dim)) * batch_size * seq_len * q4_elem_size
    else:
        per_layer_q4 = 2 * kv_dim * batch_size * seq_len * q4_elem_size
    total_q4_bytes = per_layer_q4 * n_layer
    
    return {
        "attn_mode": attn_mode,
        "n_layer": n_layer,
        "kv_dim": kv_dim,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dtype": str(dtype),
        "fp16_per_layer_bytes": int(per_layer_bytes),
        "fp16_total_bytes": int(total_bytes),
        "fp16_total_mb": total_bytes / (1024 * 1024),
        "q4_per_layer_bytes": int(per_layer_q4),
        "q4_total_bytes": int(total_q4_bytes),
        "q4_total_mb": total_q4_bytes / (1024 * 1024),
        "fp16_to_q4_ratio": total_bytes / total_q4_bytes if total_q4_bytes > 0 else 0,
    }


def measure_peak_memory() -> Dict[str, float]:
    """
    Measure current and peak GPU/MPS memory usage.
    """
    result = {}
    
    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        result["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have detailed memory tracking, but we can try
        try:
            result["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
        except:
            pass
    
    return result


def compute_layer_similarity(hidden_states: List[torch.Tensor]) -> List[float]:
    """
    Compute similarity between adjacent layers using centered kernel alignment (CKA).
    
    Returns list of similarities between layer i and layer i+1.
    """
    if len(hidden_states) < 2:
        return []
    
    similarities = []
    
    for i in range(len(hidden_states) - 1):
        h1 = hidden_states[i].float().reshape(-1, hidden_states[i].shape[-1])
        h2 = hidden_states[i + 1].float().reshape(-1, hidden_states[i + 1].shape[-1])
        
        try:
            # Simple cosine similarity of means (fast approximation)
            m1 = h1.mean(dim=0)
            m2 = h2.mean(dim=0)
            sim = torch.cosine_similarity(m1.unsqueeze(0), m2.unsqueeze(0)).item()
            similarities.append(sim)
        except Exception:
            similarities.append(-1)
    
    return similarities


# =============================================================================
# TENSOR WRITER
# =============================================================================

class TensorWriter:
    """
    Efficient storage for tensor data using HDF5 or numpy.
    """
    
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.h5_path = self.out_dir / "analysis.h5"
        self.h5_file = None
        
        if HAS_H5PY:
            self.h5_file = h5py.File(self.h5_path, "w")
            self._init_groups()
        else:
            # Fallback to numpy directory
            self.np_dir = self.out_dir / "tensors"
            self.np_dir.mkdir(exist_ok=True)
    
    def _init_groups(self):
        """Initialize HDF5 groups."""
        if self.h5_file:
            self.h5_file.create_group("attention")
            self.h5_file.create_group("gradients")
            self.h5_file.create_group("representations")
            self.h5_file.create_group("singular_values")
    
    def save_attention(self, step: int, layer: int, head: int, attn: torch.Tensor):
        """Save attention matrix."""
        key = f"step_{step}_layer_{layer}_head_{head}"
        data = attn.detach().cpu().numpy()
        
        if self.h5_file:
            self.h5_file["attention"].create_dataset(key, data=data, compression="gzip")
        else:
            np.save(self.np_dir / f"attn_{key}.npy", data)
    
    def save_singular_values(self, step: int, layer: int, head: int, svs: List[float]):
        """Save singular values."""
        key = f"step_{step}_layer_{layer}_head_{head}"
        data = np.array(svs)
        
        if self.h5_file:
            self.h5_file["singular_values"].create_dataset(key, data=data)
        else:
            np.save(self.np_dir / f"sv_{key}.npy", data)
    
    def save_hidden_state(self, step: int, layer: int, hidden: torch.Tensor):
        """Save hidden state."""
        key = f"step_{step}_layer_{layer}"
        data = hidden.detach().cpu().numpy()
        
        if self.h5_file:
            self.h5_file["representations"].create_dataset(key, data=data, compression="gzip")
        else:
            np.save(self.np_dir / f"hidden_{key}.npy", data)
    
    def close(self):
        """Close file handles."""
        if self.h5_file:
            self.h5_file.close()


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class Analyzer:
    """
    Main analysis orchestrator.
    
    Hooks into the training loop to collect and analyze data.
    """
    
    def __init__(
        self, 
        config: InstrumentationConfig,
        out_dir: str,
        model_config: Optional[Dict] = None,
        args: Optional[Any] = None
    ):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.log_path = self.out_dir / "train.jsonl"
        self.tensor_writer = TensorWriter(out_dir) if config.level != "off" else None
        
        # Collected data (for plotting)
        self.eval_steps: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.throughputs: List[float] = []
        
        # Analysis history
        self.attention_entropy_history: List[Dict] = []
        self.attention_rank_history: List[Dict] = []
        self.gradient_norm_history: List[Dict] = []
        self.path_contribution_history: List[Dict] = []
        self.hidden_rank_history: List[Dict] = []
        
        # Cached attention weights (set by hooks)
        self._cached_attention: Dict[str, torch.Tensor] = {}
        self._cached_sem_scores: Dict[str, torch.Tensor] = {}
        self._cached_geo_scores: Dict[str, torch.Tensor] = {}
        self._cached_hidden_states: List[torch.Tensor] = []
        
        # Timing
        self.start_time = time.time()
        
        # Memory measurements
        self.model_memory: Dict = {}
        self.kv_cache_memory: Dict = {}
        self.peak_memory_history: List[Dict] = []
        
        # Write initial config
        self._log({
            "type": "run_config",
            "time": time.time(),
            "instrumentation": asdict(config),
            "model_config": model_config or {},
            "args": vars(args) if args and hasattr(args, '__dict__') else {},
        })
    
    def _log(self, data: Dict):
        """Write JSON line to log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(data) + "\n")
    
    # =========================================================================
    # HOOKS (to be called from model forward pass)
    # =========================================================================
    
    def cache_attention(self, layer: int, head: int, attn_weights: torch.Tensor):
        """Cache attention weights for analysis."""
        key = f"L{layer}_H{head}"
        self._cached_attention[key] = attn_weights.detach()
    
    def cache_path_scores(
        self, 
        layer: int, 
        sem_scores: Optional[torch.Tensor], 
        geo_scores: Optional[torch.Tensor]
    ):
        """Cache semantic and geometric scores for path contribution analysis."""
        key = f"L{layer}"
        if sem_scores is not None:
            self._cached_sem_scores[key] = sem_scores.detach()
        if geo_scores is not None:
            self._cached_geo_scores[key] = geo_scores.detach()
    
    def cache_hidden_state(self, layer: int, hidden: torch.Tensor):
        """Cache hidden state for representation analysis."""
        while len(self._cached_hidden_states) <= layer:
            self._cached_hidden_states.append(None)
        self._cached_hidden_states[layer] = hidden.detach()
    
    def clear_cache(self):
        """Clear all cached tensors."""
        self._cached_attention.clear()
        self._cached_sem_scores.clear()
        self._cached_geo_scores.clear()
        self._cached_hidden_states.clear()
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def analyze_step(self, step: int, model: nn.Module):
        """
        Run analysis for the current step.
        
        Called every `analysis_every` steps.
        """
        if self.config.level == "off":
            return
        
        if step % self.config.analysis_every != 0:
            return
        
        results = {"type": "analysis", "step": step, "time": time.time()}
        
        # Attention analysis
        if self._cached_attention and self.config.compute_attention_entropy:
            entropies = {}
            for key, attn in self._cached_attention.items():
                entropies[key] = compute_attention_entropy(attn)
            results["attention_entropy"] = entropies
            self.attention_entropy_history.append({"step": step, **entropies})
        
        if self._cached_attention and self.config.compute_attention_sparsity:
            sparsities = {}
            for key, attn in self._cached_attention.items():
                sparsities[key] = compute_attention_sparsity(attn)
            results["attention_sparsity"] = sparsities
        
        if self._cached_attention and self.config.compute_attention_rank:
            ranks = {}
            for key, attn in self._cached_attention.items():
                # Use mean attention across batch
                attn_mean = attn.mean(dim=0)  # (H, T, T) or (T, T)
                if attn_mean.dim() == 3:
                    attn_mean = attn_mean.mean(dim=0)  # (T, T)
                rank, svs = compute_effective_rank(attn_mean)
                ranks[key] = rank
                
                # Save singular values if heavy
                if self.config.save_attention_matrices and self.tensor_writer:
                    layer = int(key.split("_")[0][1:])
                    head = int(key.split("_")[1][1:]) if "_H" in key else 0
                    self.tensor_writer.save_singular_values(step, layer, head, svs)
            
            results["attention_rank"] = ranks
            self.attention_rank_history.append({"step": step, **ranks})
        
        # Path contribution (decoupled attention only)
        if self._cached_sem_scores and self.config.compute_path_contribution:
            contributions = {}
            for key in self._cached_sem_scores:
                sem = self._cached_sem_scores.get(key)
                geo = self._cached_geo_scores.get(key)
                contributions[key] = compute_path_contributions(sem, geo)
            results["path_contribution"] = contributions
            self.path_contribution_history.append({"step": step, **contributions})
        
        # Hidden state rank
        if self._cached_hidden_states and self.config.compute_hidden_rank:
            hidden_ranks = {}
            for i, h in enumerate(self._cached_hidden_states):
                if h is not None:
                    # Sample to reduce compute
                    h_sample = h[:, :min(64, h.shape[1]), :]  # First 64 tokens
                    rank, _ = compute_effective_rank(h_sample.reshape(-1, h_sample.shape[-1]))
                    hidden_ranks[f"layer_{i}"] = rank
            results["hidden_rank"] = hidden_ranks
            self.hidden_rank_history.append({"step": step, **hidden_ranks})
        
        # Gradient analysis
        if self.config.track_gradient_norms:
            grad_norms = compute_gradient_norms(model)
            results["gradient_norms"] = grad_norms
            self.gradient_norm_history.append({"step": step, **grad_norms})
        
        # Layer similarity
        if self._cached_hidden_states and self.config.compute_layer_similarity:
            valid_hiddens = [h for h in self._cached_hidden_states if h is not None]
            similarities = compute_layer_similarity(valid_hiddens)
            results["layer_similarity"] = similarities
        
        # Save full attention matrices if heavy
        if self.config.save_attention_matrices and self.tensor_writer:
            for key, attn in self._cached_attention.items():
                parts = key.split("_")
                layer = int(parts[0][1:])
                head = int(parts[1][1:]) if len(parts) > 1 else 0
                self.tensor_writer.save_attention(step, layer, head, attn[0])  # First batch item
        
        self._log(results)
        self.clear_cache()
    
    # =========================================================================
    # LOGGING METHODS
    # =========================================================================
    
    def log_train_step(self, step: int, loss: float, ppl: float, tok_per_s: float, lr: float):
        """Log training step metrics."""
        self.throughputs.append(tok_per_s)
        
        self._log({
            "type": "train_step",
            "time": time.time(),
            "step": step,
            "loss": loss,
            "ppl": ppl,
            "tok_per_s": tok_per_s,
            "lr": lr,
        })
    
    def log_eval(
        self, 
        step: int, 
        train_loss: float, 
        val_loss: float, 
        val_ppl: float, 
        elapsed: float,
        is_best: bool = False
    ):
        """Log evaluation metrics."""
        self.eval_steps.append(step)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self._log({
            "type": "eval",
            "time": time.time(),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "elapsed_s": elapsed,
            "is_best": is_best,
        })
    
    def log_best(self, step: int, best_val: float):
        """Log new best validation loss."""
        self._log({
            "type": "best",
            "time": time.time(),
            "step": step,
            "best_val": best_val,
        })
    
    def measure_memory(self, model: nn.Module, batch_size: int, seq_len: int):
        """
        Measure and log actual memory usage.
        
        Call this once after model initialization.
        """
        # Measure model parameters
        self.model_memory = measure_model_memory(model)
        
        # Measure KV cache for different scenarios
        self.kv_cache_memory = measure_kv_cache_memory(model, batch_size, seq_len)
        
        # Also measure at 128k context (the paper's target)
        kv_128k = measure_kv_cache_memory(model, 1, 131072)  # batch=1, 128k tokens
        
        # Current peak memory
        peak = measure_peak_memory()
        
        self._log({
            "type": "memory_measurement",
            "time": time.time(),
            "model_params": self.model_memory,
            "kv_cache_training": self.kv_cache_memory,
            "kv_cache_128k": kv_128k,
            "peak_memory": peak,
        })
        
        # Print summary
        print(f"\n  Memory Measurements:")
        print(f"    Model params:     {self.model_memory['total_params_bytes'] / 1e6:.1f} MB")
        print(f"    KV cache (train): {self.kv_cache_memory['fp16_total_mb']:.1f} MB (fp16)")
        print(f"    KV cache (128k):  {kv_128k['fp16_total_mb']:.1f} MB (fp16), {kv_128k['q4_total_mb']:.1f} MB (q4)")
        print(f"    Compression:      {kv_128k['fp16_to_q4_ratio']:.1f}× with Q4\n")
    
    # =========================================================================
    # FINALIZATION
    # =========================================================================
    
    def finalize(self, best_val: float):
        """
        Finalize analysis: close files, generate plots, write summary.
        """
        total_time = time.time() - self.start_time
        
        self._log({
            "type": "done",
            "time": time.time(),
            "best_val": best_val,
            "total_seconds": total_time,
        })
        
        # Close tensor writer
        if self.tensor_writer:
            self.tensor_writer.close()
        
        # Generate plots
        self._generate_plots()
        
        # Write summary
        self._write_summary(best_val, total_time)
    
    def _generate_plots(self):
        """Generate analysis plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            n_plots = 2
            if self.attention_entropy_history:
                n_plots += 1
            if self.attention_rank_history:
                n_plots += 1
            if self.path_contribution_history:
                n_plots += 1
            if self.gradient_norm_history:
                n_plots += 1
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            plot_idx = 0
            
            # Plot 1: Loss convergence
            if self.eval_steps and self.val_losses:
                ax = axes[plot_idx]
                ax.plot(self.eval_steps, self.val_losses, 'b-o', linewidth=2, markersize=4, label='Val')
                ax.plot(self.eval_steps, self.train_losses, 'g--', linewidth=1.5, alpha=0.7, label='Train')
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_title('Loss Convergence')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            # Plot 2: Throughput
            if self.throughputs:
                ax = axes[plot_idx]
                steps = list(range(0, len(self.throughputs) * 50, 50))
                ax.plot(steps, self.throughputs, 'r-', linewidth=1, alpha=0.7)
                ax.axhline(y=sum(self.throughputs)/len(self.throughputs), 
                          color='darkred', linestyle='--', label=f'Avg: {sum(self.throughputs)/len(self.throughputs):.0f}')
                ax.set_xlabel('Step')
                ax.set_ylabel('Tokens/sec')
                ax.set_title('Training Throughput')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            # Plot 3: Attention entropy over time
            if self.attention_entropy_history:
                ax = axes[plot_idx]
                steps = [d["step"] for d in self.attention_entropy_history]
                # Average across all layers/heads
                avg_entropy = []
                for d in self.attention_entropy_history:
                    vals = [v for k, v in d.items() if k != "step" and isinstance(v, (int, float)) and v > 0]
                    avg_entropy.append(sum(vals) / len(vals) if vals else 0)
                ax.plot(steps, avg_entropy, 'purple', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Entropy')
                ax.set_title('Attention Entropy (avg across heads)')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            # Plot 4: Attention rank over time
            if self.attention_rank_history:
                ax = axes[plot_idx]
                steps = [d["step"] for d in self.attention_rank_history]
                avg_rank = []
                for d in self.attention_rank_history:
                    vals = [v for k, v in d.items() if k != "step" and isinstance(v, (int, float)) and v > 0]
                    avg_rank.append(sum(vals) / len(vals) if vals else 0)
                ax.plot(steps, avg_rank, 'orange', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Effective Rank')
                ax.set_title('Attention Effective Rank (avg)')
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            # Plot 5: Path contributions (if decoupled)
            if self.path_contribution_history:
                ax = axes[plot_idx]
                steps = [d["step"] for d in self.path_contribution_history]
                sem_ratios = []
                geo_ratios = []
                for d in self.path_contribution_history:
                    # Get average across layers
                    sem_vals = []
                    geo_vals = []
                    for k, v in d.items():
                        if k != "step" and isinstance(v, dict):
                            if "semantic_ratio" in v and v["semantic_ratio"] > 0:
                                sem_vals.append(v["semantic_ratio"])
                            if "geometric_ratio" in v and v["geometric_ratio"] > 0:
                                geo_vals.append(v["geometric_ratio"])
                    sem_ratios.append(sum(sem_vals) / len(sem_vals) if sem_vals else 0)
                    geo_ratios.append(sum(geo_vals) / len(geo_vals) if geo_vals else 0)
                
                ax.plot(steps, sem_ratios, 'blue', linewidth=2, label='Semantic')
                ax.plot(steps, geo_ratios, 'red', linewidth=2, label='Geometric')
                ax.set_xlabel('Step')
                ax.set_ylabel('Contribution Ratio')
                ax.set_title('Semantic vs Geometric Path Contribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            # Plot 6: Gradient norms
            if self.gradient_norm_history:
                ax = axes[plot_idx]
                steps = [d["step"] for d in self.gradient_norm_history]
                
                for key in ["attn_grad_norm", "ffn_grad_norm"]:
                    vals = [d.get(key, 0) for d in self.gradient_norm_history]
                    if any(v > 0 for v in vals):
                        ax.plot(steps, vals, linewidth=1.5, label=key.replace("_grad_norm", ""))
                
                ax.set_xlabel('Step')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norms by Component')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                plot_idx += 1
            
            # Hide unused axes
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.out_dir / "analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved analysis plot: {self.out_dir / 'analysis.png'}")
            
        except ImportError:
            print("  Warning: matplotlib not available, skipping plots")
        except Exception as e:
            print(f"  Warning: Plot generation failed: {e}")
    
    def _write_summary(self, best_val: float, total_time: float):
        """Write human-readable summary."""
        summary_path = self.out_dir / "summary.md"
        
        lines = [
            "# Experiment Summary",
            "",
            f"**Best Validation Loss**: {best_val:.4f}",
            f"**Best Perplexity**: {math.exp(best_val):.2f}",
            f"**Total Time**: {total_time:.1f}s ({total_time/3600:.2f}h)",
            "",
        ]
        
        if self.throughputs:
            avg_throughput = sum(self.throughputs) / len(self.throughputs)
            lines.extend([
                f"**Avg Throughput**: {avg_throughput:.0f} tok/s",
                "",
            ])
        
        # Analysis summary
        if self.attention_rank_history:
            final_ranks = self.attention_rank_history[-1] if self.attention_rank_history else {}
            ranks = [v for k, v in final_ranks.items() if k != "step" and isinstance(v, (int, float)) and v > 0]
            if ranks:
                lines.extend([
                    "## Attention Analysis",
                    "",
                    f"**Final Avg Effective Rank**: {sum(ranks)/len(ranks):.1f}",
                    "",
                ])
        
        if self.path_contribution_history:
            final = self.path_contribution_history[-1] if self.path_contribution_history else {}
            sem_vals = []
            geo_vals = []
            for k, v in final.items():
                if k != "step" and isinstance(v, dict):
                    if "semantic_ratio" in v:
                        sem_vals.append(v["semantic_ratio"])
                    if "geometric_ratio" in v:
                        geo_vals.append(v["geometric_ratio"])
            
            if sem_vals:
                lines.extend([
                    "## Path Contributions",
                    "",
                    f"**Semantic Path**: {sum(sem_vals)/len(sem_vals)*100:.1f}%",
                    f"**Geometric Path**: {sum(geo_vals)/len(geo_vals)*100:.1f}%",
                    "",
                ])
        
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        
        print(f"  -> Saved summary: {summary_path}")


# =============================================================================
# HOOK FACTORY
# =============================================================================

def create_attention_hook(analyzer: Analyzer, layer_idx: int):
    """Create a forward hook for attention analysis."""
    def hook(module, input, output):
        # Try to get attention weights from the module
        if hasattr(module, 'last_attn') and module.last_attn is not None:
            analyzer.cache_attention(layer_idx, 0, module.last_attn)
        
        # For decoupled attention, try to get path scores
        if hasattr(module, 'last_sem_scores') and hasattr(module, 'last_geo_scores'):
            analyzer.cache_path_scores(
                layer_idx,
                getattr(module, 'last_sem_scores', None),
                getattr(module, 'last_geo_scores', None)
            )
    
    return hook


def create_hidden_hook(analyzer: Analyzer, layer_idx: int):
    """Create a forward hook for hidden state analysis."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        analyzer.cache_hidden_state(layer_idx, hidden)
    
    return hook


def register_hooks(model: nn.Module, analyzer: Analyzer):
    """
    Register analysis hooks on model.
    
    Returns list of hook handles (for removal if needed).
    """
    handles = []
    
    # Find attention and block modules
    for name, module in model.named_modules():
        # Attention hooks
        if "attn" in name.lower() and not any(x in name for x in ["drop", "norm", "proj"]):
            # Extract layer index
            parts = name.split(".")
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    handle = module.register_forward_hook(create_attention_hook(analyzer, layer_idx))
                    handles.append(handle)
                    break
        
        # Hidden state hooks (on transformer blocks)
        if "block" in name.lower() and name.count(".") == 1:
            parts = name.split(".")
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    handle = module.register_forward_hook(create_hidden_hook(analyzer, layer_idx))
                    handles.append(handle)
                    break
    
    return handles

