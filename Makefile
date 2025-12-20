.PHONY: train paper_all paper_data paper_experiments paper_analyze paper_baseline paper_bottleneck paper_decoupled paper_gqa paper_deep paper_smoke print_config install_deps test test_verbose

# Default python used by utility targets (tests, analysis helpers, etc.)
PY ?= python3.12

train:
	# python3 v1_gradient_grouping.py --mode baseline 
	# python3 v1_gradient_grouping.py --mode grouped --sim-threshold 0.9
	# python3 v1_gradient_grouping.py --mode coarse_to_fine

	# python3 v2_optimized_gradient_grouping.py --mode baseline 
	# python3 v2_optimized_gradient_grouping.py --sim-threshold 0.9 --mode grouped 
	# python3 v2_optimized_gradient_grouping.py --mode coarse_to_fine
	
	# python3 v3_adaptive_lowrank.py
	# python3 v3_adaptive_lowrank.py --epochs 30
	# python3 v3_adaptive_lowrank.py --init-rank1 128 --init-rank2 64

	# python3 v4_adaptive_lowrank_rand.py
	# python3 v4_adaptive_lowrank_rand.py --epochs 30
	# python3 v4_adaptive_lowrank_rand.py --init-rank1 128 --init-rank2 64
	
	# python3 v5_multi_layer_adaptive.py
	# python3 v5_multi_layer_adaptive.py --epochs 30
	# python3 v5_multi_layer_adaptive.py --init-rank1 128 --init-rank2 64
	# python3 v5_1_multi_layer_adaptive_smooth.py --epochs 30
	# python3 v5_1_multi_layer_adaptive_smooth.py --init-rank1 128 --init-rank2 64
	
	# python3 v6_lowrank_backprop.py
	# python3 v6_lowrank_backprop.py --epochs 30
	# python3 v6_lowrank_backprop.py --init-rank1 128 --init-rank2 64
	
	# python3 v7_transformer_dense_baseline.py --epochs 5
	# python3 v7_transformer_dense_baseline.py --epochs 25
	# python3 v7_transformer_dense_baseline.py --epochs 50
	# python3 v7_1_transformer_lowrank.py --epochs 15
	# python3 v7_1_transformer_lowrank.py --epochs 20 --init-rank 32
	# python3 v7_1_transformer_lowrank.py --epochs 30 --init-rank 64
	# python3 v7_2_transformer_lowrank_ema.py --epochs 15
	# python3 v7_2_transformer_lowrank_ema.py --epochs 20 --init-rank 32
	# python3 v7_2_transformer_lowrank_ema.py --epochs 30 --init-rank 64
	# python3 v7_3_transformer_lowrank_autograd.py --epochs 15
	# python3 v7_3_transformer_lowrank_autograd.py --epochs 20 --init-rank 32
	# python3 v7_3_transformer_lowrank_autograd.py --epochs 30 --init-rank 64
	# python3 v7_4_transformer_lowrank_sympathetic_ema.py --epochs 15
	# python3 v7_4_transformer_lowrank_sympathetic_ema.py --epochs 20 --init-rank 32
	# python3 v7_4_transformer_lowrank_sympathetic_ema.py --epochs 30 --init-rank 64
	# python3 v7_5_transformer_lowrank_adaptive.py --epochs 15
	# python3 v7_5_transformer_lowrank_adaptive.py --epochs 20 --init-rank 32
	# python3 v7_5_transformer_lowrank_adaptive.py --epochs 30 --init-rank 64
	
	# python3 v8_transformer_lowrank_spectral.py --epochs 15
	# python3 v8_transformer_lowrank_spectral.py --epochs 20 --init-rank 32
	# python3 v8_transformer_lowrank_spectral.py --epochs 30 --init-rank 64
	
	# python3 v9_transformer_lowrank_spectral_bidirectional.py --epochs 30 --init-rank 64 --data-file wiki.train.tokens --log-file v9_log.jsonl

	# python3 v10_transformer_lowrank_scaled.py \
	# 	--data-file wiki.train.tokens \
	# 	--log-file v10_log.jsonl \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--d-model 512 \
	# 	--n-layers 6 \
	# 	--n-heads 8 \
	# 	--d-ff 2048 \
	# 	--block-size 256 \
	# 	--batch-size 32 \
	# 	--steps-per-epoch 200

	# python3 v11_transformer_lowrank_momentum.py \
	# 	--data-file wiki.train.tokens \
	# 	--log-file v11_log.jsonl \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--d-model 512 \
	# 	--n-layers 6 \
	# 	--n-heads 8 \
	# 	--d-ff 2048 \
	# 	--block-size 256 \
	# 	--batch-size 32 \
	# 	--steps-per-epoch 200

	# python3 v13_transformer_lowrank_lazy_svd_adaptive.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--log-file v13_log.jsonl

	# python3 v14_transformer_adaptive_heads_lowrank.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--max-rank 64 \
	# 	--log-file v14_log.jsonl

	# python3 v15_transformer_lowrank_adaptive_grad.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--log-file v15_log.jsonl

	# python3.12 v16_transformer_lowrank_pressure_cooker.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--max-rank 64 \
	# 	--log-file v16_log.jsonl

	# python3.12 v16_transformer_lowrank_pressure_cooker.py \
	# 	--data-file wiki.train.tokens \
	# 	--epochs 30 \
	# 	--init-rank 64 \
	# 	--max-rank 64 \
	# 	--min-rank 4 \
	# 	--compute-target 0.45 \
	# 	--warmup-epochs 1 \
	# 	--pressure-step 0.20 \
	# 	--energy-target-lo 0.85 \
	# 	--lambda-scale 100 \
	# 	--prune-every 200 \
	# 	--svd-interval 200 \
	# 	--log-file v16_log.jsonl

	# python3.12 v17_transformer_lowrank_pressure_cooker.py \
	# 	--mode train \
	# 	--data wiki.train.tokens \
	# 	--out-dir runs/v17

	# python3.12 v17_transformer_lowrank_pressure_cooker.py \
	# 	--mode generate \
	# 	--ckpt runs/v17/best.pt \
	# 	--prompt "Once upon a time" \
	# 	--max-new-tokens 400

	# python3.12 v18_transformer_lowrank_alrt.py \
	# 	--mode train --data wiki.train.tokens --out-dir runs/v18

	# python3.12 v18_transformer_lowrank_alrt.py \
	# 	--mode generate --ckpt runs/v18/best.pt \
	# 	--prompt "Once upon a time" --max-new-tokens 400 \
	# 	--temperature 0.8 --top-k 50


	# python3.12 v19_transformer_attn_bottleneck.py \
	# 	--data ./wiki.train.tokens \
	# 	--out-dir runs/v19_baseline \
	# 	--attn-dim 512

	# python3.12 v19_transformer_attn_bottleneck.py \
	# 	--data ./wiki.train.tokens \
	# 	--out-dir runs/v19_attn128 \
	# 	--attn-dim 128

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null \
		--attn-dim 128 --null-attn

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null_tie \
		--attn-dim 128 --null-attn --tie-qk

v20:
	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_baseline \
		--attn-dim 512 \
		--embed-dim 512

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_attn128 \
		--attn-dim 128 \
		--embed-dim 512

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_embed256 \
		--attn-dim 512 \
		--embed-dim 256

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_attn128_embed256 \
		--attn-dim 128 \
		--embed-dim 256

	python3.12 v20_transformer_lexical_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v20_attn128_embed128 \
		--attn-dim 128 \
		--embed-dim 128

bottleneck_attention:
	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
	 	--out-dir runs/v19_baseline \
	 	--attn-dim 512

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128 \
		--attn-dim 128

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null \
		--attn-dim 128 --null-attn

	python3.12 v19_transformer_attn_bottleneck.py \
		--data ./wiki.train.tokens \
		--out-dir runs/v19_attn128_null_tie \
		--attn-dim 128 --null-attn --tie-qk

decoupled_bottleneck:
	python3.12 v21_transformer_decoupled_bottleneck.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_bottleneck_rope \
		--attn-mode bottleneck \
		--attn-dim 128 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn

	python3.12 v21_transformer_decoupled_bottleneck.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64 \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--attn-dim 128 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn

	python3.12 v21_transformer_decoupled_bottleneck.py \
		--mode sample \
		--ckpt runs/v21_decoupled_sem32_geo64/best.pt \
		--prompt-tokens "1 2 3 4 5" \
		--max-new-tokens 200 \
		--kv-cache q4_0

prepare_fineweb:
	python3.12 prepare_fineweb.py --out fineweb_100m.tokens

support:
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_gqa_kv2_parammatch \
		--seed 1337 \
		--device mps \
		--attn-mode gqa \
		--kv-head 2 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2059 \
		--block 256 \
		--embed-dim 512 \
		--attn-dim 128 \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_small_d128_standard \
		--seed 1337 \
		--device mps \
		--attn-mode standard \
		--d-model 128 \
		--layers 6 \
		--n-head 4 \
		--d-ff 512 \
		--block 256 \
		--embed-dim 128 \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64 \
		--null-attn

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64_block1024 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--embed-dim 128 \
		--attn-dim 128 \
		--tie-qk \
		--null-attn \
		--steps 1200 \
		--eval-every 200 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 8

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_decoupled_sem32_geo64_block2048 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 2048 \
		--embed-dim 128 \
		--attn-dim 128 \
		--tie-qk \
		--null-attn \
		--steps 800 \
		--eval-every 200 \
		--eval-iters 10 \
		--lr 3e-4 \
		--batch-size 4

bigboy:
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v21_fineweb_baseline \
		--attn-mode standard \
		--d-model 512 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--batch-size 16 \
		--steps 6000 \
		--eval-every 500 \
		--lr 3e-4

	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v21_fineweb_decoupled \
		--attn-mode decoupled \
		--d-model 512 \
		--n-head 8 \
		--sem-dim 32 \
		--geo-dim 64 \
		--attn-dim 128 \
		--d-ff 2048 \
		--block 1024 \
		--batch-size 16 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 500 \
		--lr 3e-4

suggestions:
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/v21_combined_baseline_96 \
		--attn-mode bottleneck \
		--attn-dim 96 \
		--null-attn

# =============================================================================
# PAPER EXPERIMENTS - FINEWEB-EDU ONLY (NO WIKITEXT)
# =============================================================================
# All experiments use FineWeb-Edu for realistic, non-overfit training.
# Deep instrumentation enabled for understanding attention mechanics.
#
# MODEL SIZE PRESETS (choose one):
#   make paper_all SIZE=tiny    ~25M params  - Fast iteration (MPS friendly)
#   make paper_all SIZE=small   ~125M params - Respectable (MPS feasible)
#   make paper_all SIZE=medium  ~350M params - Production-relevant (needs GPU)
#   make paper_all SIZE=large   ~760M params - Serious scale (needs A100)
#
# Each experiment produces:
#   - train.jsonl      (JSONL log with all metrics)
#   - analysis.h5      (Tensor data: attention matrices, SVD, etc.)
#   - analysis.png     (Auto-generated analysis plots)
#   - summary.md       (Human-readable results)
#   - best.pt, last.pt (Checkpoints)
#
# Run: make paper_all SIZE=small
# =============================================================================

# -----------------------------------------------------------------------------
# MODEL SIZE CONFIGURATIONS
# -----------------------------------------------------------------------------
# Default to small - respectable but MPS-feasible
SIZE ?= small

# Tiny: ~25M params (fast iteration, MPS friendly)
ifeq ($(SIZE),tiny)
D_MODEL := 512
N_LAYER := 6
N_HEAD := 8
D_FF := 2048
BLOCK := 1024
BATCH := 16
STEPS := 6000
EVAL_EVERY := 200
endif

# Small: ~125M params (respectable results)
ifeq ($(SIZE),small)
D_MODEL := 768
N_LAYER := 12
N_HEAD := 12
D_FF := 3072
BLOCK := 1024
BATCH := 8
STEPS := 10000
EVAL_EVERY := 500
endif

# Medium: ~350M params (production-relevant)
ifeq ($(SIZE),medium)
D_MODEL := 1024
N_LAYER := 24
N_HEAD := 16
D_FF := 4096
BLOCK := 2048
BATCH := 4
STEPS := 15000
EVAL_EVERY := 500
endif

# Large: ~760M params (serious scale, needs A100+)
ifeq ($(SIZE),large)
D_MODEL := 1536
N_LAYER := 24
N_HEAD := 16
D_FF := 6144
BLOCK := 2048
BATCH := 2
STEPS := 20000
EVAL_EVERY := 1000
endif

# Derived dimensions for decoupled attention (scale with model)
SEM_DIM := $(shell echo "$(D_MODEL) / 16" | bc)
GEO_DIM := $(shell echo "$(D_MODEL) / 8" | bc)
ATTN_DIM := $(shell echo "$(SEM_DIM) + $(GEO_DIM)" | bc)
KV_HEAD := $(shell echo "$(N_HEAD) / 4" | bc)

# Print config
print_config:
	@echo "=============================================="
	@echo "Model Size: $(SIZE)"
	@echo "=============================================="
	@echo "  d_model:    $(D_MODEL)"
	@echo "  n_layer:    $(N_LAYER)"
	@echo "  n_head:     $(N_HEAD)"
	@echo "  d_ff:       $(D_FF)"
	@echo "  block:      $(BLOCK)"
	@echo "  batch:      $(BATCH)"
	@echo "  steps:      $(STEPS)"
	@echo "----------------------------------------------"
	@echo "  sem_dim:    $(SEM_DIM)"
	@echo "  geo_dim:    $(GEO_DIM)"
	@echo "  attn_dim:   $(ATTN_DIM)"
	@echo "  kv_head:    $(KV_HEAD)"
	@echo "=============================================="

# Master target: Run ALL paper experiments with instrumentation
paper_all: print_config paper_data paper_experiments paper_analyze
	@echo ""
	@echo "=============================================="
	@echo "  ALL PAPER EXPERIMENTS COMPLETE! (SIZE=$(SIZE))"
	@echo "=============================================="
	@echo ""
	@echo "Results saved in runs/$(SIZE)_*/"
	@echo "Analysis plots in assets/analysis/"
	@echo ""
	@echo "To re-analyze: make paper_analyze"
	@echo "=============================================="

# Prepare FineWeb dataset (scale with model size)
paper_data:
ifeq ($(SIZE),tiny)
	@echo ">>> Preparing FineWeb-Edu dataset (100M tokens)..."
	@if [ ! -f fineweb_100m.tokens ]; then \
		python3.12 prepare_fineweb.py --tokens 100M --output fineweb_100m.tokens; \
	fi
FINEWEB_DATA := fineweb_100m.tokens
else ifeq ($(SIZE),small)
	@echo ">>> Preparing FineWeb-Edu dataset (500M tokens)..."
	@if [ ! -f fineweb_500m.tokens ]; then \
		python3.12 prepare_fineweb.py --tokens 500M --output fineweb_500m.tokens; \
	fi
FINEWEB_DATA := fineweb_500m.tokens
else
	@echo ">>> Preparing FineWeb-Edu dataset (1B tokens)..."
	@if [ ! -f fineweb_1b.tokens ]; then \
		python3.12 prepare_fineweb.py --tokens 1B --output fineweb_1b.tokens; \
	fi
FINEWEB_DATA := fineweb_1b.tokens
endif

# Data file selection based on size
ifeq ($(SIZE),tiny)
DATA_FILE := fineweb_100m.tokens
else ifeq ($(SIZE),small)
DATA_FILE := fineweb_500m.tokens
else
DATA_FILE := fineweb_1b.tokens
endif

# Run all core experiments
paper_experiments: paper_baseline paper_bottleneck paper_decoupled paper_gqa
	@echo ""
	@echo "[Paper Experiments] All 4 experiments complete! (SIZE=$(SIZE))"
	@echo ""

# Post-training analysis
paper_analyze:
	@echo ">>> Running post-training analysis..."
	python3.12 analyze_run.py --all
	python3.12 generate_paper_figures.py
	@echo "Analysis complete! Check assets/analysis/"

# -----------------------------------------------------------------------------
# INDIVIDUAL EXPERIMENTS (using SIZE variables)
# -----------------------------------------------------------------------------

# Standard Baseline: Full-rank attention
paper_baseline: 
	@echo ""
	@echo ">>> [1/4] Standard Baseline ($(SIZE): d=$(D_MODEL), L=$(N_LAYER))..."
	@echo "=============================================="
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data $(DATA_FILE) \
		--out-dir runs/$(SIZE)_baseline \
		--seed 1337 \
		--device mps \
		--tokenizer tiktoken \
		--attn-mode standard \
		--d-model $(D_MODEL) \
		--layers $(N_LAYER) \
		--n-head $(N_HEAD) \
		--d-ff $(D_FF) \
		--block $(BLOCK) \
		--embed-dim $(D_MODEL) \
		--steps $(STEPS) \
		--eval-every $(EVAL_EVERY) \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size $(BATCH) \
		--instrument medium \
		--analysis-every 100

# Bottleneck: Compressed attention (d_attn = d_model/8)
paper_bottleneck: 
	@echo ""
	@echo ">>> [2/4] Bottleneck ($(SIZE): d_attn=$(ATTN_DIM))..."
	@echo "=============================================="
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data $(DATA_FILE) \
		--out-dir runs/$(SIZE)_bottleneck \
		--seed 1337 \
		--device mps \
		--tokenizer tiktoken \
		--attn-mode bottleneck \
		--attn-dim $(ATTN_DIM) \
		--d-model $(D_MODEL) \
		--layers $(N_LAYER) \
		--n-head $(N_HEAD) \
		--d-ff $(D_FF) \
		--block $(BLOCK) \
		--embed-dim $(D_MODEL) \
		--null-attn \
		--steps $(STEPS) \
		--eval-every $(EVAL_EVERY) \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size $(BATCH) \
		--instrument medium \
		--analysis-every 100

# Decoupled: Semantic + Geometric split
paper_decoupled: 
	@echo ""
	@echo ">>> [3/4] Decoupled ($(SIZE): sem=$(SEM_DIM), geo=$(GEO_DIM))..."
	@echo "=============================================="
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data $(DATA_FILE) \
		--out-dir runs/$(SIZE)_decoupled \
		--seed 1337 \
		--device mps \
		--tokenizer tiktoken \
		--attn-mode decoupled \
		--sem-dim $(SEM_DIM) \
		--geo-dim $(GEO_DIM) \
		--attn-dim $(ATTN_DIM) \
		--d-model $(D_MODEL) \
		--layers $(N_LAYER) \
		--n-head $(N_HEAD) \
		--d-ff $(D_FF) \
		--block $(BLOCK) \
		--embed-dim $(D_MODEL) \
		--tie-qk \
		--null-attn \
		--steps $(STEPS) \
		--eval-every $(EVAL_EVERY) \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size $(BATCH) \
		--instrument medium \
		--analysis-every 100

# GQA: Grouped Query Attention
paper_gqa: 
	@echo ""
	@echo ">>> [4/4] GQA ($(SIZE): $(N_HEAD)Q/$(KV_HEAD)KV heads)..."
	@echo "=============================================="
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data $(DATA_FILE) \
		--out-dir runs/$(SIZE)_gqa \
		--seed 1337 \
		--device mps \
		--tokenizer tiktoken \
		--attn-mode gqa \
		--kv-head $(KV_HEAD) \
		--attn-dim $(D_MODEL) \
		--d-model $(D_MODEL) \
		--layers $(N_LAYER) \
		--n-head $(N_HEAD) \
		--d-ff $(D_FF) \
		--block $(BLOCK) \
		--embed-dim $(D_MODEL) \
		--steps $(STEPS) \
		--eval-every $(EVAL_EVERY) \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size $(BATCH) \
		--instrument medium \
		--analysis-every 100

# -----------------------------------------------------------------------------
# HEAVY INSTRUMENTATION (for deep analysis, slower)
# -----------------------------------------------------------------------------

paper_deep: 
	@echo ">>> Running with HEAVY instrumentation ($(SIZE))..."
	@echo "This will be ~30% slower but capture full attention matrices."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data $(DATA_FILE) \
		--out-dir runs/$(SIZE)_decoupled_deep \
		--seed 1337 \
		--device mps \
		--tokenizer tiktoken \
		--attn-mode decoupled \
		--sem-dim $(SEM_DIM) \
		--geo-dim $(GEO_DIM) \
		--attn-dim $(ATTN_DIM) \
		--d-model $(D_MODEL) \
		--layers $(N_LAYER) \
		--n-head $(N_HEAD) \
		--d-ff $(D_FF) \
		--block $(BLOCK) \
		--embed-dim $(D_MODEL) \
		--tie-qk \
		--null-attn \
		--steps $(shell echo "$(STEPS) / 2" | bc) \
		--eval-every $(EVAL_EVERY) \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size $(BATCH) \
		--instrument heavy \
		--analysis-every 50

# -----------------------------------------------------------------------------
# QUICK SMOKE TEST (verify setup works)
# -----------------------------------------------------------------------------

paper_smoke:
	@echo ">>> Quick smoke test (100 steps)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data $(DATA_FILE) \
		--out-dir runs/smoke_test \
		--seed 1337 \
		--device mps \
		--tokenizer tiktoken \
		--attn-mode decoupled \
		--sem-dim $(SEM_DIM) \
		--geo-dim $(GEO_DIM) \
		--attn-dim $(ATTN_DIM) \
		--d-model $(D_MODEL) \
		--layers $(N_LAYER) \
		--n-head $(N_HEAD) \
		--d-ff $(D_FF) \
		--block $(BLOCK) \
		--embed-dim $(D_MODEL) \
		--tie-qk \
		--null-attn \
		--steps 100 \
		--eval-every 50 \
		--eval-iters 5 \
		--lr 3e-4 \
		--batch-size $(BATCH) \
		--instrument light
	@echo "Smoke test passed!"

# =============================================================================
# SETUP & DEPENDENCIES
# =============================================================================

install_deps:
	@echo ">>> Installing Python dependencies..."
	pip install torch numpy matplotlib tqdm h5py
	@echo ">>> Installing FineWeb dependencies..."
	pip install datasets tiktoken
	@echo "Done!"

# =============================================================================
# TESTS
# =============================================================================

# Fast unit test run: only files in tests/ matching test_*.py
test:
	$(PY) -m unittest discover -s tests -p "test_*.py" -q

# Verbose unit test run (useful when debugging failures)
test_verbose:
	$(PY) -m unittest discover -s tests -p "test_*.py" -v

# (Old WikiText-based targets removed - use paper_all for FineWeb experiments)

visualize: visualize_plots visualize_heatmaps
	@echo "=============================================="
	@echo "All visualizations complete!"
	@echo "Check assets/ for figures and heatmaps."
	@echo "=============================================="

visualize_plots:
	@echo ">>> Generating convergence and memory plots..."
	@mkdir -p assets
	python3.12 plot_results.py || echo "plot_results.py failed (may need log files)"
	python3.12 plot_memory.py || echo "plot_memory.py failed"
	@echo "Plots saved to assets/"

visualize_heatmaps:
	@echo ">>> Generating attention heatmaps for all checkpoints..."
	@mkdir -p assets/heatmaps
	@# Find all best.pt checkpoints and generate heatmaps
	@for ckpt in runs/*/best.pt; do \
		if [ -f "$$ckpt" ]; then \
			name=$$(dirname "$$ckpt" | sed 's/runs\///'); \
			echo "  Processing $$name..."; \
			for layer in 0 2 5; do \
				for head in 0 3 7; do \
					python3.12 vis_heatmap.py \
						--ckpt "$$ckpt" \
						--layer $$layer \
						--head $$head \
						--seq-len 16 2>/dev/null || true; \
				done; \
			done; \
		fi; \
	done
	@echo "Heatmaps saved to assets/heatmaps/"

# Quick heatmap for a single checkpoint
heatmap:
	@if [ -z "$(CKPT)" ]; then \
		echo "Usage: make heatmap CKPT=runs/v21_combined_baseline_96/best.pt"; \
		exit 1; \
	fi
	@mkdir -p assets/heatmaps
	python3.12 vis_heatmap.py --ckpt $(CKPT) --layer 0 --head 0 --seq-len 16
	python3.12 vis_heatmap.py --ckpt $(CKPT) --layer 2 --head 0 --seq-len 16
	python3.12 vis_heatmap.py --ckpt $(CKPT) --layer 5 --head 0 --seq-len 16

# ============================================================================
# ABLATION STUDIES: (d_sem, d_geo) Split Justification
# ============================================================================
# Tests different semantic/geometric dimension splits (all sum to 96)
# to justify the 32/64 choice empirically.

ablation_sem_geo: ablation_16_80 ablation_24_72 ablation_32_64 ablation_48_48 ablation_64_32
	@echo "=============================================="
	@echo "(d_sem, d_geo) Ablation Complete!"
	@echo "Run 'make analyze_ablation' to generate results table."
	@echo "=============================================="

ablation_16_80:
	@echo ">>> Running Decoupled (16/80) ablation..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/ablation_sem16_geo80 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 16 \
		--geo-dim 80 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

ablation_24_72:
	@echo ">>> Running Decoupled (24/72) ablation..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/ablation_sem24_geo72 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 24 \
		--geo-dim 72 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

ablation_32_64:
	@echo ">>> Running Decoupled (32/64) ablation..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/ablation_sem32_geo64 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

ablation_48_48:
	@echo ">>> Running Decoupled (48/48) ablation..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/ablation_sem48_geo48 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 48 \
		--geo-dim 48 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

ablation_64_32:
	@echo ">>> Running Decoupled (64/32) ablation..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/ablation_sem64_geo32 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 64 \
		--geo-dim 32 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

# ============================================================================
# MULTI-SEED VALIDATION: Confidence Intervals
# ============================================================================
# Runs key experiments with 3 different seeds for statistical validity.

multiseed_validation: multiseed_baseline multiseed_combined96 multiseed_decoupled
	@echo "=============================================="
	@echo "Multi-seed Validation Complete!"
	@echo "Run 'make analyze_multiseed' to compute confidence intervals."
	@echo "=============================================="

# Baseline with 3 seeds
multiseed_baseline: multiseed_baseline_s1337 multiseed_baseline_s42 multiseed_baseline_s123

multiseed_baseline_s1337:
	@echo ">>> Running Baseline (seed=1337)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_baseline_s1337 \
		--seed 1337 \
		--device mps \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

multiseed_baseline_s42:
	@echo ">>> Running Baseline (seed=42)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_baseline_s42 \
		--seed 42 \
		--device mps \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

multiseed_baseline_s123:
	@echo ">>> Running Baseline (seed=123)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_baseline_s123 \
		--seed 123 \
		--device mps \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

# Combined 96 with 3 seeds
multiseed_combined96: multiseed_combined96_s1337 multiseed_combined96_s42 multiseed_combined96_s123

multiseed_combined96_s1337:
	@echo ">>> Running Combined 96 (seed=1337)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_combined96_s1337 \
		--seed 1337 \
		--device mps \
		--attn-mode bottleneck \
		--attn-dim 96 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

multiseed_combined96_s42:
	@echo ">>> Running Combined 96 (seed=42)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_combined96_s42 \
		--seed 42 \
		--device mps \
		--attn-mode bottleneck \
		--attn-dim 96 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

multiseed_combined96_s123:
	@echo ">>> Running Combined 96 (seed=123)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_combined96_s123 \
		--seed 123 \
		--device mps \
		--attn-mode bottleneck \
		--attn-dim 96 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

# Decoupled 32/64 with 3 seeds
multiseed_decoupled: multiseed_decoupled_s1337 multiseed_decoupled_s42 multiseed_decoupled_s123

multiseed_decoupled_s1337:
	@echo ">>> Running Decoupled 32/64 (seed=1337)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_decoupled_s1337 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

multiseed_decoupled_s42:
	@echo ">>> Running Decoupled 32/64 (seed=42)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_decoupled_s42 \
		--seed 42 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

multiseed_decoupled_s123:
	@echo ">>> Running Decoupled 32/64 (seed=123)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data wiki.train.tokens \
		--out-dir runs/multiseed_decoupled_s123 \
		--seed 123 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 6000 \
		--eval-every 200 \
		--eval-iters 50 \
		--lr 3e-4 \
		--batch-size 64

# ============================================================================
# ANALYSIS SCRIPTS
# ============================================================================

analyze_ablation:
	@echo ">>> Analyzing (d_sem, d_geo) ablation results..."
	python3.12 analyze_experiments.py --mode ablation

analyze_multiseed:
	@echo ">>> Computing confidence intervals from multi-seed runs..."
	python3.12 analyze_experiments.py --mode multiseed

analyze_all: analyze_ablation analyze_multiseed
	@echo "Analysis complete. Check assets/ for tables and plots."

# ============================================================================
# LONG CONTEXT EXPERIMENTS: 128k Context Signal
# ============================================================================
# These experiments provide empirical signal for long-context claims.
# We can't train at 128k on consumer hardware, but we can:
# 1. Measure perplexity scaling across context lengths
# 2. Test RoPE extrapolation (train short, eval long)
# 3. Run needle-in-a-haystack retrieval tests

# Context Length Scaling: How does perplexity scale with context?
# Uses FineWeb for larger data, tests 256→512→1024→2048→4096
long_context_scaling: context_256 context_512 context_1024 context_2048 context_4096
	@echo "=============================================="
	@echo "Context Length Scaling Complete!"
	@echo "Run 'make analyze_long_context' to generate scaling curve."
	@echo "=============================================="

context_256:
	@echo ">>> Training Decoupled at 256 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_256 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 256 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 3000 \
		--eval-every 500 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 32

context_512:
	@echo ">>> Training Decoupled at 512 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_512 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 512 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 3000 \
		--eval-every 500 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 16

context_1024:
	@echo ">>> Training Decoupled at 1024 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_1024 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 3000 \
		--eval-every 500 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 8

context_2048:
	@echo ">>> Training Decoupled at 2048 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_2048 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 2048 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 3000 \
		--eval-every 500 \
		--eval-iters 10 \
		--lr 3e-4 \
		--batch-size 4

context_4096:
	@echo ">>> Training Decoupled at 4096 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_4096 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 4096 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 2000 \
		--eval-every 500 \
		--eval-iters 5 \
		--lr 3e-4 \
		--batch-size 2

# Extended context training (128GB RAM allows this!)
context_8192:
	@echo ">>> Training Decoupled at 8192 context (requires 128GB RAM)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_8192 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 8192 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 1500 \
		--eval-every 500 \
		--eval-iters 3 \
		--lr 3e-4 \
		--batch-size 1

context_16384:
	@echo ">>> Training Decoupled at 16384 context (requires 128GB RAM)..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/context_16384 \
		--seed 1337 \
		--device mps \
		--attn-mode decoupled \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 16384 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--steps 1000 \
		--eval-every 250 \
		--eval-iters 2 \
		--lr 3e-4 \
		--batch-size 1

# Baseline context scaling for comparison
long_context_baseline: baseline_context_1024 baseline_context_2048 baseline_context_4096
	@echo "Baseline context scaling complete."

baseline_context_1024:
	@echo ">>> Training Baseline at 1024 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/baseline_context_1024 \
		--seed 1337 \
		--device mps \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 1024 \
		--embed-dim 512 \
		--steps 3000 \
		--eval-every 500 \
		--eval-iters 25 \
		--lr 3e-4 \
		--batch-size 8

baseline_context_2048:
	@echo ">>> Training Baseline at 2048 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/baseline_context_2048 \
		--seed 1337 \
		--device mps \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 2048 \
		--embed-dim 512 \
		--steps 3000 \
		--eval-every 500 \
		--eval-iters 10 \
		--lr 3e-4 \
		--batch-size 4

baseline_context_4096:
	@echo ">>> Training Baseline at 4096 context..."
	python3.12 v21_transformer_decoupled_bottleneck_gqa.py \
		--data fineweb_100m.tokens \
		--out-dir runs/baseline_context_4096 \
		--seed 1337 \
		--device mps \
		--attn-mode standard \
		--d-model 512 \
		--layers 6 \
		--n-head 8 \
		--d-ff 2048 \
		--block 4096 \
		--embed-dim 512 \
		--steps 2000 \
		--eval-every 500 \
		--eval-iters 5 \
		--lr 3e-4 \
		--batch-size 2

# RoPE Extrapolation Test: Train at 1024, evaluate up to 128k!
# With 128GB unified memory, we can actually test 128k context inference
rope_extrapolation:
	@echo ">>> Running RoPE extrapolation test (up to 128k)..."
	python3.12 test_rope_extrapolation.py \
		--ckpt runs/context_1024/best.pt \
		--data fineweb_100m.tokens \
		--contexts 1024 2048 4096 8192 16384 32768 65536 131072 \
		--num-batches 10 \
		--output assets/rope_extrapolation.png

# Quick extrapolation test (shorter, for iteration)
rope_extrapolation_quick:
	@echo ">>> Running quick RoPE extrapolation test..."
	python3.12 test_rope_extrapolation.py \
		--ckpt runs/context_1024/best.pt \
		--data fineweb_100m.tokens \
		--contexts 1024 4096 16384 65536 \
		--num-batches 5 \
		--output assets/rope_extrapolation_quick.png

# Needle-in-a-Haystack Test: Retrieval at various context depths
needle_haystack:
	@echo ">>> Running Needle-in-a-Haystack test..."
	python3.12 test_needle_haystack.py \
		--ckpt runs/context_1024/best.pt \
		--depths 0.1 0.25 0.5 0.75 0.9 \
		--context-lengths 512 1024 2048 4096 \
		--output assets/needle_haystack.png

# Analyze long context experiments
analyze_long_context:
	@echo ">>> Analyzing long context scaling..."
	python3.12 analyze_experiments.py --mode long_context

# ============================================================================
# 128K CONTEXT BENCHMARK (Requires 128GB RAM)
# ============================================================================
# These benchmarks test actual 128k context inference on M4 Max with 128GB RAM

benchmark_128k:
	@echo "=============================================="
	@echo "128K Context Benchmark (Decoupled)"
	@echo "Requires: M4 Max with 128GB unified memory"
	@echo "=============================================="
	python3.12 benchmark_128k.py \
		--ckpt runs/context_1024/best.pt \
		--data fineweb_100m.tokens \
		--contexts 1024 4096 16384 32768 65536 131072 \
		--output assets/benchmark_128k.png

benchmark_128k_compare:
	@echo "=============================================="
	@echo "128K Context Comparison: Decoupled vs Baseline"
	@echo "=============================================="
	python3.12 benchmark_128k.py \
		--compare runs/context_1024/best.pt runs/baseline_context_1024/best.pt \
		--data fineweb_100m.tokens \
		--contexts 1024 4096 16384 32768 65536 131072 \
		--output assets/benchmark_128k_compare.png

# Quick 128k test (just verify it works)
benchmark_128k_quick:
	@echo ">>> Quick 128k inference test..."
	python3.12 benchmark_128k.py \
		--ckpt runs/context_1024/best.pt \
		--data fineweb_100m.tokens \
		--contexts 1024 16384 131072 \
		--output assets/benchmark_128k_quick.png

# Full long-context experiment suite
long_context_full: long_context_scaling long_context_baseline context_8192 context_16384 rope_extrapolation needle_haystack benchmark_128k
	@echo "=============================================="
	@echo "Full Long Context Experiment Suite Complete!"
	@echo "=============================================="
	@echo "Generated:"
	@echo "  - Context scaling curve (256 → 16384)"
	@echo "  - RoPE extrapolation test (up to 128k)"
	@echo "  - Needle-in-haystack retrieval"
	@echo "  - 128k inference benchmark"
	@echo "Run 'make analyze_long_context' for analysis."

# ============================================================================
# NEW ARCHITECTURES (v22, v23, v24)
# ============================================================================

# v22: "Survive Scale" - Streaming decode with heterogeneous cache
# Tests memory stability and throughput with long-context generation
run_v22_survive_scale:
	@echo ">>> Running v22 (Survive Scale)..."
	python3.12 v22_decoupled_bottleneck_survive_scale.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v22_survive_scale \
		--attn-mode decoupled \
		--d-model 512 \
		--n-head 8 \
		--sem-dim 32 \
		--geo-dim 64 \
		--d-ff 2048 \
		--block 1024 \
		--embed-dim 512 \
		--tie-qk \
		--null-attn \
		--kv-cache q4_0 \
		--kv-decode-block 128 \
		--steps 1000 \
		--eval-every 200 \
		--batch-size 8

# v23: Fused Kernels (Triton) - 1-pass decode update
# Requires Triton installation. Falls back to PyTorch if missing.
run_v23_fused:
	@echo ">>> Running v23 (Fused Kernels)..."
	python3.12 v23_transformer_decoupled_bottleneck_fused_kernels.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v23_fused \
		--attn-mode decoupled \
		--d-model 512 \
		--sem-dim 32 \
		--geo-dim 64 \
		--block 2048 \
		--kv-cache q4_0 \
		--steps 500 \
		--batch-size 4

# v24: Flash-style 2-pass Decode
# Advanced split-K decoding for very long contexts (e.g. 128k)
run_v24_flash2pass:
	@echo ">>> Running v24 (Flash 2-pass Decode)..."
	python3.12 v24_transformer_decoupled_bottleneck_flash2pass.py \
		--data fineweb_100m.tokens \
		--out-dir runs/v24_flash \
		--attn-mode decoupled \
		--d-model 512 \
		--sem-dim 32 \
		--geo-dim 64 \
		--block 4096 \
		--kv-cache q4_0 \
		--steps 500 \
		--batch-size 2