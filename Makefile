.PHONY: train v20 bottleneck_attention

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
