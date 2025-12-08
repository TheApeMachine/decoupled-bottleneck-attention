.PHONY: train

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

	python3 v10_transformer_lowrank_scaled.py \
		--data-file wiki.train.tokens \
		--log-file v10_log.jsonl \
		--epochs 30 \
		--init-rank 64 \
		--d-model 512 \
		--n-layers 6 \
		--n-heads 8 \
		--d-ff 2048 \
		--block-size 256 \
		--batch-size 32 \
		--steps-per-epoch 200
