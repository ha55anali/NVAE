python train.py --data $DATA_DIR/celeba/celeba-lmdb --root . --save . --dataset clam \
        --num_channels_enc 30 --num_channels_dec 30 --epochs 300 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 5 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-2 --num_groups_per_scale 16 \
        --batch_size 2 --num_nf 2 --ada_groups --min_groups_per_scale 4 \
        --weight_decay_norm_anneal --weight_decay_norm_init 1. --use_se --res_dist \
	--fast_adamax --num_x_bits 5 --slide_paths /data/test --patch_path /data/test/results/patches --dataloader_threads 0 \
	--num_process_per_node 1
