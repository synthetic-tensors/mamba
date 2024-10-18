torchrun --nproc-per-node "$1" run_dist_test_lin.py --iterations 4 --nproc_per_node "$1" --batch_size 1 --random_seed 42 --fsdp 1 --num_layers $2 --d_model 64
