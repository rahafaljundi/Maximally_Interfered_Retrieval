#!/bin/bash

cd ..


#MINI IMAGENET
mem=100

runs=10
subsample=50
buffer_batch_size=10
python er_main.py --method oth_cl_neib_replay --n_tasks 5 --lr 0.01 --samples_per_task -1 --dataset miniimagenet --n_runs $runs --subsample $subsample --buffer_batch_size $buffer_batch_size  --disc_iters $n_iters --mem_size $mem  --suffix 'Nei_BF_10_SS50' --log online --update_buffer_hid 0

python er_main.py --method oth_cl_neib_replay_mid_fix --n_tasks 5 --lr 0.01 --samples_per_task -1 --dataset miniimagenet --n_runs $runs --subsample $subsample --buffer_batch_size $buffer_batch_size  --disc_iters $n_iters --mem_size $mem  --suffix 'Nei_Loc_BF_10_SS50' --log online --update_buffer_hid 0 --kl_far 1 --multiplier 1

python er_main.py --method mir_replay --n_tasks 5 --lr 0.01 --samples_per_task -1 --dataset miniimagenet --n_runs $runs --subsample $subsample --buffer_batch_size $buffer_batch_size  --disc_iters $n_iters --mem_size $mem  --suffix 'ER_MIR_BF10_SS50' --log online --update_buffer_hid 0 --kl_far 0

python er_main.py --method rand_replay --n_tasks 5 --lr 0.01 --samples_per_task -1 --dataset miniimagenet --n_runs $runs --subsample $subsample --buffer_batch_size $buffer_batch_size  --disc_iters $n_iters --mem_size $mem  --suffix 'ER_Rand_BF10' --log online --update_buffer_hid 0 --kl_far 0
exit

