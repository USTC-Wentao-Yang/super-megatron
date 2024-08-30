# super-megatron
Tiny megatron project.

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 all_reduce_train_torchrun.py --iterations 100
'''

