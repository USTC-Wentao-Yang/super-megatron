import os
import torch.distributed as dist
import torch
import torch.multiprocessing as mp

def run(rank_id, world_size):
    torch.cuda.set_device(rank_id)
    tensor = torch.tensor(rank_id % 8).cuda()
    print(f'Before reduce Rank {rank_id} has data:{tensor}')
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f'After reduce Rank {rank_id} has data:{tensor}')

def init_process(rank_id, world_size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank = rank_id, world_size = world_size)
    fn(rank_id, world_size)

if __name__ == "__main__":
    world_size = 4
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size,run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()