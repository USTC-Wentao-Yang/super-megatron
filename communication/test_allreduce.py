import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank_id, world_size):
    torch.cuda.set_device(rank_id % 8)
    tensor = torch.tensor(rank_id, dtype=torch.int64).cuda()
    print(f'Before reduce rank {rank_id} has data: {tensor}')
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    print(f'After reduce rank {rank_id} has data: {tensor}')

def init_processes(rank_id, world_size, fn, backend="nccl"):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=world_size)
    fn(rank_id, world_size)

if __name__ == "__main__":
    world_size = 4
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=init_processes, args=(rank, world_size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()