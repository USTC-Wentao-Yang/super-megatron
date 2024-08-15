import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank_id, world_size):
    torch.cuda.set_device(rank_id)
    tensor = torch.tensor(rank_id % 8).cuda()
    gather_list = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    print("Rank {} gathered: {}".format(rank_id, gather_list))

def init_process(rank_id, world_size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=world_size)
    fn(rank_id, world_size)

if __name__ == "__main__":
    world_size = 4
    mp.set_start_method('spawn')
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()