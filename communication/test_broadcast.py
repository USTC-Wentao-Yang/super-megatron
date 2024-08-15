import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch
def init_process(rank_id, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank = rank_id, world_size = size)
    print(f'Rank {rank_id} is initialized')
    fn(rank_id, size)

def run(rank_id, size):
    device = 'cuda:' + str(rank_id)
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    tensor = tensor.to(device)
    if(rank_id == 2):
        print('Before broadcast , Rank ', rank_id, ' has data: ', tensor)
    dist.broadcast(tensor, src = 0)
    if(rank_id == 2):
        print('After broadcast , Rank ', rank_id, ' has data:', tensor)

if __name__ == "__main__":
    world_size = 4
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

