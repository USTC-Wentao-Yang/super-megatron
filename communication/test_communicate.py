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
    device = "cuda:" + str(rank_id)
    if rank_id == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        dist.send(tensor, 1)
        print('After send, Rank ', rank_id,' has data :', tensor)
        dist.recv(tensor, 1)
        print('After recv, Rank ', rank_id, ' has data :', tensor)
    else:
        tensor = torch.tensor([4.0, 5.0, 6.0]).to(device)
        dist.recv(tensor, 0)
        print(f'After recv, Rank ', rank_id, ' has data :', tensor)
        dist.send(tensor,0)
        print(f'After send, Rank ', rank_id, ' has data :', tensor)

if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

