import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank_id, world_size):
    torch.cuda.set_device(rank_id % 8)
    tensor = torch.tensor(rank_id, dtype=torch.int64).cuda()
    if(rank_id == 0):
        gather_list = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
        print('Before Gather Rank {}: {}'.format(rank_id, gather_list))
        dist.gather(tensor, gather_list=gather_list, dst=0)
        print('After Gather Rank {}: {}'.format(rank_id, gather_list))
    else:
        dist.gather(tensor, dst=0)
        print('After Gather Rank {}: {}'.format(rank_id, None))

def init_process(rank_id, world_size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=world_size)
    fn(rank_id, world_size)

if __name__ == '__main__':
    word_size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(4):
        p = mp.Process(target=init_process, args=(rank, word_size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

