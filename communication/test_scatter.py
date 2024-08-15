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

# def run(rank_id, size):
#     device = "cuda:" + str(rank_id)
#     tensor = torch.tensor(rank_id)
#     tensor = tensor.to(device)
#     print('Before scatter ,Rank ', rank_id, ' has data:', tensor)
#     if rank_id == 0:
#         scatter_tensor = torch.arange(4, dtype=torch.int64).to(device)
#         scatter_list = [torch.tensor(8).to(device), torch.tensor(8).to(device), torch.tensor(8).to(device), torch.tensor(8).to(device)]
#         print("scatter_list", scatter_list)
#         dist.scatter(tensor, src = 0, scatter_list=scatter_list)
#     else:
#         dist.scatter(tensor, src=0)
   # print('After scatter Rank {} has data : {}'.format(rank_id, tensor))

def run(rank_id, size):
    device = "cuda:" + str(rank_id)
    torch.cuda.set_device(rank_id)
    tensor = torch.arange(2) + 1 + 2 * rank_id
    tensor = tensor.to(device)
    print('before scatter',' Rank ', rank_id, ' has data ', tensor)
    if rank_id == 0:
        scatter_list = [torch.tensor([0,0]).to(device), torch.tensor([1,1]).to(device), torch.tensor([2,2]).to(device), torch.tensor([3,3]).to(device)]
        print('scater list:', scatter_list)
        dist.scatter(tensor, src = 0, scatter_list=scatter_list)
    else:
        dist.scatter(tensor, src = 0)
        print('after scatter',' Rank ', rank_id, ' has data ', tensor)       

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

