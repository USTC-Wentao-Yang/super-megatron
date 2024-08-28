import os
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

def run(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["RANK"] = str(rank)
    os.environ['NCCL_DEBUG']="INFO"
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    model = nn.Linear(10, 1).cuda()

    ddp_model = DDP(model, device_ids=[rank])
    

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for i in range(2):
        outputs = ddp_model(torch.randn(20, 10).cuda())
        lables = torch.randn(20, 1).cuda()

        for param in model.parameters():
            print(param)
        loss = loss_fn(outputs, lables)
        loss.backward()
        optimizer.step()
        logger.info(f"rank: {rank}, loss: {loss}")
    torch.save(model, 'model.pth')

        

def main():
    world_size = 4
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()