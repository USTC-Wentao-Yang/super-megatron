import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import torch
from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms
from loguru import logger

class Partition(object):
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data 
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def partition_dataset():
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    world_size = int(dist.get_world_size())
    total_batch_size = 128
    pbs = int(total_batch_size / world_size)
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())

    train_set = torch.utils.data.DataLoader(partition, batch_size=pbs, shuffle=True)
    return train_set, pbs

def average_gradient(model):
    world_size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= world_size

def ddp_setup():
    dist.init_process_group(backend="nccl")

def run(loop=100, seed=1):
    ddp_setup()
    rank = int(os.environ['LOCAL_RANK'])
    torch.manual_seed(seed)
    train_set, pbs = partition_dataset()
    model = Net()
    model.load_state_dict(torch.load("model_weight.pth"))
    torch.cuda.set_device(rank)
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    num_batchs = ceil(len(train_set.dataset) / float(pbs))
    for epoch in range(loop):
        epoch_loss = 0.0
        for data, target in train_set:
            data, target = Variable(data), Variable(target)
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradient(model)
            optimizer.step()
        logger.info(f'Rank {rank} echo {epoch} loss: { epoch_loss / num_batchs }')
    with open(f'{rank}.txt', 'w') as f:
        for name, parameters in model.named_parameters():
            f.write("tensor :" + name + str(parameters.data.cpu().numpy()))
    if rank == 0 :
        model = model.cpu()
        logger.info("Save parameters to model_weight.pth.")
        torch.save(model.state_dict(), "model_weight.pth")

def Train_parse():
    parser = argparse.ArgumentParser(description="Train parser.")
    parser.add_argument('--iterations', type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    world_size = 4
    processes = []
    args = Train_parse()
    logger.info("master addr:" + str(os.environ["MASTER_ADDR"]))
    run(args.iterations)
    dist.destroy_process_group()

