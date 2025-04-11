import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def silu(x):
    return x * torch.sigmoid(x)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

if __name__ == "__main__":
    x = torch.arange(-10, 10, 0.0001).float()
    y = silu(x)
    plt.figure(figsize=(10, 5))
    plt.plot(x.numpy(), y.numpy(), label='SiLU (Swish)', color='blue')
    plt.plot(x.numpy(), torch.relu(x), label='ReLU', color='red')
    plt.plot(x.numpy(), sigmoid(x), label='sigmoid', color='green')
    plt.title("SiLU Activation Function")
    plt.legend()
    plt.tight_layout()
    plt.savefig('silu.png')