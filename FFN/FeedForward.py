import torch
import torch.nn as nn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: int,
    ):
        super(FeedForward, self).__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))

def setup_distributed():
    import os
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
if __name__ == "__main__":
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    setup_distributed()
    initialize_model_parallel(model_parallel_size_=1) 
    model = FeedForward(512, 768, 128, 4).cuda()
    x = torch.randn(1, 10, 512).cuda()
    print(model(x).shape)
    for name, weight in model.named_parameters():
        print(f'{name}: {weight.shape}')