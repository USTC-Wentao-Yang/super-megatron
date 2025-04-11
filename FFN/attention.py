import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
    ):
        super(attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        self.QKV = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x, mask=None):
        Q, K, V = self.QKV(x).split(self.hidden_dim, dim=-1)
        k = K.transpose(1, 2)
        attention_score = torch.matmul(Q, k) / math.sqrt(self.hidden_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
            attention_score = F.softmax(attention_score, dim=-1)
        return torch.matmul(attention_score, V)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_mask(seq_len):
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = torch.triu((mask, )
    print(f'mask is : {mask}')

if __name__ == '__main__':
    set_random_seed(42)
    x = torch.randn(1, 10, 1256).cuda()
    generate_mask(10)
    model = attention(1256).cuda()
    print(model(x))