import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        ans = self.pe[:, :x.size(1)]
        return ans

class Embedding(nn.Module):
    def __init__(self, args,vocab_size):
        max_len = args.seq_len
        super(Embedding, self).__init__()
        self.args = args
        self.tok_embed = nn.Embedding(vocab_size, args.d_model)
        self.pos_embed = PositionalEncoding(max_len, args.d_model)
        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x, user=None, temporal=None):
        device = 'cuda:%s' % str(self.args.gpu)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(device)  # [seq_len] -> [batch_size, seq_len]
        if self.args.if_posiemb:
            embedding = self.tok_embed(x) + self.pos_embed(pos)
        else:
            embedding = self.tok_embed(x)

        return self.norm(embedding)