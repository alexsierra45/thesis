import torch.nn as nn
from trajbert.model.attention import MultiHeadAttention
from trajbert.model.utils import gelu

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        d_ff = args.d_model*4
        self.fc1 = nn.Linear(args.d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, args.d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        self.device = 'cuda:%s' % str(args.gpu)
        self.d_model = args.d_model

    def forward(self, enc_inputs, enc_self_attn_mask, idx=-1):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs