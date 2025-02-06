import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self,args):
        super(ScaledDotProductAttention, self).__init__()
        self.args = args

    def forward(self, Q, K, V, attn_mask, a=None, r_q2=None, idx=-1):
        d_k = self.args.d_model // self.args.head
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        if a != None:
            len_q = Q.size(2)
            batch_size = Q.size(0)
            n_heads = Q.size(1)
            a_scores = torch.matmul(r_q2, a.transpose(1, 2)).transpose(0, 1)
            a_scores = a_scores.contiguous().view(batch_size, n_heads, len_q, len_q) / np.sqrt(d_k)
            scores += a_scores

        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.d_k = self.d_v = args.d_model // args.head
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.head, bias=False)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.head, bias=False)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.head, bias=False)
        self.fc = nn.Linear(args.head * self.d_v, args.d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        device = 'cuda:%s' % str(self.args.gpu)
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.args.head, self.d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.args.head, self.d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.args.head, self.d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.head, 1, 1)

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, seq_len, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.args.head * self.d_v)

        output = self.fc(context)
        return nn.LayerNorm(self.args.d_model).to(device)(output + residual)  # output: [batch_size, seq_len, d_model]