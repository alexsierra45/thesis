import torch
import torch.nn as nn

from trajbert.model.embedding import Embedding
from trajbert.model.encoder import EncoderLayer
from trajbert.model.utils import gelu, get_attn_pad_mask

class TrajBERT(nn.Module):
    def __init__(self, args, vocab_size):
        super(TrajBERT, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embedding = Embedding(args, vocab_size)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.layer)])
       
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.activ = gelu
        # fc is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc = nn.Linear(args.d_model, vocab_size, bias=False)
        self.fc.weight = embed_weight

        if args.use_his:
            self.linear_prior = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Dropout(0.5))
            self.linear_next = nn.Sequential(nn.Linear(args.d_model, args.d_model), nn.Dropout(0.5))
            self.bnorm = nn.BatchNorm1d(args.d_model)

    def forward(self, input_ids, masked_pos, user_ids=None, temp_ids=None, input_prior=None, input_next=None,
                input_next_dis=None, input_prior_dis=None):
        
        output = self.embedding(input_ids, user_ids, temp_ids)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]

        if self.args.use_his:
            input_prior_embedded = self.embedding.tok_embed(input_prior)
            input_next_embedded = self.embedding.tok_embed(input_next)

        for idx, layer in enumerate(self.layers):
            output = layer(output, enc_self_attn_mask, idx)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.args.d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]

        if self.args.use_his:
            linear_prior = (self.linear_prior(input_prior_embedded))
            linear_next = (self.linear_next(input_next_embedded))
            
            h_masked = self.linear(h_masked) + linear_prior + linear_next  # [batch_size, max_pred, d_model]
            h_masked = self.bnorm(h_masked.permute(0, 2, 1)).permute(0, 2, 1)

        h_masked = self.activ(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc(h_masked)
       
        return logits_lm