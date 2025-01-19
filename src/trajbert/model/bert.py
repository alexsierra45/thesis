import torch
import torch.nn as nn

from .embedding import Embedding
from .encoder import EncoderLayer
from .utils import gelu, get_attn_pad_mask

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
            input_next_dis=None, input_prior_dis=None, input_mask=None):
    
        # Embedding inicial
        output = self.embedding(input_ids, user_ids, temp_ids)  # [batch_size, seq_len, d_model]

        # Generar máscara de atención (atención completa para input_ids)
        enc_self_attn_mask = input_ids.ne(0).unsqueeze(1).expand(-1, input_ids.size(1), -1).bool()  # [batch_size, seq_len, seq_len]

        # Si se utiliza el historial (use_his), procesar input_prior y input_next
        if self.args.use_his:
            input_prior_embedded = self.embedding.tok_embed(input_prior)
            input_next_embedded = self.embedding.tok_embed(input_next)

        # Pasar por las capas del Transformer
        for idx, layer in enumerate(self.layers):
            output = layer(output, enc_self_attn_mask, idx)

        # Seleccionar posiciones enmascaradas según masked_pos
        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.args.d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]

        # Aplicar input_mask para filtrar posiciones válidas en masked_pos
        if input_mask is not None:
            # Expandir input_mask para que coincida con las dimensiones de h_masked
            valid_mask = input_mask.unsqueeze(-1).expand(-1, -1, self.args.d_model)  # [batch_size, max_pred, d_model]
            h_masked = h_masked * valid_mask  # Solo mantener posiciones válidas

        # Si se utiliza el historial, agregar términos de input_prior y input_next
        if self.args.use_his:
            linear_prior = self.linear_prior(input_prior_embedded)
            linear_next = self.linear_next(input_next_embedded)
            
            h_masked = self.linear(h_masked) + linear_prior + linear_next  # [batch_size, max_pred, d_model]
            h_masked = self.bnorm(h_masked.permute(0, 2, 1)).permute(0, 2, 1)

        # Aplicar capa lineal y activación
        h_masked = self.activ(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc(h_masked)  # [batch_size, max_pred, vocab_size]

        return logits_lm