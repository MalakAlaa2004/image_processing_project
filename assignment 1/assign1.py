import torch
import torch.nn as nn
import math

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]


# -------------------------
# Utilities for multi-head reshape
# -------------------------
def split_heads(x, nhead):
    # x: (seq_len, batch, d_model)
    seq_len, batch, d_model = x.shape
    dk = d_model // nhead
    # convert to (batch, nhead, seq_len, dk)
    x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
    x = x.contiguous().view(batch, seq_len, nhead, dk)  # (batch, seq_len, nhead, dk)
    return x.permute(0, 2, 1, 3)  # (batch, nhead, seq_len, dk)

def combine_heads(x):
    # x: (batch, nhead, seq_len, dk) -> returns (seq_len, batch, d_model)
    batch, nhead, seq_len, dk = x.shape
    d_model = nhead * dk
    x = x.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, nhead, dk)
    x = x.view(batch, seq_len, d_model)  # (batch, seq_len, d_model)
    return x.permute(1, 0, 2)  # (seq_len, batch, d_model)


# -------------------------
# Debuggable Encoder Layer
# -------------------------
class DebugTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead

        # projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # norms/dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: (seq_len, batch, d_model)
        snapshots = {}

        # Encoder block input tensor.
        snapshots["encoder_block_input"] = src.detach().clone()

        # Q,K,V (before split): shapes (seq_len, batch, d_model)
        Q = self.W_q(src)
        K = self.W_k(src)
        V = self.W_v(src)
        snapshots["encoder_Q"] = Q.detach().clone()
        snapshots["encoder_K"] = K.detach().clone()
        snapshots["encoder_V"] = V.detach().clone()

        # Multi-head split
        Qh = split_heads(Q, self.nhead)  # (batch, nhead, seq_len, dk)
        Kh = split_heads(K, self.nhead)
        Vh = split_heads(V, self.nhead)
        snapshots["encoder_Q_split"] = Qh.detach().clone()
        snapshots["encoder_K_split"] = Kh.detach().clone()
        snapshots["encoder_V_split"] = Vh.detach().clone()

        # Attention scores before softmax: (batch, nhead, seq_len, seq_len)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.dk)
        snapshots["encoder_attn_scores_pre_softmax"] = scores.detach().clone()

        # Softmax (post-softmax scores)
        attn_weights = torch.softmax(scores, dim=-1)
        snapshots["encoder_attn_scores_post_softmax"] = attn_weights.detach().clone()

        # Multi-head attention output (per-head)
        attn_output_heads = torch.matmul(attn_weights, Vh)  # (batch, nhead, seq_len, dk)
        snapshots["encoder_attn_output_heads"] = attn_output_heads.detach().clone()

        # Concatenate heads
        attn_output = combine_heads(attn_output_heads)  # (seq_len, batch, d_model)
        attn_output = self.out_proj(attn_output)
        snapshots["encoder_attn_output_concat"] = attn_output.detach().clone()

        # Residual connection tensors (before norm)
        src_res1 = src + self.dropout1(attn_output)
        src_after_norm1 = self.norm1(src_res1)
        snapshots["encoder_residual_after_attn"] = src_res1.detach().clone()
        snapshots["encoder_norm1_output"] = src_after_norm1.detach().clone()

        # Feed-forward input
        ffn_input = src_after_norm1
        snapshots["encoder_ffn_input"] = ffn_input.detach().clone()

        ff1 = torch.relu(self.linear1(ffn_input))
        snapshots["encoder_ffn_linear1"] = ff1.detach().clone()

        ff2 = self.linear2(ff1)
        snapshots["encoder_ffn_linear2"] = ff2.detach().clone()

        # Residual + norm 2
        src_res2 = src_after_norm1 + self.dropout2(ff2)
        final_output = self.norm2(src_res2)
        snapshots["encoder_residual_after_ffn"] = src_res2.detach().clone()
        snapshots["encoder_output"] = final_output.detach().clone()

        return final_output, snapshots


# -------------------------
# Debuggable Decoder Layer
# -------------------------
class DebugTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead

        # masked self-attn projections
        self.W_q_self = nn.Linear(d_model, d_model)
        self.W_k_self = nn.Linear(d_model, d_model)
        self.W_v_self = nn.Linear(d_model, d_model)
        self.out_proj_self = nn.Linear(d_model, d_model)

        # cross-attn projections (Q from decoder, K/V from encoder)
        self.W_q_cross = nn.Linear(d_model, d_model)
        self.W_k_cross = nn.Linear(d_model, d_model)
        self.W_v_cross = nn.Linear(d_model, d_model)
        self.out_proj_cross = nn.Linear(d_model, d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # norms/dropouts
        self.norm1 = nn.LayerNorm(d_model)  # after masked self-attn
        self.norm2 = nn.LayerNorm(d_model)  # after cross-attn
        self.norm3 = nn.LayerNorm(d_model)  # after FFN

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: (tgt_seq_len, batch, d_model)
        # memory: (src_seq_len, batch, d_model)
        snapshots = {}

        snapshots["decoder_block_input"] = tgt.detach().clone()

        # ---- Masked Self-Attention ----
        Qs = self.W_q_self(tgt)
        Ks = self.W_k_self(tgt)
        Vs = self.W_v_self(tgt)
        snapshots["decoder_masked_Q"] = Qs.detach().clone()
        snapshots["decoder_masked_K"] = Ks.detach().clone()
        snapshots["decoder_masked_V"] = Vs.detach().clone()

        Qh_s = split_heads(Qs, self.nhead)  # (batch, nhead, tgt_len, dk)
        Kh_s = split_heads(Ks, self.nhead)
        Vh_s = split_heads(Vs, self.nhead)
        snapshots["decoder_masked_Q_split"] = Qh_s.detach().clone()
        snapshots["decoder_masked_K_split"] = Kh_s.detach().clone()
        snapshots["decoder_masked_V_split"] = Vh_s.detach().clone()

        # scores before mask
        scores_self_pre_mask = torch.matmul(Qh_s, Kh_s.transpose(-2, -1)) / math.sqrt(self.dk)
        snapshots["decoder_masked_attn_scores_before_mask"] = scores_self_pre_mask.detach().clone()

        # Mask tensor (expand to match scores shape)
        # tgt_mask expected to be shape (tgt_len, tgt_len) with float('-inf') or bool. If None, construct causal mask.
        if tgt_mask is None:
            tgt_len = tgt.size(0)
            causal = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
            # create a mask value tensor shaped for broadcast: (1,1,tgt_len,tgt_len)
            mask_tensor = causal
        else:
            # transform to boolean causal where True means masked
            mask_tensor = tgt_mask.bool()

        # Save mask (as boolean)
        snapshots["decoder_mask_tensor"] = mask_tensor.detach().clone() if isinstance(mask_tensor, torch.Tensor) else mask_tensor

        # Apply mask: we need to match shape (batch, nhead, tgt_len, tgt_len)
        # Convert mask to float additive mask
        if isinstance(mask_tensor, torch.Tensor):
            additive = torch.zeros(1, 1, mask_tensor.size(0), mask_tensor.size(1)).to(scores_self_pre_mask.device)
            additive = additive.masked_fill(mask_tensor.unsqueeze(0).unsqueeze(0), float("-1e9"))
        else:
            additive = torch.zeros_like(scores_self_pre_mask)

        scores_self_masked = scores_self_pre_mask + additive
        # masked + softmax
        attn_weights_self = torch.softmax(scores_self_masked, dim=-1)
        snapshots["decoder_masked_attn_scores_after_mask_and_softmax"] = attn_weights_self.detach().clone()

        attn_output_heads_self = torch.matmul(attn_weights_self, Vh_s)
        snapshots["decoder_masked_attn_output_heads"] = attn_output_heads_self.detach().clone()

        # concat
        attn_output_self = combine_heads(attn_output_heads_self)  # (tgt_len, batch, d_model)
        attn_output_self = self.out_proj_self(attn_output_self)
        snapshots["decoder_masked_attn_output_concat"] = attn_output_self.detach().clone()

        # Residual + norm
        tgt_res1 = tgt + self.dropout1(attn_output_self)
        tgt_after_norm1 = self.norm1(tgt_res1)
        snapshots["decoder_residual_after_masked_self_attn"] = tgt_res1.detach().clone()
        snapshots["decoder_norm_after_masked_self_attn"] = tgt_after_norm1.detach().clone()

        # ---- Cross-Attention ----
        # Q from decoder (after masked self-attn), K/V from encoder memory
        Qc = self.W_q_cross(tgt_after_norm1)
        Kc = self.W_k_cross(memory)
        Vc = self.W_v_cross(memory)
        snapshots["decoder_cross_Q"] = Qc.detach().clone()
        snapshots["decoder_cross_K"] = Kc.detach().clone()
        snapshots["decoder_cross_V"] = Vc.detach().clone()

        Qh_c = split_heads(Qc, self.nhead)  # (batch, nhead, tgt_len, dk)
        Kh_c = split_heads(Kc, self.nhead)  # (batch, nhead, src_len, dk)
        Vh_c = split_heads(Vc, self.nhead)
        snapshots["decoder_cross_Q_split"] = Qh_c.detach().clone()
        snapshots["decoder_cross_K_split"] = Kh_c.detach().clone()
        snapshots["decoder_cross_V_split"] = Vh_c.detach().clone()

        # Cross-attn scores pre-softmax: (batch, nhead, tgt_len, src_len)
        scores_cross_pre = torch.matmul(Qh_c, Kh_c.transpose(-2, -1)) / math.sqrt(self.dk)
        snapshots["decoder_cross_attn_scores_pre_softmax"] = scores_cross_pre.detach().clone()

        attn_weights_cross = torch.softmax(scores_cross_pre, dim=-1)
        snapshots["decoder_cross_attn_scores_post_softmax"] = attn_weights_cross.detach().clone()

        attn_output_heads_cross = torch.matmul(attn_weights_cross, Vh_c)  # (batch, nhead, tgt_len, dk)
        snapshots["decoder_cross_attn_output_heads"] = attn_output_heads_cross.detach().clone()

        attn_output_cross = combine_heads(attn_output_heads_cross)  # (tgt_len, batch, d_model)
        attn_output_cross = self.out_proj_cross(attn_output_cross)
        snapshots["decoder_cross_attn_output_concat"] = attn_output_cross.detach().clone()

        # Residual + norm after cross-attn
        tgt_res2 = tgt_after_norm1 + self.dropout2(attn_output_cross)
        tgt_after_norm2 = self.norm2(tgt_res2)
        snapshots["decoder_residual_after_cross_attn"] = tgt_res2.detach().clone()
        snapshots["decoder_norm_after_cross_attn"] = tgt_after_norm2.detach().clone()

        # ---- Decoder FFN ----
        ff_in = tgt_after_norm2
        snapshots["decoder_ffn_input"] = ff_in.detach().clone()

        ff1 = torch.relu(self.linear1(ff_in))
        snapshots["decoder_ffn_linear1"] = ff1.detach().clone()

        ff2 = self.linear2(ff1)
        snapshots["decoder_ffn_linear2"] = ff2.detach().clone()

        tgt_res3 = tgt_after_norm2 + self.dropout3(ff2)
        final_output = self.norm3(tgt_res3)
        snapshots["decoder_residual_after_ffn"] = tgt_res3.detach().clone()
        snapshots["decoder_output"] = final_output.detach().clone()

        return final_output, snapshots


# -------------------------
# Full model + run
# -------------------------
def main():
    # Model params
    d_model = 128
    nhead = 4
    dim_ff = 256
    vocab_size = 1000
    num_enc_layers = 2
    num_dec_layers = 2

    # Build modules
    embedding = nn.Embedding(vocab_size, d_model)
    pos_encoder = PositionalEncoding(d_model)
    enc_layers = nn.ModuleList([DebugTransformerEncoderLayer(d_model, nhead, dim_ff) for _ in range(num_enc_layers)])
    dec_layers = nn.ModuleList([DebugTransformerDecoderLayer(d_model, nhead, dim_ff) for _ in range(num_dec_layers)])
    output_projection = nn.Linear(d_model, vocab_size)

    # Dummy tokens
    src_tokens = torch.tensor([[11, 24, 56, 78, 91, 45, 62]], dtype=torch.long)   # (1, seq_len)
    tgt_tokens = torch.tensor([[101, 202, 303, 404, 505]], dtype=torch.long)      # (1, tgt_len)

    snapshots = {}
    snapshots["raw_src_tokens"] = src_tokens.detach().clone()
    snapshots["raw_tgt_tokens"] = tgt_tokens.detach().clone()

    # Convert to (seq_len, batch)
    src = src_tokens.transpose(0, 1)
    tgt = tgt_tokens.transpose(0, 1)

    # Embeddings
    src_lookup = embedding(src)
    tgt_lookup = embedding(tgt)
    snapshots["src_input_embeddings_after_lookup"] = src_lookup.detach().clone()
    snapshots["tgt_input_embeddings_after_lookup"] = tgt_lookup.detach().clone()

    src_emb = pos_encoder(src_lookup * math.sqrt(d_model))
    tgt_emb = pos_encoder(tgt_lookup * math.sqrt(d_model))
    snapshots["src_embeddings_after_positional_encoding"] = src_emb.detach().clone()
    snapshots["tgt_embeddings_after_positional_encoding"] = tgt_emb.detach().clone()

    # Pass through encoder stack
    memory = src_emb
    for i, layer in enumerate(enc_layers):
        memory, enc_snap = layer(memory)
        # Save with layer index in key
        snapshots.update({f"{k}_layer{i+1}": v for k, v in enc_snap.items()})

    # Target mask
    tgt_seq_len = tgt.size(0)
    subsequent_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()
    snapshots["mask_tensor"] = subsequent_mask.detach().clone()

    # Pass through decoder stack
    out = tgt_emb
    for j, layer in enumerate(dec_layers):
        out, dec_snap = layer(out, memory, tgt_mask=subsequent_mask)
        snapshots.update({f"{k}_layer{j+1}": v for k, v in dec_snap.items()})

    # Final projection
    snapshots["decoder_final_sequence_output_before_projection"] = out.detach().clone()
    logits = output_projection(out)
    snapshots["logits_after_final_projection"] = logits.detach().clone()
    snapshots["logits_slice_first_token"] = logits[0, 0, :10].detach().clone()
    return snapshots


if __name__ == "__main__":
    all_snapshots = main()
