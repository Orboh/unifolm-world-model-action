import torch
import torch.nn.functional as F
import xformers.ops as xops

# prebuilt xformers 0.0.33.post1 is built for torch 2.9 and cannot load its CUDA ext
# on Thor torch 2.11 -> memory_efficient_attention is unavailable. unifolm_wma passes
# (B*heads, N, dim_head) 3D tensors with an optional additive (B*heads, Lq, Lk) bias and
# no custom scale. Reroute to torch SDPA (default scale 1/sqrt(K) == xformers default).
def _mea_sdpa(query, key, value, attn_bias=None, p=0.0, scale=None, op=None):
    q = query.unsqueeze(1)
    k = key.unsqueeze(1)
    v = value.unsqueeze(1)
    mask = None
    if attn_bias is not None:
        m = attn_bias if torch.is_tensor(attn_bias) else attn_bias.materialize(
            (query.shape[0], query.shape[1], key.shape[1]), dtype=query.dtype, device=query.device)
        mask = m.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p, scale=scale)
    return out.squeeze(1)

xops.memory_efficient_attention = _mea_sdpa
