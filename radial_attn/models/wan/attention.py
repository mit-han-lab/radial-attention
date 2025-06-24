from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from ...attn_mask import RadialAttention
from torch.nn.attention import sdpa_kernel, SDPBackend

class WanSparseAttnProcessor2_0:
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"  # default to radial attention, can be changed to "dense" for dense attention
    
    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.layer_idx = layer_idx
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)
        
        if timestep is None: # this is the case for dense attention or cross attention
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        else: # this is the case for sparse attention
            if numeral_timestep < self.dense_timestep or self.layer_idx < self.dense_block or self.sparse_type == "dense":
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
            else:
                batch_size = query.shape[0]
                # transform (batch_size, num_heads, seq_len, head_dim) to (seq_len * batch_size, num_heads, head_dim)
                query = rearrange(query, "b h s d -> (b s) h d")
                key = rearrange(key, "b h s d -> (b s) h d")
                value = rearrange(value, "b h s d -> (b s) h d")
                # apply radial attention
                hidden_states = RadialAttention(
                    query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type="radial", block_size=128, decay_factor=self.decay_factor, model_type="wan",
                )
                # transform back to (batch_size, num_heads, seq_len, head_dim)
                hidden_states = rearrange(hidden_states, "(b s) h d -> b h s d", b=batch_size)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
