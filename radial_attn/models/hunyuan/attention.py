import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from ...attn_mask import RadialAttention
from typing import Optional
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import sdpa_kernel, SDPBackend

class HunyuanVideoAttnSparseProcessor2_0:
    mask_map = None
    dense_timestep = 0
    dense_block = 0
    decay_factor = 1.0
    sparse_type = "radial"  # default to radial attention, can be changed to
    
    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
            
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        if timestep is None: # this is the case for dense attention
            with sdpa_kernel(bzsackends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        else:  # this is the case for sparse attention
            # print(f"numeral_timestep: {numeral_timestep}, dense_timestep: {self.dense_timestep}, layer_idx: {self.layer_idx}, dense_block: {self.dense_block}, sparse_type: {self.sparse_type}")
            if numeral_timestep < self.dense_timestep or self.layer_idx < self.dense_block or self.sparse_type == "dense":
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
            else:
                batch_size = query.shape[0]
                query = rearrange(query, "b h s d" " -> (b s) h d")
                key = rearrange(key, "b h s d" " -> (b s) h d")
                value = rearrange(value, "b h s d" " -> (b s) h d")
                # apply radial attention
                hidden_states = RadialAttention(
                    query=query, key=key, value=value, mask_map=self.mask_map, sparsity_type=self.sparse_type, block_size=128, decay_factor=self.decay_factor, model_type="hunyuan",
                )
                hidden_states = rearrange(hidden_states, "(b s) h d -> b h s d", b=batch_size)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
