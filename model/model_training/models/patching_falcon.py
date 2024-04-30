from typing import Optional, Tuple

import torch
import torch.nn as nn

from .patching_utils import compute_flash_attention


def falcon_forward_with_flash_attn(
    self,
    flash_attn: nn.Module,  # flash_attn.modules.mha.FlashSelfAttention
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    head_mask, alibi & output_attention are not supported.
    Reference to the original `FalconAttention.forward()` method which this patch replaces:
    https://github.com/huggingface/transformers/blob/c965d302791cf935d6ea7776428749be678cf509/src/transformers/models/falcon/modeling_falcon.py#L281
    """

    assert head_mask is None  # not supported.
    assert alibi is None  # not supported.
    assert not output_attentions  # not supported.

    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(
        batch_size * num_kv_heads,
        query_length,
        self.head_dim,
    )
    value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

    past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
    query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length)

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size * self.num_heads, kv_length, head_dim]
        #  - value: [batch_size * self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=1)
        value_layer = torch.cat((past_value, value_layer), dim=1)

    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
    key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
    value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

    q = query_layer_.permute(0, 2, 1, 3)
    k = key_layer_.permute(0, 2, 1, 3).expand(q.shape)
    v = value_layer_.permute(0, 2, 1, 3).expand(q.shape)

    if attention_mask is not None:
        attention_mask = attention_mask[:, 0, -1]

    flash_attn.train(self.training)
    attn_output = compute_flash_attention(flash_attn, q, k, v, attention_mask=attention_mask)
    attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

    output_tensor = self.dense(attn_output)

    return output_tensor, present
