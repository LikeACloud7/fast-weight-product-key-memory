from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Union
import logging

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update, rope_config_validation
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None

# Uncomment to disable kernels
# causal_conv1d_update, causal_conv1d_fn = None, None
# chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
# FusedRMSNormGated = None

from src.models.pkm.memory import HashingMemory
from src.models.fwpkm.fwpkm import FastWeightProductKeyMemory
from src.models.fwpkm.fwmlp import FastWeightMLP


logger = logging.getLogger(__name__)


def fn_by_compile_flag(fn, module, no_compile):
    if no_compile:
        return partial(fn.__wrapped__, module)
    else:
        return fn


@dataclass
class Qwen3NextMemModelOutput(BaseModelOutputWithPast):
    all_pkm_idcs: Optional[torch.LongTensor] = None
    all_fwpkm_losses: Optional[torch.FloatTensor] = None
    all_fwpkm_idcs: Optional[torch.LongTensor] = None
    all_fwpkm_scores: Optional[torch.FloatTensor] = None
    all_fwpkm_addr_stats: Optional[list[dict]] = None
    all_fwpkm_grad_norms: Optional[list] = None
    all_fwpkm_gates: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3NextMemLMOutput(CausalLMOutputWithPast, Qwen3NextMemModelOutput):
    pass


class Qwen3NextMemConfig(PretrainedConfig):
    model_type = "qwen3_next_mem"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        layer_types=None,
        sliding_window=None,
        # PKM
        pkm_layers=None,
        pkm_k_dim=512,
        pkm_v_dim=512,
        pkm_heads=4,
        pkm_topk=32,
        pkm_n_subkeys=512,
        pkm_query_rmsnorm=True,
        # FwPKM
        # - general
        fwpkm_layers=None,
        fwpkm_k_dim=512,
        fwpkm_v_dim=512,
        fwpkm_heads=1,
        fwpkm_topk=8,
        fwpkm_n_subkeys=512,
        fwpkm_variant="pkm",  # "pkm" | "mlp"
        fwpkm_fp32_fw=True,
        fwpkm_before_attn=True,
        # - optimization
        fwpkm_update_chunk_size=512,
        fwpkm_loss_type="mse",  # "mse" only atm
        fwpkm_optimizer_type="sgd",  # "sgd" only atm
        fwpkm_optimizer_lr=1.0,
        fwpkm_optimizer_weight_decay=0.0,
        fwpkm_grad_clip=False,
        fwpkm_addr_loss="me",  # None | "me"
        fwpkm_addr_loss_weight=1.0,
        fwpkm_weight_loss_with_gates=True,
        fwpkm_mem_grad_to_values_only=True,
        # - input
        fwpkm_query_src="hidden",  # "key" | "query" | "value" | "raw_output" | "output" | "hidden"
        fwpkm_value_src="hidden",  # "key" | "query" | "value" | "raw_output" | "output" | "hidden"
        fwpkm_compress_query=None,  # None | "l2norm" | "zero_mean"
        fwpkm_compress_value="zero_mean",  # None | "l2norm" | "zero_mean"
        fwpkm_target_value_lookahead=1,
        # - score
        fwpkm_score_nonlinear="softmax",  # "softmax" | "silu" | "relu"
        fwpkm_qk_score_type="idw",  # "dot_product", "idw"
        fwpkm_score_temperature=1.0,
        # - output
        fwpkm_out_fuse_gate=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if bool((i + 1) % interval_pattern):
                    self.layer_types.append("linear_attention")
                elif self.sliding_window is not None:
                    self.layer_types.append("sliding_attention")
                else:
                    self.layer_types.append("full_attention")

        # linear attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # pkm part
        self.pkm_layers = pkm_layers if pkm_layers is not None else []
        self.pkm_k_dim = pkm_k_dim
        self.pkm_v_dim = pkm_v_dim
        self.pkm_heads = pkm_heads
        self.pkm_topk = pkm_topk
        self.pkm_n_subkeys = pkm_n_subkeys
        self.pkm_query_rmsnorm = pkm_query_rmsnorm

        # fwpkm part
        self.fwpkm_layers = fwpkm_layers if fwpkm_layers is not None else []
        self.fwpkm_k_dim = fwpkm_k_dim
        self.fwpkm_v_dim = fwpkm_v_dim
        self.fwpkm_heads = fwpkm_heads
        self.fwpkm_topk = fwpkm_topk
        self.fwpkm_n_subkeys = fwpkm_n_subkeys
        self.fwpkm_variant = fwpkm_variant
        self.fwpkm_fp32_fw = fwpkm_fp32_fw
        self.fwpkm_before_attn = fwpkm_before_attn
        self.fwpkm_update_chunk_size = fwpkm_update_chunk_size
        self.fwpkm_loss_type = fwpkm_loss_type
        self.fwpkm_optimizer_type = fwpkm_optimizer_type
        self.fwpkm_optimizer_lr = fwpkm_optimizer_lr
        self.fwpkm_optimizer_weight_decay = fwpkm_optimizer_weight_decay
        self.fwpkm_grad_clip = fwpkm_grad_clip
        self.fwpkm_addr_loss = fwpkm_addr_loss
        self.fwpkm_addr_loss_weight = fwpkm_addr_loss_weight
        self.fwpkm_weight_loss_with_gates = fwpkm_weight_loss_with_gates
        self.fwpkm_mem_grad_to_values_only = fwpkm_mem_grad_to_values_only
        self.fwpkm_out_fuse_gate = fwpkm_out_fuse_gate
        self.fwpkm_query_src = fwpkm_query_src
        self.fwpkm_value_src = fwpkm_value_src
        self.fwpkm_target_value_lookahead = fwpkm_target_value_lookahead
        self.fwpkm_compress_value = fwpkm_compress_value
        self.fwpkm_score_nonlinear = fwpkm_score_nonlinear
        self.fwpkm_compress_query = fwpkm_compress_query
        self.fwpkm_qk_score_type = fwpkm_qk_score_type
        self.fwpkm_score_temperature = fwpkm_score_temperature

        self.num_fwpkm_layers = len(self.fwpkm_layers)


class Qwen3NextMemRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


class Qwen3NextMemDynamicCache:
    is_compileable = False

    def __init__(self, config: Qwen3NextMemConfig):
        super().__init__()
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers) if self.layer_types[i] in ["full_attention", "sliding_attention"]
        ]
        self.last_linear_layer = (
            len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
            if "linear_attention" in self.layer_types
            else None
        )
        self.first_fwpkm_layer = config.fwpkm_layers[0] if len(config.fwpkm_layers) > 0 else None

        # Initialize everything to None -> will be lazy initialized to allow multi-gpu (device_map) inference
        self.conv_states = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states = [None for _ in range(config.num_hidden_layers)]
        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        # (q, hyp_v, ref_v, gates, mask, ids)
        self.fwpkm_cache = [(None, None, None, None, None, None) for _ in range(config.num_hidden_layers)]

    def __len__(self):
        return len(self.layer_types)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset_kv_cache(self):
        """Resets the key and value caches for all layers."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = None
            self.value_cache[layer_idx] = None

    def reset_fwpkm_cache(self):
        """Resets the FwPKM cache for all layers."""
        for layer_idx in range(len(self.fwpkm_cache)):
            self.fwpkm_cache[layer_idx] = (None, None, None, None, None, None)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        if len(self.transformer_layers) == 0:
            return 0
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_fwpkm_cache_length(self) -> int:
        """Returns the length of the FwPKM cache, which is determined by the first FwPKM layer."""
        if self.first_fwpkm_layer is None:
            return None
        fwpkm_cache = self.fwpkm_cache[self.first_fwpkm_layer]
        if fwpkm_cache[0] is None:
            return 0
        return fwpkm_cache[0].shape[1]

    @property
    def has_previous_state(self):
        """We have a previous state if the last linear (conv) layer was already updated."""
        return self.conv_states[self.last_linear_layer] is not None


class Qwen3NextMemRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3NextMemConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3NextMemRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3NextMemAttention(nn.Module):
    def __init__(self, config: Qwen3NextMemConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3NextMemRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3NextMemRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        raw_attn_output = attn_output
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)

        # Reshape back to (batch_size, q_seq_len, hidden_size)
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2)[:, -input_shape[1] :].contiguous()
        value_states = value_states.transpose(1, 2)[:, -input_shape[1] :].contiguous()

        return query_states, key_states, value_states, raw_attn_output, attn_output, attn_weights


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3NextMemGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3NextMemConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            Qwen3NextMemRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_current_dtype(),
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Qwen3NextMemDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        raw_output = core_attn_out

        output = self.out_proj(core_attn_out)

        return query, key, value, raw_output, output


class Qwen3NextMemMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3NextMemDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3NextMemConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.fwpkm_layer_idx = config.fwpkm_layers.index(layer_idx) if layer_idx in config.fwpkm_layers else None

        # token mixer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextMemGatedDeltaNet(config, layer_idx)
        elif self.layer_type in ["full_attention", "sliding_attention"]:
            self.self_attn = Qwen3NextMemAttention(config, layer_idx)
        else:
            raise NotImplementedError(f"Layer type {self.layer_type} not implemented")

        self.input_layernorm = Qwen3NextMemRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextMemRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx in config.pkm_layers:
            self.pkm = HashingMemory(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                mem_k_dim=config.pkm_k_dim,
                mem_v_dim=config.pkm_v_dim,
                mem_heads=config.pkm_heads,
                mem_knn=config.pkm_topk,
                mem_n_keys=config.pkm_n_subkeys,
                mem_share_values=False,
                mem_query_rmsnorm=config.pkm_query_rmsnorm,
            )
            self.mlp = None
        else:
            self.pkm = None
            self.mlp = Qwen3NextMemMLP(config, intermediate_size=config.intermediate_size)

        if layer_idx in config.fwpkm_layers:
            # Query
            fwpkm_query_src_dim = self.get_src_state_size(config.fwpkm_query_src)
            self.fw_q_norm = Qwen3NextMemRMSNorm(fwpkm_query_src_dim, eps=config.rms_norm_eps)
            self.fw_q_proj = nn.Linear(fwpkm_query_src_dim, config.fwpkm_k_dim * config.fwpkm_heads)

            # Value
            fwpkm_value_src_dim = self.get_src_state_size(config.fwpkm_value_src)
            self.fw_v_norm = Qwen3NextMemRMSNorm(fwpkm_value_src_dim, eps=config.rms_norm_eps)
            self.fw_v_proj = nn.Linear(fwpkm_value_src_dim, config.fwpkm_v_dim)

            if config.fwpkm_variant == "pkm":
                self.fwpkm = FastWeightProductKeyMemory(
                    mem_k_dim=config.fwpkm_k_dim,
                    mem_v_dim=config.fwpkm_v_dim,
                    mem_heads=config.fwpkm_heads,
                    mem_topk=config.fwpkm_topk,
                    mem_n_subkeys=config.fwpkm_n_subkeys,
                    lookahead=config.fwpkm_target_value_lookahead,
                    qk_score_type=config.fwpkm_qk_score_type,
                    optimizer_type=config.fwpkm_optimizer_type,
                    learning_rate=config.fwpkm_optimizer_lr,
                    weight_decay=config.fwpkm_optimizer_weight_decay,
                    loss_type=config.fwpkm_loss_type,
                    mem_grad_to_values_only=config.fwpkm_mem_grad_to_values_only,
                    grad_clip=config.fwpkm_grad_clip,
                    addr_loss=config.fwpkm_addr_loss,
                    addr_loss_weight=config.fwpkm_addr_loss_weight,
                    fp32_fw=config.fwpkm_fp32_fw,
                    score_nonlinear=config.fwpkm_score_nonlinear,
                    score_temperature=config.fwpkm_score_temperature,
                )
            elif config.fwpkm_variant == "mlp":
                assert config.fwpkm_heads == 1
                self.fwpkm = FastWeightMLP(
                    input_dim=config.fwpkm_k_dim,
                    output_dim=config.fwpkm_v_dim,
                    size=config.fwpkm_n_subkeys,
                    lookahead=config.fwpkm_target_value_lookahead,
                    optimizer_type=config.fwpkm_optimizer_type,
                    learning_rate=config.fwpkm_optimizer_lr,
                    weight_decay=config.fwpkm_optimizer_weight_decay,
                    grad_clip=config.fwpkm_grad_clip,
                )
            else:
                raise NotImplementedError(f"FwPKM variant {config.fwpkm_variant} not implemented")

            if config.fwpkm_out_fuse_gate:
                # out size cannot be 1, 2, 4 due to compiling error
                self.fw_out_gate_norm = Qwen3NextMemRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.fw_out_gate = nn.Linear(config.hidden_size, 8)

            self.fw_out_norm = Qwen3NextMemRMSNorm(config.fwpkm_v_dim, eps=config.rms_norm_eps)
            self.fw_out_proj = nn.Linear(config.fwpkm_v_dim, config.hidden_size)
        else:
            self.fwpkm = None

    def get_src_state_size(self, state_src: str) -> int:
        hidden_state_size = self.config.hidden_size
        if self.layer_type == "linear_attention":
            query_state_size = self.config.linear_key_head_dim * self.config.linear_num_key_heads
            key_state_size = self.config.linear_key_head_dim * self.config.linear_num_key_heads
            value_state_size = self.config.linear_value_head_dim * self.config.linear_num_value_heads
            raw_output_state_size = value_state_size
        elif self.layer_type in ["full_attention", "sliding_attention"]:
            query_state_size = self.config.head_dim * self.config.num_attention_heads
            key_state_size = self.config.head_dim * self.config.num_key_value_heads
            value_state_size = self.config.head_dim * self.config.num_key_value_heads
            raw_output_state_size = query_state_size
        else:
            raise NotImplementedError(f"Layer type {self.layer_type} not implemented")

        if state_src in ["hidden", "output"]:
            return hidden_state_size
        elif state_src == "query":
            return query_state_size
        elif state_src == "key":
            return key_state_size
        elif state_src == "value":
            return value_state_size
        elif state_src == "raw_output":
            return raw_output_state_size
        else:
            raise NotImplementedError(f"FwPKM state source {state_src} not implemented")

    def fwpkm_construct_states(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        raw_attn_output_states: torch.Tensor,
        attn_output_states: torch.Tensor,
        state_src: str,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.size()

        if state_src == "hidden":
            states = hidden_states
        elif state_src == "key":
            states = key_states
        elif state_src == "query":
            states = query_states
        elif state_src == "value":
            states = value_states
        elif state_src == "raw_output":
            states = raw_attn_output_states
        elif state_src == "output":
            states = attn_output_states
        else:
            raise NotImplementedError(f"FwPKM state source {state_src} not implemented")
        return states

    def compute_gates(self, gate_input: torch.Tensor) -> torch.Tensor:
        gate_input = self.fw_out_gate_norm(gate_input)
        gate_values = self.fw_out_gate(gate_input).mean(dim=-1, keepdim=True)
        gate_values = torch.sigmoid(gate_values)
        return gate_values

    def compress_fwpkm_inputs(self, type, x, two_parts=False):
        def _compress(type, x_part):
            if type == "l2norm":
                return l2norm(x_part)
            elif type == "zero_mean":
                mean = x_part.mean(dim=-1, keepdim=True)
                std = x_part.std(dim=-1, keepdim=True) + 1e-6
                return (x_part - mean) / std
            else:
                raise NotImplementedError(f"FwPKM input compression type {type} not implemented")

        if two_parts:
            half_dim = x.size(-1) // 2
            x_part1 = x[:, :, :half_dim]
            x_part2 = x[:, :, half_dim:]
            x_part1 = _compress(type, x_part1)
            x_part2 = _compress(type, x_part2)
            return torch.cat([x_part1, x_part2], dim=-1)
        else:
            return _compress(type, x)

    @torch.no_grad()
    @torch.compiler.disable(recursive=True)
    def compute_addr_stats(
        self,
        idcs: torch.LongTensor,
        addr_loss: Optional[torch.FloatTensor] = None,
    ) -> dict[str, float]:
        def _compute_addr_stats(self, idcs: torch.LongTensor) -> dict[str, float]:
            addr_stats = {}
            idx_counter = torch.bincount(idcs.view(-1), minlength=self.config.fwpkm_n_subkeys**2)
            addr_stats["collision_ratio"] = idx_counter[idx_counter > 1].sum().item() / (idcs.numel() + 1e-6)
            addr_stats["coverage_ratio"] = (idx_counter > 0).sum().item() / self.config.fwpkm_n_subkeys**2
            addr_stats["kld"] = torch.distributions.kl.kl_divergence(
                torch.distributions.Categorical(probs=idx_counter.float() / (idx_counter.sum() + 1e-6)),
                torch.distributions.Categorical(probs=torch.ones_like(idx_counter).float() / idx_counter.numel()),
            ).item()
            return addr_stats

        full_addr_stats = {}
        addr_stats = _compute_addr_stats(self, idcs)
        for k, v in addr_stats.items():
            full_addr_stats[f"{k}"] = v
        if addr_loss is not None:
            full_addr_stats[f"addressing_loss/{self.config.fwpkm_addr_loss}"] = addr_loss.mean().item()
        return full_addr_stats

    def fwpkm_forward(
        self,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        raw_attn_output_states: torch.Tensor,
        attn_output_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
    ):
        residual = hidden_states

        # Query projection
        fwpkm_query_states = self.fwpkm_construct_states(
            hidden_states=hidden_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            raw_attn_output_states=raw_attn_output_states,
            attn_output_states=attn_output_states,
            state_src=self.config.fwpkm_query_src,
        )
        fwpkm_query_states = self.fw_q_norm(fwpkm_query_states)
        fwpkm_query_states = self.fw_q_proj(fwpkm_query_states)

        # Value projection
        fwpkm_value_states = self.fwpkm_construct_states(
            hidden_states=hidden_states,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            raw_attn_output_states=raw_attn_output_states,
            attn_output_states=attn_output_states,
            state_src=self.config.fwpkm_value_src,
        )
        fwpkm_value_states = self.fw_v_norm(fwpkm_value_states)
        fwpkm_value_states = self.fw_v_proj(fwpkm_value_states)

        # Compression
        if self.config.fwpkm_compress_query is not None:
            fwpkm_query_states = self.compress_fwpkm_inputs(
                self.config.fwpkm_compress_query,
                fwpkm_query_states,
                two_parts=True,
            )
        if self.config.fwpkm_compress_value is not None:
            fwpkm_value_states = self.compress_fwpkm_inputs(
                self.config.fwpkm_compress_value,
                fwpkm_value_states,
                two_parts=False,
            )

        # Gating projection
        gate_values = None
        if self.config.fwpkm_out_fuse_gate:
            gate_values = self.compute_gates(hidden_states)  # [batch, seq_len, 1]

        # attention mask: 4D -> 2D
        if attention_mask is not None and attention_mask.dim() == 4:
            attention_mask = attention_mask[:, 0, -1, :]

        # FwPKM forward
        if self.config.fwpkm_variant == "pkm":
            fwpkm_output_dict = self.fwpkm(
                q=fwpkm_query_states,
                ref_v=fwpkm_value_states,
                gates=gate_values.squeeze(-1) if self.config.fwpkm_weight_loss_with_gates else None,
                chunk_size=self.config.fwpkm_update_chunk_size,
                loss_mask=attention_mask,
                past_key_values=past_key_values.fwpkm_cache[self.layer_idx] if past_key_values is not None else None,
                token_ids=input_ids,
            )
            fwpkm_output = fwpkm_output_dict["output"]
            fwpkm_idcs = fwpkm_output_dict["indices"]
            fwpkm_scores = fwpkm_output_dict["scores"]
            fwpkm_losses = fwpkm_output_dict["losses"]
            fwpkm_grad_norms = fwpkm_output_dict["grad_norms"]
            fwpkm_addr_loss = fwpkm_output_dict["addr_loss"]
            fwpkm_past_key_values = fwpkm_output_dict["past_key_values"]
        elif self.config.fwpkm_variant == "mlp":
            fwpkm_output_dict = self.fwpkm(
                q=fwpkm_query_states,
                ref_v=fwpkm_value_states,
                chunk_size=self.config.fwpkm_update_chunk_size,
                loss_mask=attention_mask,
                past_key_values=past_key_values.fwpkm_cache[self.layer_idx] if past_key_values is not None else None,
            )
            fwpkm_output = fwpkm_output_dict["output"]
            fwpkm_idcs = None
            fwpkm_scores = None
            fwpkm_losses = fwpkm_output_dict["losses"]
            fwpkm_grad_norms = fwpkm_output_dict["grad_norms"]
            fwpkm_addr_loss = None
            fwpkm_past_key_values = fwpkm_output_dict["past_key_values"]
        else:
            raise NotImplementedError(f"FwPKM variant {self.config.fwpkm_variant} not implemented")

        # Update past key values
        if past_key_values is not None:
            past_key_values.fwpkm_cache[self.layer_idx] = fwpkm_past_key_values

        # Addressing stats
        if fwpkm_idcs is not None:
            fwpkm_addr_stats = self.compute_addr_stats(fwpkm_idcs, fwpkm_addr_loss)
        else:
            fwpkm_addr_stats = None

        # Gating
        if self.config.fwpkm_out_fuse_gate:
            fwpkm_output = fwpkm_output * gate_values

        # Residual value
        if self.config.fwpkm_out_fuse_gate:
            fwpkm_value_states = fwpkm_value_states * (1.0 - gate_values)
        fwpkm_output = fwpkm_output + fwpkm_value_states

        # Output projection
        fwpkm_output = self.fw_out_norm(fwpkm_output)
        fwpkm_output = self.fw_out_proj(fwpkm_output)

        # Final residual connection
        hidden_states = residual + fwpkm_output

        return (
            hidden_states,
            fwpkm_output,
            fwpkm_idcs,
            fwpkm_scores,
            fwpkm_addr_stats,
            fwpkm_losses,
            fwpkm_grad_norms,
            fwpkm_past_key_values,
            gate_values,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        B, T = hidden_states.size(0), hidden_states.size(1)

        # FwPKM outputs
        fwpkm_output = None
        fwpkm_idcs = None
        fwpkm_scores = None
        fwpkm_addr_stats = None
        fwpkm_losses = None
        fwpkm_grad_norms = None
        fwpkm_past_key_values = None
        gate_values = None

        # FwPKM (before attention)
        if self.fwpkm is not None and self.config.fwpkm_before_attn:
            assert self.config.fwpkm_query_src in ("hidden")
            assert self.config.fwpkm_value_src in ("hidden")
            (
                hidden_states,
                fwpkm_output,
                fwpkm_idcs,
                fwpkm_scores,
                fwpkm_addr_stats,
                fwpkm_losses,
                fwpkm_grad_norms,
                fwpkm_past_key_values,
                gate_values,
            ) = self.fwpkm_forward(
                hidden_states=hidden_states,
                query_states=None,
                key_states=None,
                value_states=None,
                raw_attn_output_states=None,
                attn_output_states=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # Token Mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            query_states, key_states, value_states, raw_attn_output_states, attn_output_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type in ["full_attention", "sliding_attention"]:
            query_states, key_states, value_states, raw_attn_output_states, attn_output_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values if use_cache else None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + attn_output_states

        # Fast weight PKM (after attention)
        if self.fwpkm is not None and not self.config.fwpkm_before_attn:
            (
                hidden_states,
                fwpkm_output,
                fwpkm_idcs,
                fwpkm_scores,
                fwpkm_addr_stats,
                fwpkm_losses,
                fwpkm_grad_norms,
                fwpkm_past_key_values,
                gate_values,
            ) = self.fwpkm_forward(
                hidden_states=hidden_states,
                query_states=query_states.view(B, T, -1).contiguous(),
                key_states=key_states.view(B, T, -1).contiguous(),
                value_states=value_states.view(B, T, -1).contiguous(),
                raw_attn_output_states=raw_attn_output_states.view(B, T, -1).contiguous(),
                attn_output_states=attn_output_states.view(B, T, -1).contiguous(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.pkm is not None:
            hidden_states, pkm_idcs = self.pkm(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            pkm_idcs = None
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states

        hidden_states = residual + hidden_states

        return {
            "hidden_states": hidden_states,
            "pkm_idcs": pkm_idcs,
            "fwpkm_losses": fwpkm_losses,
            "fwpkm_idcs": fwpkm_idcs,
            "fwpkm_scores": fwpkm_scores,
            "fwpkm_addr_stats": fwpkm_addr_stats,
            "fwpkm_grad_norms": fwpkm_grad_norms,
            "fwpkm_gates": gate_values.detach() if gate_values is not None else None,
            "fwpkm_past_key_values": fwpkm_past_key_values,
        }


class Qwen3NextMemPreTrainedModel(PreTrainedModel):
    config: Qwen3NextMemConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3NextMemDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "hidden_states": Qwen3NextMemDecoderLayer,
        "attentions": Qwen3NextMemAttention,
    }
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3NextMemGatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.uniform_(0, 16).log_()
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3NextMemRMSNorm):
            module.weight.data.zero_()
        elif isinstance(
            module,
            (
                FastWeightProductKeyMemory,
                HashingMemory,
                FastWeightMLP,
            ),
        ):
            module.reset_parameters()


class Qwen3NextMemModel(Qwen3NextMemPreTrainedModel):
    def __init__(self, config: Qwen3NextMemConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3NextMemDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextMemRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextMemRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3NextMemModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3NextMemDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        sliding_causal_mask = (
            create_sliding_window_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if self.config.sliding_window
            else None
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds

        all_hidden_states = []
        all_pkm_idcs = []
        all_fwpkm_losses = []
        all_fwpkm_idcs = []
        all_fwpkm_scores = []
        all_fwpkm_addr_stats = []
        all_fwpkm_grad_norms = []
        all_fwpkm_gates = []

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if decoder_layer.layer_type == "linear_attention":
                layer_mask = linear_attn_mask
            elif decoder_layer.layer_type == "full_attention":
                layer_mask = causal_mask
            elif decoder_layer.layer_type == "sliding_attention":
                layer_mask = sliding_causal_mask

            layer_output_dict = decoder_layer(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                all_hidden_states=all_hidden_states,
                **kwargs,
            )
            hidden_states = layer_output_dict["hidden_states"]
            pkm_idcs = layer_output_dict["pkm_idcs"]
            fwpkm_losses = layer_output_dict["fwpkm_losses"]
            fwpkm_idcs = layer_output_dict["fwpkm_idcs"]
            fwpkm_scores = layer_output_dict["fwpkm_scores"]
            fwpkm_addr_stats = layer_output_dict["fwpkm_addr_stats"]
            fwpkm_grad_norms = layer_output_dict["fwpkm_grad_norms"]
            fwpkm_gates = layer_output_dict["fwpkm_gates"]

            all_hidden_states.append(hidden_states)

            if pkm_idcs is not None:
                all_pkm_idcs.append(pkm_idcs)
            if fwpkm_losses is not None:
                all_fwpkm_losses.append(fwpkm_losses)
            if fwpkm_idcs is not None:
                all_fwpkm_idcs.append(fwpkm_idcs)
            if fwpkm_scores is not None:
                all_fwpkm_scores.append(fwpkm_scores)
            if fwpkm_addr_stats is not None:
                all_fwpkm_addr_stats.append(fwpkm_addr_stats)
            if fwpkm_grad_norms is not None:
                all_fwpkm_grad_norms.append(fwpkm_grad_norms)
            if fwpkm_gates is not None:
                all_fwpkm_gates.append(fwpkm_gates)

        all_pkm_idcs = (
            torch.stack(all_pkm_idcs, dim=2) if all_pkm_idcs else None
        )  # (batch_size, seq_len, num_pkm_layers, knn)
        all_fwpkm_losses = (
            torch.stack(all_fwpkm_losses, dim=2) if all_fwpkm_losses else None
        )  # (batch_size, seq_len, num_pkm_layers, hidden_size)
        all_fwpkm_idcs = (
            torch.stack(all_fwpkm_idcs, dim=2) if all_fwpkm_idcs else None
        )  # (batch_size, seq_len, num_fwpkm_layers, knn)
        all_fwpkm_scores = (
            torch.stack(all_fwpkm_scores, dim=2) if all_fwpkm_scores else None
        )  # [batch_size, seq_len, num_fwpkm_layers, knn]
        all_fwpkm_addr_stats = all_fwpkm_addr_stats  # num_fwpkm_layers x dict
        all_fwpkm_grad_norms = all_fwpkm_grad_norms  # num_fwpkm_layers x (num_fw)
        all_fwpkm_gates = (
            torch.stack(all_fwpkm_gates, dim=2) if all_fwpkm_gates else None
        )  # (batch_size, seq_len, num_fwpkm_layers, 1)

        hidden_states = self.norm(hidden_states)

        return Qwen3NextMemModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            all_pkm_idcs=all_pkm_idcs,
            all_fwpkm_losses=all_fwpkm_losses,
            all_fwpkm_idcs=all_fwpkm_idcs,
            all_fwpkm_scores=all_fwpkm_scores,
            all_fwpkm_addr_stats=all_fwpkm_addr_stats,
            all_fwpkm_grad_norms=all_fwpkm_grad_norms,
            all_fwpkm_gates=all_fwpkm_gates,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask


class Qwen3NextMemForCausalLM(Qwen3NextMemPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3NextMemModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def adjust_fwpkm_update_chunksize(self, input_len: int, past_key_values: Qwen3NextMemDynamicCache = None):
        if past_key_values is None:
            self.config.fwpkm_update_chunk_size = input_len
        else:
            fwpkm_cache_len = past_key_values.get_fwpkm_cache_length()
            if fwpkm_cache_len is not None:
                self.config.fwpkm_update_chunk_size = input_len + fwpkm_cache_len
            else:
                self.config.fwpkm_update_chunk_size = input_len
        # print(f"Adjusted FwPKM update chunk size to {self.config.fwpkm_update_chunk_size}")

    @torch.compile
    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Qwen3NextMemDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3NextMemLMOutput:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: Qwen3NextMemModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3NextMemLMOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_pkm_idcs=outputs.all_pkm_idcs,
            all_fwpkm_losses=outputs.all_fwpkm_losses,
            all_fwpkm_idcs=outputs.all_fwpkm_idcs,
            all_fwpkm_scores=outputs.all_fwpkm_scores,
            all_fwpkm_addr_stats=outputs.all_fwpkm_addr_stats,
            all_fwpkm_grad_norms=outputs.all_fwpkm_grad_norms,
            all_fwpkm_gates=outputs.all_fwpkm_gates,
        )

    def forward(self, no_compile: bool = False, **kwargs):
        fn = fn_by_compile_flag(self._forward, self, no_compile=no_compile)
        return fn(**kwargs)
