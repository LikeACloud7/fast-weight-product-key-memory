from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as TransformerMLP
from fla.modules import RMSNorm
import torch.nn as nn

from src.models.lact_model.layer_lact_swiglu_hf_rope import LaCTSWIGLULayer

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class LaCTSWIGLUConfig(PretrainedConfig):
    """
    Configuration for LaCT-SWIGLU model.
    It implements the LaCT-SWIGLU layer mixed with in-layer sliding window attention

    Args:
        hidden_size (int, optional): The hidden size of the model. Defaults to 2048.
        num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 24.
        num_attn_heads (int, optional): The number of attention heads in the model. Defaults to 32.
        num_lact_heads (int, optional): The number of feed-forward heads in the model. Defaults to 4.
    """

    model_type = "lact_swiglu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attn_heads: int = 32,
        num_attention_heads: int = 32,  # for compatibility
        num_lact_heads: int = 4,
        inter_multi: int = 1,
        qkv_bias: bool = False,
        attn_qk_norm: bool = False,
        lact_chunk_size: int = 2048,
        use_muon: bool = False,
        lr_dim: int = 1,
        qkv_silu: bool = True,  # if True, apply silu to q, k, v.
        no_v_silu: bool = False,  # if True, don't apply silu to v, will overwrite qkv_silu.
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = True,
        use_momentum: bool = True,
        ttt_loss_type: str = "dot_product",  # "l2"
        ttt_prenorm: bool = False,  # pre-norm or post-norm for ttt.
        # prenorm ttt:  state = state + f(norm(state))
        # postnorm ttt:  state = norm(state + f(state)
        ttt_nope: bool = False,  # if True, no positional encoding for query and key used in ttt.
        w0_w2_low_rank: int = -1,  # -1 means fully learnable.  > 1 means low rank parameterization of the initial learnable weights.
        window_size: int = 2048,
        rope_theta: Optional[float] = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.006,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        last_layer_fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        fw_init_gain: float = 0.5,
        use_fused_kernel: bool = False,  # use triton kernel for ttt implementation
        fp32_states: bool = False,  # whether to keep the fast weights in fp32
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_heads = num_attn_heads
        self.num_attention_heads = num_attention_heads
        self.num_lact_heads = num_lact_heads
        self.inter_multi = inter_multi
        self.qkv_bias = qkv_bias
        self.attn_qk_norm = attn_qk_norm
        self.lact_chunk_size = lact_chunk_size
        self.use_muon = use_muon
        self.lr_dim = lr_dim
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.window_size = window_size
        self.lr_parameterization = lr_parameterization
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.last_layer_fuse_norm = last_layer_fuse_norm  # seems that you need to set this to False to use activation checkpointing for every layer.
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        self.use_momentum = use_momentum
        self.ttt_loss_type = ttt_loss_type
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class HFRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LaCTSWIGLUConfig, device=None):
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


class LaCTBlock(nn.Module):
    def __init__(self, config: LaCTSWIGLUConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = LaCTSWIGLULayer(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            qkv_bias=config.qkv_bias,
            attn_qk_norm=config.attn_qk_norm,
            qkv_silu=config.qkv_silu,
            no_v_silu=config.no_v_silu,
            lr_dim=config.lr_dim,
            use_muon=config.use_muon,
            ttt_prenorm=config.ttt_prenorm,
            ttt_nope=config.ttt_nope,
            lr_parameterization=config.lr_parameterization,
            learnable_ttt_scale=config.learnable_ttt_scale,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            w0_w2_low_rank=config.w0_w2_low_rank,
            use_momentum=config.use_momentum,
            ttt_loss_type=config.ttt_loss_type,
            fw_init_gain=config.fw_init_gain,
            use_fused_kernel=config.use_fused_kernel,
            fp32_states=config.fp32_states,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class LaCTPreTrainedModel(PreTrainedModel):
    config_class = LaCTSWIGLUConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LaCTBlock"]
    _supports_cache_class = True
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

        if isinstance(module, LaCTSWIGLULayer):
            #### Initialize the parameters of the model
            nn.init.ones_(module.qk_scale)
            nn.init.zeros_(module.qk_offset)

            logger.info(f"in PreTrainedModel initialize fast weights for LaCTSWIGLULayer")
            # init w0, w1, w2
            if module.w0_w2_low_rank > 0:
                module.w0._init_weights()
                module.w2._init_weights()
            else:
                nn.init.normal_(module.w0, mean=0.0, std=1.0 / math.sqrt(module.fw_head_dim))
                nn.init.normal_(module.w2, mean=0.0, std=1.0 / math.sqrt(module.fw_head_dim))

            nn.init.normal_(module.w1, mean=0.0, std=1.0 / math.sqrt(module.d_h))


class LaCTModel(LaCTPreTrainedModel):
    def __init__(self, config: LaCTSWIGLUConfig) -> LaCTModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LaCTBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # for flame, full act_ckpt will throw error if we fuse the last layer norm,
        self.norm = (RMSNorm if config.last_layer_fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.rotary_emb = HFRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if position_ids is None:
            seq_length = hidden_states.size(1)
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(hidden_states.size(0), -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    **kwargs,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class LaCTForCausalLM(LaCTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LaCTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        use_cache = False
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0 and use_cache:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        model_inputs.update(
            {
                "past_key_values": None,  # past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        use_cache = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], criterion.ignore_index),
                ),
                1,
            )
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
