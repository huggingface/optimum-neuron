import torch
from models.gqa import (  # noqa: E402
    determine_sharding_strategy,  # noqa: E402
    get_shardable_head_counts,  # noqa: E402
)  # noqa: E402
from modules.cache import NeuronStaticCache
from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402


class DecoderModelWrapper(torch.nn.Module):
    """Wraps the calls to the traced decoder models

    This wrapper is common to all prefill and encode models, but the forward() path
    is different if the traced model is intended to perform prefill and decode.

    The selection between the two forward paths is done at initialization by passing
    a special flag.

    The wrapper role is to:
    - prepare named inputs for the underlying model,
    - initialize and update and prepare the KV cache entries.
    """

    def __init__(
        self, model, batch_size: int, max_length: int, tensor_parallel_size: int, dtype: torch.dtype, is_prefill: bool
    ) -> None:
        assert parallel_state.model_parallel_is_initialized()
        super().__init__()
        self.model = model
        self.is_prefill = is_prefill
        hidden_size = model.config.hidden_size
        num_key_value_heads = model.config.num_key_value_heads
        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        gqa_sharding_strategy = determine_sharding_strategy(tensor_parallel_size, num_key_value_heads)
        _, num_key_value_heads = get_shardable_head_counts(
            tensor_parallel_size, num_attention_heads, num_key_value_heads, gqa_sharding_strategy
        )
        num_kv_heads_per_partition = utils.divide(num_key_value_heads, tensor_parallel_size)

        hidden_dim_per_head = hidden_size // num_attention_heads

        self.kv_cache = NeuronStaticCache(
            max_batch_size=batch_size,
            max_length=max_length,
            num_kv_heads_per_partition=num_kv_heads_per_partition,
            hidden_dim_per_head=hidden_dim_per_head,
            num_hidden_layers=num_hidden_layers,
            dtype=dtype,
        )

    def _create_context_attn_mask(self, attention_mask, n_positions):
        batch_size = attention_mask.shape[0]
        mask = torch.full((n_positions, n_positions), True, device=attention_mask.device).tril(diagonal=0)
        return mask[None, None, :, :].expand(batch_size, 1, n_positions, n_positions)

    def _create_simple_attn_mask(self, attention_mask, n_positions):
        batch_size = attention_mask.shape[0]
        return attention_mask[:, None, None, :].expand(batch_size, 1, 1, n_positions).to(torch.bool)

    def prefill(self, input_ids, attention_mask, position_ids, seq_ids):
        past_key_values = None
        # Prepare 4D attention mask
        attention_mask = self._create_context_attn_mask(
            attention_mask,
            n_positions=self.n_positions,
        )
        # Actual model call
        outputs, past_key_values = self.model(input_ids, attention_mask, position_ids, past_key_values=None)
        # Extract updated kv cache tensors and return them: this seems required by the tracing code
        updated_kv_cache = []
        for layer_idx, kv_per_layer in enumerate(past_key_values):
            key_states, value_states = kv_per_layer
            k_cache, v_cache = self.kv_cache.add(key_states, value_states, layer_idx, self.n_positions)
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        return [outputs] + updated_kv_cache

    def decode(self, input_ids, attention_mask, position_ids, seq_ids):
        past_key_values = []
        for layer_idx in range(0, self.model.config.num_hidden_layers):
            key_state, value_state = self.kv_cache.get_sliced_kv_cache(layer_idx, seq_len=self.n_positions)
            past_key_values.append([key_state, value_state])
        # Prepare 4D attention mask
        attention_mask = self._create_simple_attn_mask(
            attention_mask,
            n_positions=self.n_positions,
        )
        # Actual model call
        outputs, past_key_values = self.model(input_ids, attention_mask, position_ids, past_key_values=past_key_values)
        # Extract updated kv cache tensors and return them: this seems required by the tracing code
        updated_kv_cache = []
        for layer_idx, kv_per_layer in enumerate(past_key_values):
            key_states, value_states = kv_per_layer
            k_cache, v_cache = self.kv_cache.append(
                key_states, value_states, position_ids, layer_idx, seq_len=self.n_positions
            )
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        return [outputs] + updated_kv_cache

    def forward(self, input_ids, attention_mask, position_ids, seq_ids):
        assert attention_mask is not None
        assert position_ids is not None

        # The same model is traced twice for prefill and decode, with different forward path
        if self.is_prefill:
            return self.prefill(input_ids, attention_mask, position_ids, seq_ids)
        return self.decode(input_ids, attention_mask, position_ids, seq_ids)
