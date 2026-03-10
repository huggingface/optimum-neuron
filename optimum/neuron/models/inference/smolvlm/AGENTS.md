# SmolVLM (Idefics3) Inference Model Guide

This directory contains the Neuron-optimized SmolVLM/Idefics3 inference implementation.
It extends the NxD decoder stack with a separately compiled vision encoder and on-device
image-token feature injection.

For shared NxD decoder guidance, read [optimum/neuron/models/inference/AGENTS.md](../AGENTS.md).

## HF references
- Transformers Idefics3 modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics3

## Core architecture in this implementation

### Compiled artifacts (named bundles)
Compiled bundles are saved with a named scheme defined in
[backend/pretrained_model.py](../backend/pretrained_model.py).
Multi-bundle models (like VLMs) use `model_{bundle_name}.pt`:

- `model_vision.pt`: vision encoder bundle — SigLIP vision encoder + HF `Idefics3Connector` traced with ModelBuilder.
- `model_text.pt`: text decoder graph bundle — context/chunked-prefill + token generation.

For SmolVLM, there are exactly two compiled artifacts: `model_vision.pt` and `model_text.pt`.

Both are managed by `SmolVLMNxDModelForImageTextToText` in [modeling_smolvlm.py](modeling_smolvlm.py).

### Vision stack is custom SigLIP-compatible
`NeuronIdefics3VisionEncoder` is built from:
- `NeuronSigLIPVisionEmbeddings`
- `NeuronSigLIPEncoderLayer` / `NeuronSigLIPEncoder`
- `NeuronSigLIPVisionTransformer`
- HF `Idefics3Connector`

The custom embeddings intentionally keep HF key names (`patch_embedding`, `position_embedding`) so
state dict loading maps directly.

### Position IDs are precomputed for static compiled image size
`NeuronSigLIPVisionEmbeddings` precomputes position IDs using HF fractional-coordinate bucketing
for a full patch mask. This is important for numerical parity with HF embeddings.

### Decoder remains Llama-style text core
- `NxDSmolVLMDecoderModel` subclasses `NxDLlamaModel` and uses `config.text_config`.
- VLM-specific behavior is injected by VLM wrappers/builders, not by replacing the text stack.

## VLM-specific forward behavior

### Pixel values path through generate
`generate()` stores `pixel_values` on `self._current_pixel_values` because `_sample()` in the base
path does not forward `pixel_values` into `forward()`.

### Context/chunked-prefill uses image injection tensors
For context encoding (or each prefill chunk), the model builds:
- `image_embeds`: shape `[B, S, H]`
- `image_token_mask`: shape `[B, S]` (bool)

Image features are inserted at `image_token_id` positions and passed into VLM wrappers:
- `NxDDecoderWrapperForImageTextToText` (context encoding / chunked prefill)
- `NxDTokenGenerationWrapperForImageTextToText` (token generation — passes dummy image tensors to match the uniform 6-tensor compiled signature)

### Vision feature mapping rules
- `pixel_values` supports `[B, N, C, H, W]` or `[B, C, H, W]`.
- Tiles are flattened and cast to Neuron dtype before vision forward.
- Inputs are padded/truncated to compiled vision batch size:
	`batch_size * max_num_images`.
- Only features for real (non-padding) tiles are used.
- Feature rows are consumed in order and mapped to `image_token_id` positions in token order.
- If image token count exceeds available features, a warning is logged and remaining positions stay text-only.

## Export/load details that matter

### Export compiles all bundles via NxDPreTrainedModel.compile()
`_export()` (defined in `NxDModelForImageTextToText`, the shared VLM base class) calls
`create_graph_builders()` which returns `{"vision": {...}, "text": {...}}`, then passes
this to `NxDPreTrainedModel.compile()` which iterates over bundles and traces each one.
The compiled artifacts are saved as `model_vision.pt` and `model_text.pt`.

### Vision checkpoint loader avoids from_pretrained allocation path
`_vision_checkpoint_loader_fn()` (in `NxDModelForImageTextToText`) is returned by
`get_checkpoint_loader_fn("vision")` and used by the multi-bundle weight loading
infrastructure. It calls `_get_vision_encoder_state_dict()` (overridden in
`SmolVLMNxDModelForImageTextToText`) to extract and remap vision weights from the
full HF state dict:
- `model.vision_model.*` -> `vision_model.*`
- `model.connector.*` -> `connector.*`

This avoids `from_pretrained` allocation interactions under model-builder meta/init contexts.

### Saved weights re-initialization on load
`_from_pretrained()` (in `NxDPreTrainedModel`) calls `create_graph_builders()` to
determine the expected bundle names (`"vision"`, `"text"`), then loads each compiled
artifact by its expected filename (`model_vision.pt`, `model_text.pt`).
After constructing the model, `load_weights()` is called to restore the shard
checkpoints for all bundles.

## Neuron config behavior

### Auto `max_num_images`
`_get_neuron_config()` computes `max_num_images` to account for Idefics3 tiling
plus global view, based on `image_size` and a `2048` longest-edge assumption.

### Image sequence length derivation
`image_seq_len = (image_size // patch_size) ** 2 // (scale_factor ** 2)`.

## State dict conversion behavior

`convert_hf_to_neuron_state_dict()`:
- Removes `model.vision_model.*` and `model.connector.*` (vision lives in separate artifact).
- Optionally fuses QKV for text decoder (`convert_state_dict_to_fused_qkv`).
- Adds `rank_util.rank` tensors required by Neuron attention modules.

`update_state_dict_for_tied_weights()` ties `lm_head.weight` to `embed_tokens.weight` if missing.

## Key files
- [optimum/neuron/models/inference/smolvlm/modeling_smolvlm.py](modeling_smolvlm.py)
- [optimum/neuron/models/inference/backend/modules/decoder/vlm_decoder.py](../backend/modules/decoder/vlm_decoder.py)
- [optimum/neuron/models/inference/backend/modules/decoder/vlm_wrappers.py](../backend/modules/decoder/vlm_wrappers.py)
- [optimum/neuron/models/inference/backend/modules/decoder/vlm_builders.py](../backend/modules/decoder/vlm_builders.py)
