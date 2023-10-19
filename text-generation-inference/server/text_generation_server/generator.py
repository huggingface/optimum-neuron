import copy
import logging
from abc import ABC
from enum import Enum
from typing import List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.generation import GenerationConfig

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.generation import TokenSelector

from .model import fetch_model
from .pb.generate_pb2 import (
    Batch,
    CachedBatch,
    FinishReason,
    GeneratedText,
    Generation,
    InfoResponse,
    Request,
)


# Disable optimum-neuron warnings as it seems to block the server after a while
optimum_logger = logging.getLogger("optimum.neuron")
optimum_logger.setLevel("CRITICAL")


class Generator(ABC):
    """An abstract class to represent the workhorse behind TextGenerationService.

    Ideally, it should not rely on protobuf constructs, but in a first step it does.
    Implementations would typically need a model and a tokenizer to implement the Generator methods.
    """

    @property
    def info(self) -> InfoResponse:
        """This should simply return the expected InfoResponse"""
        raise NotImplementedError

    def warmup(self, batch: Batch) -> int:
        """Verify if the hardware can support the target load.

        Args:
            batch (`Batch`):
                A batch corresponding to the maximum number of concurrent requests.

        Return:
            The maximum number of tokens the model supports.
        """
        raise NotImplementedError

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill is called whenever new requests need to be added.

        When this method returns successfully, a decode method will follow
        with both the current and newly prefilled batch(es).

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        raise NotImplementedError

    def decode(self, batches: List[Batch]) -> Tuple[List[Generation], CachedBatch]:
        """Decode after a prefill or another decode."""
        raise NotImplementedError

    def filter(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        """Remove requests that are not listed from the specified batch"""
        raise NotImplementedError

    def clear(self):
        """Remove all requests from the generator"""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_id: str, revision: Optional[str]):
        """Factory method "a la transformers" """
        raise NotImplementedError


class Slot:
    """Represents a slot in a static batch"""

    class State(Enum):
        EMPTY = 0
        PAUSE = 1
        READY = 2

    def __init__(self, id: int):
        self._id = id
        self.clear()

    def clear(self):
        """Clear the slot and mark it as available."""
        self._state = Slot.State.EMPTY
        self._request_id = None
        self._inputs = ""
        self._generation_config = None
        self._tokens = []
        self._mask = []
        self._selector = None
        self._generated_tokens = 0
        self._next_token_text = ""

    @property
    def id(self) -> int:
        return self._id

    @property
    def state(self) -> "Slot.State":
        return self._state

    @property
    def request_id(self) -> int:
        return self._request_id

    @property
    def inputs(self) -> str:
        return self._inputs

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def generated_tokens(self) -> torch.LongTensor:
        return self._generated_tokens

    def assign(self, request: Request, generation_config: GenerationConfig):
        """Assign a request to a slot.

        Args:
            request (`Request`):
                The request to be assigned. Contains the inputs and tokens selection parameters.
            generation_config (`transformers.GenerationConfig`):
                The base generation config (might be modified by the request generation parameters).
        """
        self._state = Slot.State.READY
        self._request_id = request.id
        self._inputs = request.inputs
        self._generation_config = copy.deepcopy(generation_config)
        # Update generation config with token chooser parameters
        self._generation_config.temperature = request.parameters.temperature
        self._generation_config.top_k = request.parameters.top_k
        self._generation_config.top_p = request.parameters.top_p
        self._generation_config.typical_p = request.parameters.typical_p
        self._generation_config.do_sample = request.parameters.do_sample
        self._generation_config.repetition_penalty = request.parameters.repetition_penalty
        # TODO: seed, watermark
        self._generation_config.max_new_tokens = request.stopping_parameters.max_new_tokens
        # TODO: stop_sequences, ignore_eos_token

    def reset(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, selector: TokenSelector):
        """Reset the slot for the next generation.

        Args:
            input_ids: (`torch.LongTensor`):
                The new input_ids to use to generate the next token.
            attention_mask: (`torch.LongTensor`):
                The new attention_mask to use to generate the next token.
            selector: (`optimum.neuron.generation.TokenSelector`):
                An object implementing the updated token selection logic.
        """
        self._tokens = input_ids.clone()
        self._mask = attention_mask.clone()
        self._selector = selector

    def pause(self):
        """Mark the current slot as paused for generation.

        Note that the KV cache for this slot will still be filled.
        """
        self._state = Slot.State.PAUSE

    def resume(self):
        """Mark the slot as ready for generation."""
        if self._state == Slot.State.PAUSE and self.next_token is not None:
            # The generation of this slot was inhibited during a prefill, but it
            # already had a pending token, so we need to increase attention mask
            self._mask = torch.cat([self._mask, torch.LongTensor([1])])
        self._state = Slot.State.READY

    def append(self, next_token: int, next_token_text: str):
        """Append a new generated token to this slot

        The new token is added to the list of generated tokens, which impacts
        directly the generated_text and stopped property.

        The new token is however not added immediately to the slot inputs: it will
        be added later on when it has effectively been used to produce the next token.

        Args:
            next_token (`int`):
                The newly generated token.
            next_token_test (`str`):
                The corresponding decoded text.
        """
        self._tokens = torch.cat([self._tokens, torch.LongTensor([next_token])])
        self._mask = torch.cat([self._mask, torch.LongTensor([1])])
        self._generated_tokens += 1
        # Now that a new token has been generated, we can append the previous one to the inputs
        self._inputs += self._next_token_text
        self._next_token_text = next_token_text

    def select(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.LongTensor:
        """Select the next token from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A scalar torch.LongTensor` containing the selected token.
        """
        return self._selector.select(input_ids, logits)[0]

    @property
    def stopped(self) -> bool:
        return self._selector.stopping_criteria(self._tokens, None)

    @property
    def generated_text(self) -> str:
        return self._inputs + self._next_token_text

    @property
    def next_token(self) -> int:
        return None if len(self._tokens) == 0 else self._tokens[-1]

    @property
    def attention_mask(self) -> torch.LongTensor:
        return self._mask

    @property
    def max_token(self) -> int:
        return self._generation_config.max_length


class NeuronGenerator(Generator):
    """A Generator for Neuron models."""

    def __init__(
        self,
        model: NeuronModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.special_tokens = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        self.slots = [Slot(i) for i in range(self.model.batch_size)]

    @property
    def info(self) -> InfoResponse:
        """Returns the expected InfoResponse."""
        dtype = getattr(self.model.config, "torch_dtype", "float32")
        return InfoResponse(
            requires_padding=True,
            dtype=str(dtype),
            device_type="xla",
        )

    def warmup(self, batch: Batch) -> int:
        """Verify if the hardware can support the target load.

        Args:
            batch (`Batch`):
                A batch corresponding to the maximum number of concurrent requests.

        Return:
            The maximum number of tokens the model supports.
        """
        self.prefill(batch)
        return self.model.batch_size * self.model.max_length

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill new requests.

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        slots = {state: [] for state in Slot.State}
        for slot in self.slots:
            slots[slot.state].append(slot)
        active_slots = slots[Slot.State.READY]
        empty_slots = slots[Slot.State.EMPTY]
        if len(empty_slots) < len(batch.requests):
            raise ValueError(
                f"Cannot prefill {len(batch.requests)} new request(s) with only {len(empty_slots)} empty slots."
                f"Please align the number of concurrent requests with the static batch size: {self.model.batch_size}."
            )
        # Assign each request to an empty slot
        logger.debug(f"Prefilling {len(batch.requests)} new request(s) with {len(empty_slots)} empty slot(s)")
        for request in batch.requests:
            slot = empty_slots.pop()
            slot.assign(request, self.model.generation_config)
            logger.debug(f"Request {slot.request_id} assigned to slot {slot.id}")
        # Reconstruct the full inputs (without padding)
        inputs = [slot.inputs for slot in self.slots]
        # Tokenize with padding
        padded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        #  If needed truncate sequences to fit into the static dimensions
        seq_length = min(padded_inputs.input_ids.shape[-1], self.model.max_length)
        input_ids = padded_inputs.input_ids[:, :seq_length]
        attention_mask = padded_inputs.attention_mask[:, :seq_length]
        # Each slot must be reset with the padded inputs and masks
        for i, slot in enumerate(self.slots):
            if slot.state != slot.state.EMPTY:
                slot_input_ids = input_ids[i : i + 1, :]
                # Padded input ids are also required to set logits processors and stopping criterias
                selector = TokenSelector.create(
                    slot_input_ids, slot.generation_config, self.model, self.model.max_length
                )
                slot_input_ids = slot_input_ids.squeeze().type(torch.int64)
                slot_attention_mask = attention_mask[i]
                slot.reset(slot_input_ids, slot_attention_mask, selector)
        # Clear KV cache
        self.model.reset_generation()
        # Pause previously active slots during generation.
        # Their KV cache will be prefilled but new tokens will be ignored, as they
        # have already been generated and sent back in the last decode.
        for slot in active_slots:
            slot.pause()
        generation, next_batch = self._generate_token(batch.id, input_ids, attention_mask)
        # Reactivate previously active slots for the next decode.
        for slot in active_slots:
            slot.resume()
        logger.debug("Model ready for decoding")
        return generation, next_batch

    def decode(self, batches: List[CachedBatch]) -> Tuple[List[Generation], CachedBatch]:
        """Decode the specified prefilled requests.

        Args:
            batches (`List[CachedBatch]`):
                A list of previous batches containing the prefilled requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        # batches contains a list composed of:
        # - the batch id returned by the last decode,
        # - the batch id(s) returned by the last prefill(s)
        # Batches are always concatenated during prefill, so we can
        # just carry on with decoding. We adopt the id of the first
        # batch in the list as our next batch id.
        next_batch_id = batches[0].id
        # Reconstruct input_ids and attention_mask from slots
        input_ids = None
        attention_mask = None
        for i, slot in enumerate(self.slots):
            if slot.state != Slot.State.EMPTY:
                if input_ids is None:
                    # Create blank inputs covering all slots (even empty ones)
                    input_ids = torch.full(
                        [self.model.batch_size, 1], fill_value=self.tokenizer.eos_token_id, dtype=torch.int64
                    )
                # input_ids are simply the tokens generated by the last decode or prefill requests (other tokens are cached)
                input_ids[i, 0] = slot.next_token
                if attention_mask is None:
                    # Create default mask covering all slots (even empty ones)
                    attention_mask = torch.zeros(
                        [self.model.batch_size, slot.attention_mask.size(-1)], dtype=torch.int64
                    )
                    attention_mask[:, -1] = 1
                attention_mask[i, :] = slot.attention_mask
        if input_ids is None:
            raise ValueError("Unable to decode tokens for non-prefilled batches (probably due to a previous failure)")
        return self._generate_token(next_batch_id, input_ids, attention_mask)

    def _generate_token(
        self, next_batch_id: int, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[List[Generation], CachedBatch]:
        model_inputs = self.model.prepare_inputs_for_generation(input_ids, attention_mask)
        outputs = self.model(
            **model_inputs,
            return_dict=True,
        )
        generations = []
        request_ids = []
        active_slots = False
        for i, slot in enumerate(self.slots):
            if slot.state != Slot.State.READY:
                continue
            request_id = slot.request_id
            request_ids.append(request_id)
            next_token_logits = outputs.logits[i : i + 1, -1, :]
            slot_input_ids = input_ids[i : i + 1, :]
            next_token = slot.select(slot_input_ids, next_token_logits)
            next_token_text = self.tokenizer.decode(next_token)
            if not slot.generated_text.endswith(" ") and not next_token_text.startswith(" "):
                # Some tokenizers do not prepend spaces automatically when decoding a single token
                contextual_text = self.tokenizer.decode([slot.next_token, next_token])
                if contextual_text[: -len(next_token_text)].endswith(" "):
                    next_token_text = " " + next_token_text
            slot.append(next_token, next_token_text)
            generated_text = None
            finish_reason = None
            if next_token == self.tokenizer.eos_token_id:
                finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
            elif slot.stopped:
                finish_reason = FinishReason.FINISH_REASON_STOP_SEQUENCE
            if finish_reason is not None:
                # We must include the generated text for each finished sequence in the response
                generated_text = GeneratedText(
                    text=slot.generated_text, generated_tokens=slot.generated_tokens, finish_reason=finish_reason
                )
                logger.debug(f"Finished generating tokens for request {request_id}")
                # mark the slot as available
                slot.clear()
            else:
                active_slots = True
            generations.append(
                Generation(
                    request_id=request_id,
                    prefill_tokens=None,
                    token_id=next_token,
                    token_logprob=None,
                    token_text=next_token_text,
                    token_is_special=(next_token in [self.special_tokens]),
                    generated_text=generated_text,
                )
            )
        batch = None
        if active_slots:
            # Whatever initial batch these requests came from, we always return all pending requests in a single batch
            batch = self._cached_batch(next_batch_id, request_ids)
        else:
            logger.debug("No more pending requests")
        return generations, batch

    def _cached_batch(self, batch_id: int, request_ids: List):
        size = len(request_ids)
        max_tokens = size * self.model.max_length
        return CachedBatch(id=batch_id, request_ids=request_ids, size=size, max_tokens=max_tokens)

    def filter(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        """Remove requests that are not listed from the specified batch

        Args:
            batch_id (`int`):
                The id of a cached batch.
            request_ids(`List[int]`):
                The list of requests that must be kept.

        Return:
            A `CachedBatch` containing the pending requests.
        """
        self._clear(request_ids)
        return self._cached_batch(batch_id, request_ids)

    def clear(self):
        """Remove all requests from the generator"""
        return self._clear([])

    def _clear(self, request_ids: List):
        for slot in self.slots:
            if slot.state != Slot.State.EMPTY and slot.request_id not in request_ids:
                logger.debug(f"Removing request {slot.request_id}")
                slot.clear()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str],
    ):
        """Instantiate a NeuronGenerator.

        Args:
            model_id (`str`):
                The *model_id* of a model on the HuggingFace hub or the path to a local model.
                In either case, the hub or local path must also contain a Tokenizer.
            revision (`str`):
                The revision of the model on the HuggingFace hub.

        Returns:
            A NeuronGenerator.
        """
        model_path = fetch_model(model_id, revision)
        model = NeuronModelForCausalLM.from_pretrained(model_path, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return cls(model, tokenizer)
