# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neuron configuration base classes."""

import re
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import torch

from optimum.utils import logging

from ...exporters.base import ExportConfig
from ...neuron.utils import ImageEncoderArguments, InputShapesArguments, is_neuron_available


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from optimum.utils import DummyInputGenerator


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MissingMandatoryAxisDimension(ValueError):
    pass


class NeuronExportConfig(ExportConfig):
    """Base class for Neuron exportable models

    Class attributes:

    - INPUT_ARGS (`tuple[str | tuple[str | tuple[str]]]`) -- A tuple where each element is either:
        - An argument  name, for instance "batch_size" or "sequence_length", that indicates that the argument can
        be passed to export the model,
        - Or a tuple containing two elements:
            - The first one is either a string or a tuple of strings and specifies for which task(s) the argument is relevant
            - The second one is the argument name.

    Input arguments can be mandatory for some export types, as specified in child classes.

    Args:
        task (`str`):
            The task the model should be exported for.
    """

    INPUT_ARGS = ()

    @classmethod
    def get_input_args_for_task(cls, task: str) -> tuple[str]:
        axes = []
        for axis in cls.INPUT_ARGS:
            if isinstance(axis, tuple):
                tasks, name = axis
                if not isinstance(tasks, tuple):
                    tasks = (tasks,)
                if task not in tasks:
                    continue
            else:
                name = axis
            axes.append(name)
        return tuple(axes)


class NeuronDefaultConfig(NeuronExportConfig, ABC):
    """
    Base class for configuring the export of Neuron TorchScript models.

    Class attributes:

    - NORMALIZED_CONFIG_CLASS (`Type`) -- A class derived from [`~optimum.utils.NormalizedConfig`] specifying how to
    normalize the model config.
    - DUMMY_INPUT_GENERATOR_CLASSES (`tuple[Type]`) -- A tuple of classes derived from
    [`~optimum.utils.DummyInputGenerator`] specifying how to create dummy inputs.
    - ATOL_FOR_VALIDATION (`float | dict[str, float]`) -- A float or a dictionary mapping task names to float,
    where the float values represent the absolute tolerance value to use during model conversion validation.
    - INPUT_ARGS (`tuple[str | tuple[str | tuple[str]]]`) -- A tuple where each element is either:
        - An argument  name, for instance "batch_size" or "sequence_length", that indicates that the argument MUST
        be passed to export the model,
        - Or a tuple containing two elements:
            - The first one is either a string or a tuple of strings and specifies for which task(s) the argument must be passed
            - The second one is the argument name.

        For example: `INPUT_ARGS = ("batch_size", "sequence_length", ("multiple-choice", "num_choices"))` means that
        to export the model, the batch size and sequence length values always need to be specified, and that a value
        for the number of possible choices is needed when the task is multiple-choice.

    Args:
        config (`transformers.PretrainedConfig`):
            The model configuration.
        task (`str`, defaults to `"feature-extraction"`):
            The task the model should be exported for.
        dynamic_batch_size (`bool`, defaults to `False`):
            Whether the Neuron compiled model supports dynamic batch size.
        int_dtype (`str`, defaults to `"int64"`):
            The data type of integer tensors, could be ["int64", "int32", "int8"], default to "int64".
        float_dtype (`str`, defaults to `"fp32"`):
            The data type of float tensors, could be ["fp32", "fp16", "bf16"], default to "fp32".

        The rest of the arguments are used to specify the shape of the inputs the model can take.
        They are required or not depending on the model the `NeuronDefaultConfig` is designed for.
    """

    NORMALIZED_CONFIG_CLASS = None
    DUMMY_INPUT_GENERATOR_CLASSES = ()
    ATOL_FOR_VALIDATION: float | dict[str, float] = 1e-5
    MODEL_TYPE = None
    LIBRARY_NAME = "transformers"
    CUSTOM_MODEL_WRAPPER = None

    _TASK_TO_COMMON_OUTPUTS = {
        "depth-estimation": ["predicted_depth"],
        "feature-extraction": ["last_hidden_state", "pooler_output"],
        "fill-mask": ["logits"],
        "image-classification": ["logits"],
        "image-segmentation": ["logits"],
        "image-to-image": ["reconstruction"],
        "masked-im": ["logits"],
        "multiple-choice": ["logits"],
        "object-detection": ["logits", "pred_boxes"],
        "question-answering": ["start_logits", "end_logits"],
        "semantic-segmentation": ["logits"],
        "text-classification": ["logits"],
        "token-classification": ["logits"],
        "audio-classification": ["logits"],
        "audio-frame-classification": ["logits"],
        "automatic-speech-recognition": ["logits"],
        "audio-xvector": ["logits"],
    }

    def __init__(
        self,
        config: "PretrainedConfig",
        task: str,
        input_shapes: InputShapesArguments,
        preprocessors: list | None = None,
        compiler_type: str | None = None,
        compiler_version: str | None = None,
        tensor_parallel_size: int = 1,
        dynamic_batch_size: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        int_dtype: str | torch.dtype = "int64",  # Int dtype of dummy inputs used for tracing
        float_dtype: str | torch.dtype = "fp32",  # Float dtype of dummy inputs used for tracing
    ):
        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
        self.mandatory_axes = ()
        self.tensor_parallel_size = tensor_parallel_size
        self.task = task
        self._axes: dict[str, int] = {}
        self.dynamic_batch_size = dynamic_batch_size
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype

        if self.dynamic_batch_size is True and is_neuron_available():
            logger.info("Overwriting batch size to 1 for neuron dynamic batch size support.")
            batch_size = 1
        else:
            batch_size = input_shapes.batch_size

        if preprocessors:
            for preprocessor in preprocessors:
                if hasattr(preprocessor, "feature_extractor_type"):
                    input_shapes.nb_max_frames = input_shapes.nb_max_frames or getattr(
                        preprocessor, "nb_max_frames", None
                    )

        # To avoid using **kwargs.
        axes_values = {
            "batch_size": batch_size,
            "text_batch_size": input_shapes.text_batch_size,
            "image_batch_size": input_shapes.image_batch_size,
            "sequence_length": input_shapes.sequence_length,
            "num_choices": input_shapes.num_choices,
            "width": input_shapes.width,
            "height": input_shapes.height,
            "num_channels": input_shapes.num_channels or getattr(self._config, "num_channels", None),
            "feature_size": input_shapes.feature_size,
            "nb_max_frames": input_shapes.nb_max_frames,
            "audio_sequence_length": input_shapes.audio_sequence_length,
            "point_batch_size": input_shapes.point_batch_size,
            "nb_points_per_image": input_shapes.nb_points_per_image,
            "num_beams": input_shapes.num_beams,
            "image_size": input_shapes.image_size or getattr(self._config, "image_size", None),
            "patch_size": input_shapes.patch_size or getattr(self._config, "patch_size", None),
            "vae_scale_factor": input_shapes.vae_scale_factor,
            "encoder_hidden_size": input_shapes.encoder_hidden_size,
            "image_encoder_shapes": ImageEncoderArguments(
                sequence_length=getattr(input_shapes.image_encoder_shapes, "sequence_length", None),
                hidden_size=getattr(input_shapes.image_encoder_shapes, "hidden_size", None),
                projection_dim=getattr(input_shapes.image_encoder_shapes, "projection_dim", None),
            ),
            "rotary_axes_dim": input_shapes.rotary_axes_dim,
        }
        valid_input_shapes = {}
        for name, value in axes_values.items():
            if value is not None:
                is_empty_dataclass = is_dataclass(value) and all(
                    getattr(value, field.name) is None for field in fields(value)
                )
                if not is_empty_dataclass:
                    valid_input_shapes[name] = value
            setattr(self, name, value)
        setattr(self, "input_shapes", valid_input_shapes)
        setattr(self, "output_attentions", output_attentions)
        setattr(self, "output_hidden_states", output_hidden_states)
        setattr(self, "compiler_type", compiler_type)
        setattr(self, "compiler_version", compiler_version)

    @classmethod
    def get_mandatory_axes_for_task(cls, task: str) -> tuple[str]:
        return cls.get_input_args_for_task(task)

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, value: str):
        self._task = value
        self.mandatory_axes = self.get_mandatory_axes_for_task(self.task)

    @property
    def tensor_parallel_size(self) -> int:
        return self._tensor_parallel_size

    @tensor_parallel_size.setter
    def tensor_parallel_size(self, value: int):
        self._tensor_parallel_size = value

    def __getattr__(self, attr_name) -> Any:
        if attr_name != "_axes" and attr_name in self._axes:
            return self._axes[attr_name]
        else:
            raise AttributeError(attr_name)

    def __setattr__(self, name: str, value: Any) -> None:
        mandatory_axes = getattr(self, "mandatory_axes", [])
        if name in mandatory_axes:
            if value is None:
                if self._normalized_config.has_attribute(name):
                    value = getattr(self._normalized_config, name)
            self._axes[name] = value
        else:
            return super().__setattr__(name, value)

    def _validate_mandatory_axes(self, **kwargs):
        for name, axis_dim in self._axes.items():
            if axis_dim is None:
                raise MissingMandatoryAxisDimension(
                    f"The value for the {name} axis is missing, it is needed to perform the export to Neuron compiled model."
                )

    def _create_dummy_input_generator_classes(self, **kwargs) -> list["DummyInputGenerator"]:
        for name, axis_dim in self._axes.items():
            self._axes[name] = kwargs.pop(name, axis_dim)

        return [cls_(self.task, self._normalized_config, **self._axes) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]

    @property
    def values_override(self) -> dict[str, Any] | None:
        """
        Dictionary of keys to override in the model's config before exporting.

        Returns:
            `dict[str, Any] | None`: A dictionary specifying the configuration items to override.
        """

        return None

    @property
    @abstractmethod
    def inputs(self) -> list[str]:
        """
        List containing the names of the inputs the exported model should take.

        Returns:
            `list[str]`: A list of input names.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> list[str]:
        """
        List containing the names of the outputs the exported model should have.

        Returns:
            `list[str]`: A list of output names.
        """
        return self._TASK_TO_COMMON_OUTPUTS[self.task]

    def generate_dummy_inputs(
        self, return_tuple: bool = False, **kwargs
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor]:
        """
        Generates dummy inputs that the exported model should be able to process.
        This method is actually used to determine the input specs and their static shapes that are needed for the export.

        Returns:
            `dict[str, torch.Tensor] | tuple[torch.Tensor]`: A dictionary mapping input names to dummy tensors or a tuple with dummy tensors.
        """
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)
        dummy_inputs = {}

        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    # TODO: remove the mapper and use directly torch float dtype after the PR in Optimum makes its way to a release: https://github.com/huggingface/optimum/pull/2117
                    mapper = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
                    if isinstance(self.float_dtype, torch.dtype):
                        float_dtype = mapper[self.float_dtype]
                    else:
                        float_dtype = self.float_dtype
                    dummy_inputs[input_name] = dummy_input_gen.generate(
                        input_name, framework="pt", int_dtype=self.int_dtype, float_dtype=float_dtype
                    )
                    input_was_inserted = True
                    break
            if not input_was_inserted:
                raise RuntimeError(
                    f'Could not generate dummy inputs for "{input_name}". Try adding a proper dummy input generator '
                    "to the model Neuron config."
                )

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs

    @classmethod
    def flatten_inputs(cls, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten nested structure in dummy inputs, e.g `addition_embed_type` of unet model.
        """
        flatten = {}
        for name, value in inputs.items():
            if isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    flatten[sub_name] = sub_value
            else:
                flatten[name] = value
        return flatten

    @classmethod
    def unflatten_inputs(cls, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Re-construct inputs that have been flatten for tracing.
        """
        unflatten = {}
        to_group = {}
        for name, value in inputs.items():
            name_with_idx = re.findall(r"(.*?)_(\d+)", name)
            if len(name_with_idx) > 0:
                if name_with_idx[0][0] in to_group:
                    to_group[name_with_idx[0][0]].append((int(name_with_idx[0][1]), value))
                else:
                    to_group[name_with_idx[0][0]] = [(int(name_with_idx[0][1]), value)]
            else:
                unflatten[name] = value

        if to_group:
            for name, values in to_group.items():
                ordered = sorted(values, key=lambda x: x[0])
            unflatten[name] = tuple([item[1] for item in ordered])

        return unflatten

    def patch_model_and_prepare_aliases(
        self,
        model: "PreTrainedModel",
        input_names: list[str] = None,
        forward_with_tuple: bool = False,
        eligible_outputs: list[str | int] | None = None,
        device: str | None = None,
    ):
        """
        Patch the model and generate aliased for tracing.

        This function performs the following:
        1. Verifies that the input order of the model's `forward` method matches the structure
        of the generated dummy inputs. This ensures the dummy inputs tuple is correctly ordered
        for tracing.
        2. Applies model sharding if tensor parallelism is enabled (using `CUSTOM_MODEL_WRAPPER`).
        3. Prepares I/O aliases to identify specific input tensors as state tensors.
        These state tensors will remain on the device, helping to reduce host-device I/O overhead.
        """
        output_hidden_states = self.output_hidden_states

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model: "PreTrainedModel", input_names: list[str]):
                super().__init__()
                self.model = model
                self.input_names = input_names

            def forward(self, *input):
                if len(input) != len(self.input_names):
                    raise ValueError(
                        f"The model needs {len(self.input_names)} inputs: {self.input_names}."
                        f" But only {len(input)} inputs are passed."
                    )

                ordered_inputs = dict(zip(self.input_names, input))

                if forward_with_tuple is True:
                    outputs = self.model(*ordered_inputs.values())
                else:
                    if output_hidden_states:
                        ordered_inputs["output_hidden_states"] = True
                    outputs = self.model(**ordered_inputs)

                if isinstance(outputs, dict):
                    if eligible_outputs is not None:
                        outputs = {name: outputs[name] for name in outputs.keys() & eligible_outputs}

                if isinstance(outputs, tuple) and eligible_outputs is not None:
                    if not all(isinstance(x, int) for x in eligible_outputs):
                        raise ValueError(
                            "To extract outputs from a tuple, `eligible_outputs` must be a list of integers only."
                        )
                    outputs = [outputs[i] for i in eligible_outputs]

                return outputs

        if self.CUSTOM_MODEL_WRAPPER is None:
            # Order dummy input and build empty alias
            return ModelWrapper(model, input_names), {}
        else:
            return self.CUSTOM_MODEL_WRAPPER(model, input_names), {}
