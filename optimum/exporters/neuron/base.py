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

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from ...exporters.base import ExportConfig
from ...neuron.utils import is_neuron_available, is_transformers_neuronx_available
from ...utils import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from optimum.utils import DummyInputGenerator


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MissingMandatoryAxisDimension(ValueError):
    pass


class NeuronConfig(ExportConfig):
    """Base class for Neuron exportable models

    Class attributes:

    - INPUT_ARGS (`Tuple[Union[str, Tuple[Union[str, Tuple[str]]]]]`) -- A tuple where each element is either:
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
    def get_input_args_for_task(cls, task: str) -> Tuple[str]:
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


class NeuronDefaultConfig(NeuronConfig, ABC):
    """
    Base class for configuring the export of Neuron TorchScript models.

    Class attributes:

    - NORMALIZED_CONFIG_CLASS (`Type`) -- A class derived from [`~optimum.utils.NormalizedConfig`] specifying how to
    normalize the model config.
    - DUMMY_INPUT_GENERATOR_CLASSES (`Tuple[Type]`) -- A tuple of classes derived from
    [`~optimum.utils.DummyInputGenerator`] specifying how to create dummy inputs.
    - ATOL_FOR_VALIDATION (`Union[float, Dict[str, float]]`) -- A float or a dictionary mapping task names to float,
    where the float values represent the absolute tolerance value to use during model conversion validation.
    - INPUT_ARGS (`Tuple[Union[str, Tuple[Union[str, Tuple[str]]]]]`) -- A tuple where each element is either:
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
    ATOL_FOR_VALIDATION: Union[float, Dict[str, float]] = 1e-5
    MODEL_TYPE = None

    _TASK_TO_COMMON_OUTPUTS = {
        "feature-extraction": ["last_hidden_state", "pooler_output"],
        "fill-mask": ["logits"],
        "image-classification": ["logits"],
        "image-segmentation": ["logits", "pred_boxes", "pred_masks"],
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
        compiler_type: Optional[str] = None,
        compiler_version: Optional[str] = None,
        batch_size: Optional[int] = None,
        dynamic_batch_size: bool = False,
        sequence_length: Optional[int] = None,
        num_choices: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_channels: Optional[int] = None,
        feature_size: Optional[int] = None,
        nb_max_frames: Optional[int] = None,
        audio_sequence_length: Optional[int] = None,
        point_batch_size: Optional[int] = None,
        nb_points_per_image: Optional[int] = None,
        num_beams: int = 1,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # TODO: add custom dtype after optimum 1.13 release
        # int_dtype: str = "int64",
        # float_dtype: str = "fp32",
    ):
        self._config = config
        self._normalized_config = self.NORMALIZED_CONFIG_CLASS(self._config)
        self.mandatory_axes = ()
        self.task = task
        self._axes: Dict[str, int] = {}
        self.dynamic_batch_size = dynamic_batch_size

        if self.dynamic_batch_size is True and is_neuron_available():
            logger.info("Overwriting batch size to 1 for neuron dynamic batch size support.")
            batch_size = 1

        # To avoid using **kwargs.
        axes_values = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "num_choices": num_choices,
            "width": width,
            "height": height,
            "num_channels": num_channels,
            "feature_size": feature_size,
            "nb_max_frames": nb_max_frames,
            "audio_sequence_length": audio_sequence_length,
            "point_batch_size": point_batch_size,
            "nb_points_per_image": nb_points_per_image,
            "num_beams": num_beams,
        }
        input_shapes = {}
        for name, value in axes_values.items():
            if value is not None:
                input_shapes[name] = value
            setattr(self, name, value)
        setattr(self, "input_shapes", input_shapes)
        setattr(self, "output_attentions", output_attentions)
        setattr(self, "output_hidden_states", output_hidden_states)
        setattr(self, "compiler_type", compiler_type)
        setattr(self, "compiler_version", compiler_version)

    @classmethod
    def get_mandatory_axes_for_task(cls, task: str) -> Tuple[str]:
        return cls.get_input_args_for_task(task)

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, value: str):
        self._task = value
        self.mandatory_axes = self.get_mandatory_axes_for_task(self.task)

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

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        for name, axis_dim in self._axes.items():
            self._axes[name] = kwargs.pop(name, axis_dim)

        self._validate_mandatory_axes()
        return [cls_(self.task, self._normalized_config, **self._axes) for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES]

    @property
    def values_override(self) -> Optional[Dict[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting.

        Returns:
            `Optional[Dict[str, Any]]`: A dictionary specifying the configuration items to override.
        """

        return None

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        """
        List containing the names of the inputs the exported model should take.

        Returns:
            `List[str]`: A list of input names.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> List[str]:
        """
        List containing the names of the outputs the exported model should have.

        Returns:
            `List[str]`: A list of output names.
        """
        return self._TASK_TO_COMMON_OUTPUTS[self.task]

    def generate_dummy_inputs(
        self, return_tuple: bool = False, **kwargs
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Generates dummy inputs that the exported model should be able to process.
        This method is actually used to determine the input specs and their static shapes that are needed for the export.

        Returns:
            `Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]`: A dictionary mapping input names to dummy tensors or a tuple with dummy tensors.
        """
        dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)
        dummy_inputs = {}

        for input_name in self.inputs:
            input_was_inserted = False
            for dummy_input_gen in dummy_inputs_generators:
                if dummy_input_gen.supports_input(input_name):
                    dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework="pt")
                    # TODO: add custom dtype after optimum 1.13 release
                    # dummy_inputs[input_name] = dummy_input_gen.generate(input_name, framework="pt", int_dtype=self.int_dtype, float_dtype=self.float_dtype)
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
    def flatten_inputs(cls, inputs: Dict[str, Any]) -> Dict[str, Any]:
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

    def patch_model_for_export(
        self,
        model: "PreTrainedModel",
        dummy_inputs: Optional[Dict[str, torch.Tensor]] = None,
        forward_with_tuple: bool = False,
        eligible_outputs: Optional[List[Union[str, int]]] = None,
    ):
        """
        Checks if inputs order of the model's forward pass correspond to the generated dummy inputs to ensure the dummy inputs tuple used for
        tracing are under the correct order.
        """

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model: "PreTrainedModel", input_names: List[str]):
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
                    outputs = self.model(**ordered_inputs)

                if isinstance(outputs, dict) and eligible_outputs is not None:
                    outputs = {name: outputs[name] for name in outputs.keys() & eligible_outputs}

                if isinstance(outputs, tuple) and eligible_outputs is not None:
                    if not all(isinstance(x, int) for x in eligible_outputs):
                        raise ValueError(
                            "To extract outputs from a tuple, `eligible_outputs` must be a list of integers only."
                        )
                    outputs = [outputs[i] for i in eligible_outputs]

                return outputs

        return ModelWrapper(model, list(dummy_inputs.keys()))


class NeuronDecoderConfig(NeuronConfig):
    """
    Base class for configuring the export of Neuron Decoder models

    Class attributes:

    - INPUT_ARGS (`Tuple[Union[str, Tuple[Union[str, Tuple[str]]]]]`) -- A tuple where each element is either:
        - An argument  name, for instance "batch_size" or "sequence_length", that indicates that the argument can
        be passed to export the model,
    - NEURONX_CLASS (`str`) -- the name of the transformers-neuronx class to instantiate for the model.
    It is a full class name defined relatively to the transformers-neuronx module, e.g. `gpt2.model.GPT2ForSampling`
    [`~optimum.utils.DummyInputGenerator`] specifying how to create dummy inputs.

    The NEURONX_CLASS must always be defined in each model configuration.

    Args:
        task (`str`): The task the model should be exported for.
    """

    INPUT_ARGS = ("batch_size", "sequence_length")
    NEURONX_CLASS = None

    def __init__(self, task: str):
        if not is_transformers_neuronx_available():
            raise ModuleNotFoundError(
                "The mandatory transformers-neuronx package is missing. Please install optimum[neuronx]."
            )
        module_name, class_name = self.NEURONX_CLASS.rsplit(".", maxsplit=1)
        module = importlib.import_module(f"transformers_neuronx.{module_name}")
        self._neuronx_class = getattr(module, class_name, None)
        if self._neuronx_class is None:
            raise ImportError(f"{class_name} not found in {module_name}. Please check transformers-neuronx version.")

    @property
    def neuronx_class(self):
        return self._neuronx_class
