# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import unittest
from unittest.mock import Mock, patch

from optimum.neuron.models.training import AutoModel, AutoModelForCausalLM


class TestAutoModelClasses(unittest.TestCase):
    """Test cases for AutoModel and AutoModelForCausalLM classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.llama_model_name = "meta-llama/Llama-2-7b-hf"
        self.granite_model_name = "ibm-granite/granite-8b-code-instruct"
        self.qwen_model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.unsupported_model_name = "microsoft/DialoGPT-medium"

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_for_causal_lm_llama(self, mock_auto_config):
        """Test AutoModelForCausalLM with Llama model."""
        # Mock config
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock the model class
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.from_pretrained.return_value = mock_model_instance
            mock_get_model_class.return_value = mock_model_class

            # Test loading
            result = AutoModelForCausalLM.from_pretrained(self.llama_model_name)

            # Verify calls
            mock_auto_config.from_pretrained.assert_called_once_with(self.llama_model_name)
            mock_get_model_class.assert_called_once_with("llama", "causal-lm", "training")
            mock_model_class.from_pretrained.assert_called_once_with(self.llama_model_name)

            # Verify result
            self.assertEqual(result, mock_model_instance)

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_for_causal_lm_granite(self, mock_auto_config):
        """Test AutoModelForCausalLM with Granite model."""
        # Mock config
        mock_config = Mock()
        mock_config.model_type = "granite"
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock the model class
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.from_pretrained.return_value = mock_model_instance
            mock_get_model_class.return_value = mock_model_class

            # Test loading
            result = AutoModelForCausalLM.from_pretrained(self.granite_model_name)

            # Verify calls
            mock_auto_config.from_pretrained.assert_called_once_with(self.granite_model_name)
            mock_get_model_class.assert_called_once_with("granite", "causal-lm", "training")
            mock_model_class.from_pretrained.assert_called_once_with(self.granite_model_name)

            # Verify result
            self.assertEqual(result, mock_model_instance)

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_for_causal_lm_qwen(self, mock_auto_config):
        """Test AutoModelForCausalLM with Qwen model."""
        # Mock config
        mock_config = Mock()
        mock_config.model_type = "qwen2.5"
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock the model class
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.from_pretrained.return_value = mock_model_instance
            mock_get_model_class.return_value = mock_model_class

            # Test loading
            result = AutoModelForCausalLM.from_pretrained(self.qwen_model_name)

            # Verify calls
            mock_auto_config.from_pretrained.assert_called_once_with(self.qwen_model_name)
            mock_get_model_class.assert_called_once_with("qwen2.5", "causal-lm", "training")
            mock_model_class.from_pretrained.assert_called_once_with(self.qwen_model_name)

            # Verify result
            self.assertEqual(result, mock_model_instance)

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_for_causal_lm_unsupported_model(self, mock_auto_config):
        """Test AutoModelForCausalLM with unsupported model type."""
        # Mock config for unsupported model
        mock_config = Mock()
        mock_config.model_type = "gpt2"  # Unsupported model type
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock get_neuron_model_class to raise ValueError
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_get_model_class.side_effect = ValueError(
                "Model type gpt2 is not supported for task causal-lm in neuron in training mode. "
                "Supported types are: ['granite', 'llama', 'qwen2.5']."
            )

            # Test that ValueError is raised
            with self.assertRaises(ValueError) as context:
                AutoModelForCausalLM.from_pretrained(self.unsupported_model_name)

            # Verify error message
            self.assertIn("Model type gpt2 is not supported", str(context.exception))
            self.assertIn("causal-lm", str(context.exception))
            self.assertIn("training mode", str(context.exception))

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_for_causal_lm_with_kwargs(self, mock_auto_config):
        """Test AutoModelForCausalLM with additional kwargs."""
        # Mock config
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock the model class
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.from_pretrained.return_value = mock_model_instance
            mock_get_model_class.return_value = mock_model_class

            # Test loading with kwargs
            kwargs = {"torch_dtype": "float16", "device_map": "auto"}
            result = AutoModelForCausalLM.from_pretrained(self.llama_model_name, **kwargs)

            # Verify calls
            mock_auto_config.from_pretrained.assert_called_once_with(self.llama_model_name, **kwargs)
            mock_get_model_class.assert_called_once_with("llama", "causal-lm", "training")
            mock_model_class.from_pretrained.assert_called_once_with(self.llama_model_name, **kwargs)

            # Verify result
            self.assertEqual(result, mock_model_instance)

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_llama(self, mock_auto_config):
        """Test AutoModel with Llama model."""
        # Mock config
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock the model class
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_model_class = Mock()
            mock_model_instance = Mock()
            mock_model_class.from_pretrained.return_value = mock_model_instance
            mock_get_model_class.return_value = mock_model_class

            # Test loading
            result = AutoModel.from_pretrained(self.llama_model_name)

            # Verify calls
            mock_auto_config.from_pretrained.assert_called_once_with(self.llama_model_name)
            mock_get_model_class.assert_called_once_with("llama", "model", "training")
            mock_model_class.from_pretrained.assert_called_once_with(self.llama_model_name)

            # Verify result
            self.assertEqual(result, mock_model_instance)

    @patch("optimum.neuron.models.training.auto.AutoConfig")
    def test_auto_model_unsupported_model(self, mock_auto_config):
        """Test AutoModel with unsupported model type."""
        # Mock config for unsupported model
        mock_config = Mock()
        mock_config.model_type = "bert"  # Unsupported for base model task
        mock_auto_config.from_pretrained.return_value = mock_config

        # Mock get_neuron_model_class to raise ValueError
        with patch("optimum.neuron.models.training.auto.get_neuron_model_class") as mock_get_model_class:
            mock_get_model_class.side_effect = ValueError(
                "Model type bert is not supported for task model in neuron in training mode. Supported types are: []."
            )

            # Test that ValueError is raised
            with self.assertRaises(ValueError) as context:
                AutoModel.from_pretrained("bert-base-uncased")

            # Verify error message
            self.assertIn("Model type bert is not supported", str(context.exception))
            self.assertIn("task model", str(context.exception))
            self.assertIn("training mode", str(context.exception))

    def test_auto_model_classes_are_classes(self):
        """Test that AutoModel classes are actual classes, not instances."""
        self.assertTrue(hasattr(AutoModel, "from_pretrained"))
        self.assertTrue(hasattr(AutoModelForCausalLM, "from_pretrained"))
        self.assertTrue(callable(AutoModel.from_pretrained))
        self.assertTrue(callable(AutoModelForCausalLM.from_pretrained))


if __name__ == "__main__":
    unittest.main()
