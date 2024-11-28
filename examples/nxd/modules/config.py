import os
import json
import torch
from dataclasses import dataclass


@dataclass
class NeuronInferenceConfig():
    """
    This class contains attributes that are needed for various inference
    optimization/features in NxD.

    """
    FILENAME = "neuron_config.json"

    tp_degree: int
    batch_size: int
    max_input_tokens: int
    max_total_tokens: int
    auto_cast_type: str
    is_continuous_batching: bool = False
    enable_bucketing: bool = True


    def save_pretrained(self, path) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.__dict__
        file_path = os.path.join(path, self.FILENAME)
        with open(file_path, 'w') as f:
            f.write(json.dumps(config_dict, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_pretrained(cls, path):
        file_path = os.path.join(path, cls.FILENAME)
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
