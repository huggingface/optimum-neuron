from transformers import AutoConfig, PretrainedConfig


class NeuronInferenceConfig(PretrainedConfig):
    """
    Base config class for inference in NxD.

    This class contains attributes that are needed for various inference
    optimization/features in NxD.
    """

    def __init__(
        self, tp_degree: int = 1, batch_size: int = 1, seq_len: int = 128, padding_side: str = "right", **kwargs
    ) -> None:
        # Basic config for inference in NxD
        self.tp_degree = tp_degree
        self.batch_size = batch_size
        self.padding_side = padding_side
        self.n_positions = seq_len

        # fallback to seq_len is for compatibility with vllm
        self.max_context_length = kwargs.pop("max_context_length", seq_len)
        self.max_new_tokens = seq_len - self.max_context_length
        if self.max_new_tokens == 0:
            self.max_new_tokens = None
        self.max_length = seq_len

        # Continuous batching
        self.max_batch_size = kwargs.get("max_batch_size", batch_size)
        self.is_continuous_batching = kwargs.get("is_continuous_batching", False)

        # On-device sampling
        self.on_device_sampling = kwargs.get("on_device_sampling", False)

        # Bucketing
        self.enable_bucketing = kwargs.get("enable_bucketing", False)

        super().__init__(**kwargs)

    @classmethod
    def from_model_config(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(**config.to_dict(), **kwargs)
