from dataclasses import dataclass

from ..utils.import_utils import is_trl_available
from .training_args import NeuronTrainingArguments
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import GRPOConfig
else:

    @dataclass
    class GRPOConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"You need to install the `trl=={TRL_VERSION}` library to use the `NeuronGRPOConfig`.")


@dataclass
class NeuronGRPOConfig(NeuronTrainingArguments, GRPOConfig):
    def __post_init__(self):
        # remove SFT specific argument validaiton from NeuronTrainingArguments __post_init__
        self.do_eval = False
        self.eval_strategy = "no"
        self.eval_steps = None
        super().__post_init__()

        # TRL>=0.24 expects steps_per_generation to be defined
        try:
            steps_pg = getattr(self, "steps_per_generation", None)
        except Exception:
            steps_pg = None

        if steps_pg is None:
            mapped_value = getattr(self, "num_generations", 1)
            try:
                self.steps_per_generation = int(mapped_value)
            except Exception:
                self.steps_per_generation = 1