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