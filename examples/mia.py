import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.ner_args import NERArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.privacy_args import PrivacyArgs
from src.arguments.trainer_args import TrainerArgs
from src.models.model_factory import ModelFactory


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            PrivacyArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def mia(model_args: ModelArgs,
        dataset_args: DatasetArgs,
        env_args: EnvArgs,
        config_args: ConfigArgs):
    """ This script performs a classical membership inference attack (MIA) on a language model (LM).
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        env_args = config_args.get_env_args()

    lm = ModelFactory.from_model_args(model_args, env_args=env_args).load()



# ----------------------------------------------------------------------------
if __name__ == "__main__":
    mia(*parse_args())
# ----------------------------------------------------------------------------
