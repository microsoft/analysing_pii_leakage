from dataclasses import dataclass, field

import yaml

from src.arguments.attack_args import AttackArgs
from src.arguments.trainer_args import TrainerArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.ner_args import NERArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.privacy_args import PrivacyArgs
from src.arguments.sampling_args import SamplingArgs
from src.utils.output import print_warning


@dataclass
class ConfigArgs:

    config_path: str = field(default=None, metadata={
        "help": "path to the yaml configuration file (*.yml)"
    })

    def exists(self):
        return self.config_path is not None

    args_to_config = {  # specify the config keys to read in the *.yml file
        EnvArgs.CONFIG_KEY: EnvArgs(),
        DatasetArgs.CONFIG_KEY: DatasetArgs(),
        ModelArgs.CONFIG_KEY: ModelArgs(),
        OutdirArgs.CONFIG_KEY: OutdirArgs(),
        TrainerArgs.CONFIG_KEY: TrainerArgs(),
        SamplingArgs.CONFIG_KEY: SamplingArgs(),
        NERArgs.CONFIG_KEY: NERArgs(),
        PrivacyArgs.CONFIG_KEY: PrivacyArgs(),
        AttackArgs.CONFIG_KEY: AttackArgs()
    }

    def get_env_args(self) -> EnvArgs:
        return self.args_to_config[EnvArgs.CONFIG_KEY]

    def get_privacy_args(self) -> PrivacyArgs:
        return self.args_to_config[PrivacyArgs.CONFIG_KEY]

    def get_ner_args(self) -> NERArgs:
        return self.args_to_config[NERArgs.CONFIG_KEY]

    def get_dataset_args(self) -> DatasetArgs:
        return self.args_to_config[DatasetArgs.CONFIG_KEY]

    def get_model_args(self) -> ModelArgs:
        return self.args_to_config[ModelArgs.CONFIG_KEY]

    def get_sampling_args(self) -> SamplingArgs:
        return self.args_to_config[SamplingArgs.CONFIG_KEY]

    def get_outdir_args(self) -> OutdirArgs:
        return self.args_to_config[OutdirArgs.CONFIG_KEY]

    def get_attack_args(self) -> AttackArgs:
        return self.args_to_config[AttackArgs.CONFIG_KEY]

    def get_trainer_args(self) -> TrainerArgs:
        """ Gets the trainer args. Output directory is always from output dir"""
        trainer_args = self.args_to_config[TrainerArgs.CONFIG_KEY]

        if (trainer_args.output_dir is None) or (len(trainer_args.output_dir) == 0):
            # if not specified, create a new one
            outdir_args = self.get_outdir_args()
            trainer_args.output_dir = outdir_args.create_folder_name()
        return trainer_args

    def __post_init__(self):
        if self.config_path is None:
            return

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.keys = list(data.keys())

        # load arguments
        keys_not_found = []
        for entry, values in data.items():
            for key, value in values.items():
                if key not in self.args_to_config[entry].__dict__.keys():
                    keys_not_found += [(entry, key)]
                self.args_to_config[entry].__dict__[key] = value
        if len(keys_not_found) > 0:
            print_warning(f"Could not find these keys: {keys_not_found}. Make sure they exist.")

        for key, value in self.args_to_config.items():
            if hasattr(value, "__post_init__"):
                value.__post_init__()





