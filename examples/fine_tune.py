# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pprint import pprint

import transformers

from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.arguments.outdir_args import OutdirArgs
from pii_leakage.arguments.privacy_args import PrivacyArgs
from pii_leakage.arguments.sampling_args import SamplingArgs
from pii_leakage.arguments.trainer_args import TrainerArgs
from pii_leakage.dataset.real_dataset import RealDataset
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.utils.output import print_highlighted, print_dict_highlighted


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


def fine_tune(model_args: ModelArgs,
              ner_args: NERArgs,
              train_args: TrainerArgs,
              dataset_args: DatasetArgs,
              privacy_args: PrivacyArgs,
              outdir_args: OutdirArgs,
              env_args: EnvArgs,
              config_args: ConfigArgs):
    """ Fine-tunes a language model (LM) on some text dataset with/without privacy.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        train_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        privacy_args = config_args.get_privacy_args()
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(config_args.get_privacy_args()))

    # -- Load the datasets
    train_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("train"),
                                                                  ner_args=ner_args, env_args=env_args)
    eval_dataset: RealDataset = DatasetFactory.from_dataset_args(dataset_args.set_split("test"),
                                                                 ner_args=ner_args, env_args=env_args)

    # -- Load the LM
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load()

    # -- Print configuration
    output_folder = outdir_args.create_folder_name()

    print_highlighted(f"Saving LM to: {output_folder}. Train Size: {len(train_dataset)},"
                      f" Eval Size: {len(eval_dataset)}")
    print_highlighted(f"Train Sample: {train_dataset.shuffle().first()}")

    # -- Fine-tune the LM
    lm.fine_tune(train_dataset, eval_dataset, train_args, privacy_args)

    # -- Print using the LM
    pprint(lm.generate(SamplingArgs(N=1)))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    fine_tune(*parse_args())
# ----------------------------------------------------------------------------
