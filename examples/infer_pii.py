from pprint import pprint

import transformers

from src.arguments.attack_args import AttackArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.ner_args import NERArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.privacy_args import PrivacyArgs
from src.arguments.sampling_args import SamplingArgs
from src.arguments.trainer_args import TrainerArgs
from src.attacks.attack_factory import AttackFactory
from src.attacks.extraction.naive_extraction import NaiveExtractionAttack
from src.attacks.inference.perpexity_inference import PerplexityInferenceAttack
from src.dataset.real_dataset import RealDataset
from src.models.language_model import LanguageModel
from src.models.model_factory import ModelFactory
from src.dataset.dataset_factory import DatasetFactory
from src.utils.output import print_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,    # used to load the model
                                            NERArgs,      # used to load the named entity recognizer
                                            AttackArgs,   # used to load the attack
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def infer_pii(model_args: ModelArgs,
                ner_args: NERArgs,
                attack_args: AttackArgs,
                env_args: EnvArgs,
                config_args: ConfigArgs):
    """ Given a masked sentence where <T-MASK> is the target mask (that should be inferred) and
    <MASK> is a mask for any other PII, this function infers the most likely candidate replacement
    for the target mask.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        ner_args = config_args.get_ner_args()
        env_args = config_args.get_env_args()

    # -- Load the LM
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)
    lm.print_sample("On 8 May 2003")

    attack: PerplexityInferenceAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    attack.attack(lm, verbose=True)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    infer_pii(*parse_args())
# ----------------------------------------------------------------------------
