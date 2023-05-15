from src.arguments.attack_args import AttackArgs
from src.arguments.env_args import EnvArgs
from src.arguments.ner_args import NERArgs
from src.attacks.extraction.naive_extraction import NaiveExtractionAttack
from src.attacks.privacy_attack import PrivacyAttack


class AttackFactory:
    @staticmethod
    def from_attack_args(attack_args: AttackArgs, ner_args: NERArgs = None, env_args: EnvArgs = None) -> PrivacyAttack:
        if attack_args.attack_name == "naive_extraction":
            return NaiveExtractionAttack(attack_args=attack_args, ner_args=ner_args, env_args=env_args)
        else:
            raise ValueError(attack_args.attack_name)
