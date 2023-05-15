from src.arguments.attack_args import AttackArgs
from src.arguments.env_args import EnvArgs
from src.arguments.ner_args import NERArgs


class PrivacyAttack:

    def __init__(self, attack_args: AttackArgs, ner_args: NERArgs = None, env_args: EnvArgs = None):
        self.attack_args = attack_args
        self.env_args = env_args if env_args is not None else EnvArgs()
        self.ner_args = ner_args if ner_args is not None else NERArgs()

    def attack(self, *args, **kwargs):
        raise NotImplementedError
