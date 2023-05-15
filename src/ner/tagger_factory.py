from src.arguments.env_args import EnvArgs
from src.arguments.ner_args import NERArgs
from src.ner.flair_tagger import FlairTagger
from src.ner.tagger import Tagger


class TaggerFactory:

    @staticmethod
    def from_ner_args(ner_args: NERArgs, env_args: EnvArgs = None) -> Tagger:
        if ner_args.ner == "flair":
            return FlairTagger(ner_args, env_args)
        else:
            raise ValueError(ner_args.ner)