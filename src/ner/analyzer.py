from typing import List

from src.arguments.env_args import EnvArgs
from src.arguments.ner_args import NERArgs
from src.ner.recognizer_result import RecognizerResult
from src.ner.tagger import Tagger
from src.ner.tagger_factory import TaggerFactory


class Analyzer:
    MIN_CHARS_PER_PII = 3   # PII with fewer characters are never considered PII

    def __init__(self, ner_args: NERArgs, env_args: EnvArgs = None):
        super().__init__()
        self.ner_args = ner_args
        self.env_args = env_args if env_args is None else EnvArgs()
        self.tagger: Tagger = TaggerFactory.from_ner_args(ner_args, env_args)

    def analyze(self, text: str) -> List[RecognizerResult]:
        """ Analyze a string for PII.
        """
        return [x for x in self.tagger.analyze(text) if len(x.mention) >= self.MIN_CHARS_PER_PII]

    def pseudonomize(self, text: str):
        """ Analyze a string for PII and pseudonomize strings.
        """
        raise NotImplementedError
