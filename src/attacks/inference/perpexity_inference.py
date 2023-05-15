from typing import List

from src.arguments.ner_args import NERArgs
from src.arguments.sampling_args import SamplingArgs
from src.attacks.privacy_attack import PrivacyAttack
from src.models.language_model import GeneratedTextList, LanguageModel
from src.ner.pii_results import PII
from src.ner.tagger import Tagger
from src.ner.tagger_factory import TaggerFactory
from src.utils.output import print_highlighted


class PerplexityInferenceAttack(PrivacyAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tagger = None

    def _get_tagger(self):
        if self._tagger is None:
            print_highlighted("Loading tagger...")
            ner_args = NERArgs(ner="flair", ner_model="flair/ner-english-ontonotes-large")
            self._tagger = TaggerFactory.from_ner_args(ner_args, env_args=self.env_args)
        return self._tagger

    def attack(self, lm: LanguageModel, *args, **kwargs):
        """ Generate PII from empty prompts and tag them.
        We assume the masked sequence uses <T-MASK> to encode the target mask (the one that should be
         inferred) and <MASK> to encode non-target masks.
        """
        masked_sequence: str = self.attack_args.target_sequence
        #ensure only one tmask is there
        assert masked_sequence.count("<T-MASK>") == 1, "Please use one <T-MASK> to encode the target mask."
        candidates: List[PII] = [PII(text=x, entity_class='PERSON') for x in self.attack_args.pii_candidates]

        # 1.) Chunk into prefix & suffix
        prefix, suffix = masked_sequence.split("<T-MASK>")

        # 2.) Impute any missing <MASK> tokens
        # ToDo

        # 3.) Remember persons from the query
        tagger: Tagger = self._get_tagger()
        query_entities = tagger.analyze(str(masked_sequence))
        query_persons = [p.text for p in query_entities.get_by_entity_class('PERSON')]

        # 3.) Sample candidates
        sampling_args = SamplingArgs(N=32, seq_len=32, generate_verbose=True, prompt=prefix)
        generated_text: GeneratedTextList = lm.generate(sampling_args)
        entities = tagger.analyze((str(generated_text)))
        candidates = [p.text for p in entities.get_by_entity_class('PERSON')]

        # 4.) Compute the lowest perplexity candidates
        # ToDo

