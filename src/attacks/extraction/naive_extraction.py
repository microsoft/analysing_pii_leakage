from src.arguments.ner_args import NERArgs
from src.arguments.sampling_args import SamplingArgs
from src.attacks.privacy_attack import PrivacyAttack
from src.models.language_model import LanguageModel, GeneratedTextList
from src.ner.tagger import Tagger
from src.ner.tagger_factory import TaggerFactory
from src.utils.output import print_highlighted


class NaiveExtractionAttack(PrivacyAttack):

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
        """ Generate PII from empty prompts and tag them. """
        sampling_args = SamplingArgs(N=100, seq_len=64, generate_verbose=True)

        generated_text: GeneratedTextList = lm.generate(sampling_args)

        tagger: Tagger = self._get_tagger()
        entities = tagger.analyze(str(generated_text))
        persons = entities.get_by_entity_class('PERSON')

        persons_mentions = [p.text for p in persons]
        print(f"Found the following PERSON entities: {persons_mentions}")
        return persons_mentions
