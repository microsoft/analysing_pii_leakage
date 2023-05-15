import datasets
from datasets import load_dataset

from src.extern.CustomBuilder import CustomBuilder

_DESCRIPTION = "A custom wrapper for the Enron E-mail dataset."
_TEXT = "text"

_URLS = {
    "url": "conceptofmind/pile_enron_emails"  # URL to the public repositry
}


class CustomEnron(datasets.GeneratorBasedBuilder):
    """ A custom wrapper for the enron dataset. For internal use only.  """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        CustomBuilder(name="raw", sample_duplication_rate=1, version=VERSION, description="unprotected, private data"),
        CustomBuilder(name="scrubbed", sample_duplication_rate=1, version=VERSION,
                      description="PII replaced with [UNK]"),
    ]
    DEFAULT_CONFIG_NAME = "raw"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = None

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),  # the sample's content
                "cardinal": datasets.Value("string"),  # cardinal value
                "ordinal": datasets.Value("string"),  # ordinal value
                "date": datasets.Value("string"),  # date value
                "event": datasets.Value("string"),  # event name
                "fac": datasets.Value("string"),  # building name
                "gpe": datasets.Value("string"),  # geo-political entity
                "language": datasets.Value("string"),  # language name
                "law": datasets.Value("string"),  # law name
                "money": datasets.Value("string"),  # money name
                "norp": datasets.Value("string"),  # affiliation
                "person": datasets.Value("string"),  # person name
                "loc": datasets.Value("string"),  # location
                "org": datasets.Value("string"),  # organization
                "percent": datasets.Value("string"),  # percentage
                "product": datasets.Value("string"),  # product
                "quantity": datasets.Value("string"),  # quantity value
                "time": datasets.Value("string"),  # time value
                "work_of_art": datasets.Value("string"),  # name of work of art
                "phone_number": datasets.Value("string"),  # presidio
                "email_address": datasets.Value("string"),  # presidio
                "url": datasets.Value("string"),  # presidio
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features
        )

    def _split_generators(self, dl_manager):
        self.df = load_dataset(_URLS["url"])["train"]["text"]
        total_length = len(self.df)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "start": 0,
                    "end": int(.15 * total_length)
                },
            ),
            datasets.SplitGenerator(  # use 10% of the training samples for test
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "start": int(.15 * total_length),
                    "end": int(.2 * total_length)
                },
            ),
            datasets.SplitGenerator(  # Use 45% of the data for training shadow models
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "start": int(.2 * total_length),
                    "end": int(0.35 * total_length)
                },
            ),
        ]

    def _generate_examples(self, split: str, start: int, end: int):
        analyzer = NERAnalyzer()
        engine = AnonymizerEngine()
        unique_identifier = int(len(self.df) * start)

        subset = self.df[start:end]
        print(
            f"Length of data: {len(subset)} (from {start} to {end}), Sample Duplication Rate: {self.config.sample_duplication_rate}")

        for i, text in enumerate(subset):
            results = analyzer.analyze(text)

            entities: dict = {e: [] for e in analyzer.get_entity_classes()}
            for result in results:
                entities[result.entity_class] = entities.setdefault(result.entity_class, []) + [
                    text[result.start:result.end]]

            if self.config.name == "scrubbed":
                text = engine.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators={
                        k: OperatorConfig("replace", {"new_value": f"[{k}]"}) for k, _ in entities.items()
                    },
                ).text

            if i == 0:
                print_highlighted(text)

            for _ in range(self.config.sample_duplication_rate):
                unique_identifier += 1
                yield f"{unique_identifier}", {
                    "text": text,
                    **{k.lower(): v for k, v in entities.items()}
                }
