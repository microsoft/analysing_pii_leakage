from dataclasses import dataclass, field


@dataclass
class NERArgs:
    """ This class encapsulates all parameters for named entity recognition (NER). """
    CONFIG_KEY = "ner_args"

    ner: str = field(default='flair', metadata={
        "help": "the framework to use.",
        "choices": ['flair']
    })

    ner_model: str = field(default="flair/ner-english-ontonotes-large", metadata={
        "help": "the NER model to use",
        "choices": ["flair/ner-english-ontonotes-large"]
    })

    anon_token: str = field(default="<MASK>", metadata={
        "help": "anonymization for PII"
    })

    tag_n_batches: int = field(default=-1, metadata={
        "help": "stop tagging PII after this many batches."
    })

    anonymize: bool = field(default=False, metadata={
        "help": "whether to anonymize PII"
    })


