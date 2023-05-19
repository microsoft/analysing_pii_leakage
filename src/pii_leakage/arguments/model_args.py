# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    """ This class encapsulates all parameters for a language model. """
    CONFIG_KEY = "model_args"

    model_ckpt: str = field(default=None, metadata={
        "help": "path to the checkpoint of the model."
    })

    architecture: str = field(default="gpt2", metadata={
        "help": "the architecture of the model",
        "choices": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    })

    pre_trained: bool = field(default=True, metadata={
        "help": "True: load pre-trained, public weights of the model. If additionally, a checkpoint is provided,"
                "      we always load the checkpoint last."
                "False: Randomly initialize model."
    })

    tokenizer_use_fast: bool = field(default=True, metadata={
        "help": "whether to set the flag use_fast in the tokenizer loading function."
    })

    def hash(self, suffix=""):
        """ Compute a unique hash based on this dict"""
        return hashlib.sha256(repr({
            "checkpoint": self.model_ckpt,
            "pre_trained": self.pre_trained,
            "suffix": suffix
        }).encode('utf-8')).hexdigest()

