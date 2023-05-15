import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AttackArgs:
    CONFIG_KEY = "attack_args"

    attack_name: str = field(default="naive_extraction", metadata={
        "help": "number of workers",
        "choices": ["perplexity_inference", "perplexity_reconstruction", "naive_extraction"]
    })

    target_sequence: str = field(default="", metadata={
        "help": "the sequence to be attacked for PII reconstruction & inference. "
                "Replace the PII with <T-MASK> and other PII with <MASK>. "
    })

    pii_candidates: List[str] = field(default_factory=lambda: [], metadata={
        "help": "PII candidates for a PII inference attack. Please ensure the casing is correct. "
    })
