from typing import List, Tuple

import numpy as np

from src.attacks.mia.base_signal import BaseEntityInferenceSignal
from src.models.language_model import LanguageModel


class PPLEntityInferenceSignal(BaseEntityInferenceSignal):

    def __init__(self, target_model: LanguageModel):
        super().__init__(target_model)

    @staticmethod
    def pick_best(candidates: List[str], scores: List[float]) -> Tuple[str, float]:
        """ Sorts so that the 'best' value is at the beginning
        """
        idx = np.asarray(scores).argsort()  # lowest score wins
        return candidates[idx[0]], scores[idx[0]]

    def get_signal(self, sample: str, candidates: List[str], mask_token='[MASK]') -> List[float]:
        """ Expects a sentence with a [MASK] in place of the PII that should be predicted.
        """
        scores = []
        for candidate in candidates:
            new_sample = sample.replace(mask_token, candidate)

            offset = len(self.target_model.tokenizer.encode(sample[:sample.index(mask_token)-1]))
            scores += [self.target_model.perplexity(new_sample, offset=offset, verbose=False)]
        return scores