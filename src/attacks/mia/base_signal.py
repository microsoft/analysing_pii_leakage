from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from src.models.language_model import LanguageModel


class BaseEntityInferenceSignal:
    INVALID_PREDICTION = np.inf

    def __init__(self, target_model: LanguageModel):
        self.target_model = target_model

    @staticmethod
    @abstractmethod
    def pick_best(candidates: List[str], scores: List[float]) -> Tuple[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_signal(self, sequence: str, entity: str):
        raise NotImplementedError