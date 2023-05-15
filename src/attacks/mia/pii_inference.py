from abc import abstractmethod
from typing import List, Tuple


class EUIIInferenceBase:
    def __init__(self, candidate_entities: List[str]):
        self.candidate_entities = candidate_entities    # search space

    @abstractmethod
    def infer(self, sequence: str, **kwargs) -> Tuple[str, float]:
        """ Given a sequence where the EUII to predict is masked with the string '[MASK]' (no quotation marks),
         predicts the most likely candidate """
        raise NotImplementedError