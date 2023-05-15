from dataclasses import dataclass
from typing import List

from src.utils import print_highlighted


@dataclass
class PIIInferenceResult:
    pii_real: str      # Ground-Truth EUII
    pii_pred: str      # Predicted EUII
    score: float       # score

    def match(self):
        return self.pii_pred == self.pii_real

@dataclass
class PIIInferenceResults:
    results: List[PIIInferenceResult]

    def __init__(self):
        super().__init__()
        self.results = []

    def append(self, result: PIIInferenceResult):
        self.results.append(result)

    def accuracy(self):
        return len([x for x in self.results if x.match()]) / len(self.results)

    def print_stats(self):
        if len(self.results) > 0:
            last = self.results[-1]
            if last.match():
                print_highlighted(f"Pred={last.pii_pred} Real={last.pii_real}, Total Acc: {self.accuracy():.2f}")
            else:
                print(f"Pred={last.pii_pred} Real={last.pii_real}, Total Acc: {self.accuracy():.2f}")

