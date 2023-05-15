from dataclasses import dataclass
from typing import List

from src.utils import print_highlighted


@dataclass
class ReconstructionResult:
    pii_pred: str
    pii_real: str

    def match(self):
        return self.pii_pred == self.pii_real

@dataclass
class ReconstructionResults:
    results: List[ReconstructionResult]

    def __init__(self):
        self.results = []

    def append(self, result: ReconstructionResult):
        self.results.append(result)

    def print_stats(self):
        if len(self.results) > 0:
            last = self.results[-1]
            if last.match():
                print_highlighted(f"Pred={last.pii_pred} Real={last.pii_real}, Total Acc: {self.accuracy():.2f}")
            else:
                print(f"Pred={last.pii_pred} Real={last.pii_real}, Total Acc: {self.accuracy():.4f}")

    def accuracy(self):
        return len([x for x in self.results if x.match()]) / len(self.results)