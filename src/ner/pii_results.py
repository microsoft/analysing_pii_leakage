import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class PII:
    text: str
    entity_class: str

    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None

    def lower(self):
        return self.text.lower()

    def match(self, other):
        return self.text.lower() == other.lower()

class PIIEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PII):
            return asdict(o)
        return super().default(o)

@dataclass
class ListPII:
    data: List[PII] = field(default_factory=lambda: [], metadata={"help": "list of PII"})

    def get_entity_classes(self) -> List[str]:
        return list(set([pii.entity_class for pii in self.data]))

    def get_by_entity_class(self, entity_class: str):
        return ListPII([pii for pii in self.data if pii.entity_class == entity_class])

    def group_by_class(self) -> dict[str, 'ListPII']:
        return {
            entity_class: ListPII([pii for pii in self.data if pii.entity_class == entity_class])
            for entity_class in self.get_entity_classes()
        }

    def dumps(self) -> str:
        return json.dumps(self.__dict__, cls=PIIEncoder)

    def sort(self, reverse=False):
        self.data.sort(key=lambda x: x.start, reverse=reverse)
        return self

    def __iter__(self):
        return self.data.__iter__()

@dataclass
class DatasetPII:
    data: dict[int, List[PII]] = field(default_factory=lambda: {}, metadata={"help": "data dict"})

    @staticmethod
    def load(path: str):
        if os.path.exists(path):
            return DatasetPII(**json.loads(path))
        return DatasetPII()

    def save(self, path: str) -> str:
        data = json.dumps(self.__dict__)
        with open(path, 'w') as f:
            json.dump(data, f)
        return data

    def flatten(self, entity_classes: List[str] = None):
        """ gets all PII mention for the entity classes (all if none is specified) """
        if entity_classes is not None:
            return [item for sublist in self.data.values() for item in sublist if item.entity_class in entity_classes]
        return [item for sublist in self.data.values() for item in sublist]

    def get_unique_pii(self, entity_classes: List[str] = None):
        """ gets all unique PII mentions of the entity classes (all if none is specified) """
        if entity_classes is not None:
            return [x for x in list(set(list(self.flatten()))) if x.entity_class in entity_classes]
        return list(set(list(self.flatten())))

    def get_pii_count(self, pii: PII):
        """ counts the number of times a PII occurs """
        return len([x for x in self.flatten() if pii.match(x)])

    def last_batch_idx(self):
        """ Gets the highest batch idx. """
        return max(list(self.data.keys()))

    def add_pii(self, idx: int, piis: List[PII]):
        """ Adds a list of PII to the idx. """
        self.data[idx] = self.data.setdefault(idx, []) + piis
