import os.path
from abc import abstractmethod, ABCMeta, ABC
from typing import List

import evaluate
from tqdm import tqdm

from src.arguments.sampling_args import SamplingArgs
from src.dataset.generated_dataset import GeneratedDatasetWrapper
from src.dataset.utils import load_json_if_exists, save_json
from src.models.language_model import LanguageModel

import hashlib

class MIASignal:
    CACHE_PATH = os.path.expanduser("~/.cache/") # root path where all the cache files are stored.

    def __init__(self, target_model: LanguageModel, save_cache_every: int = 100, enable_caching: bool = True,
                 **kwargs):
        self.target_model = target_model
        self.save_cache_every = save_cache_every
        self.enable_caching = enable_caching
        self.cache = {}

    def get_target_model(self) -> LanguageModel:
        return self.target_model

    def get_name(self):
        return "MIA"

    @abstractmethod
    def measure(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_cache_fp(self) -> str:
        raise NotImplementedError

    def load_cache(self) -> dict:
        self.cache = load_json_if_exists(os.path.join(self.CACHE_PATH, self.get_cache_fp()), nexists_ok=True, verbose=True)
        return self.cache

    def save_cache(self):
        os.makedirs(self.CACHE_PATH, exist_ok=True)
        if len(self.cache) > 0:
            save_json(self.cache, os.path.join(self.CACHE_PATH, self.get_cache_fp()))

    @staticmethod
    def hash_element(x: str, bitlen: int = 64):
        return hashlib.md5(x.encode('utf-8')).hexdigest()[:bitlen]

    @abstractmethod
    def add_elem_to_cache(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_elem_from_cache(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def larger_is_better():
        return True

class MIASignalNoCache(MIASignal, ABC):
    def __init__(self, target_model: LanguageModel, **kwargs):
        if 'enable_caching' in kwargs:
            kwargs.pop('enable_caching')    # never cache PPL
        super().__init__(target_model, enable_caching=False, **kwargs)

    def get_cache_fp(self) -> str:
        pass

    def add_elem_to_cache(self, **kwargs):
        pass

    def get_elem_from_cache(self, **kwargs):
        pass


class PairedMIASignal(MIASignal, metaclass=ABCMeta):
    def add_elem_to_cache(self, candidate: str, reference: List[str], score):
        if self.enable_caching:
            candidate_key = self.hash_element(candidate)
            reference_key = self.hash_element("".join(reference))
            self.cache.setdefault(candidate_key, {}).setdefault(reference_key, score)

    def get_elem_from_cache(self, candidate: str, reference: List[str]):
        if not self.enable_caching:
            return None
        candidate_key = self.hash_element(candidate)
        elem: dict = self.cache.get(candidate_key)
        return elem if elem is None else elem.get(self.hash_element("".join(reference)))


class FrugalScoreSignal(PairedMIASignal):
    def __init__(self, target_model: LanguageModel, sampling_args: SamplingArgs, N: int = 128, **kwargs):
        super().__init__(target_model, **kwargs)
        self.N = N
        self.gen_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)
        self.frugalscore = evaluate._load("frugalscore", "moussaKam/frugalscore_medium_bert-base_mover-score", cache_dir=os.path.expanduser("~/.cache/my_cache"))

    def get_cache_fp(self) -> str:
        return f"frugalscore_{self.target_model.get_name()}_ckpt={self.target_model.get_checkpoint()}.json"

    def __str__(self):
        return f"FrugalScore, N={self.N}, caching={self.enable_caching}, Model={self.target_model.get_name()}"

    def get_name(self):
        return "Frugal"

    def measure(self, x: List[str], batch_size=128, max_length=128) -> List[float]:
        generated_samples = [gen_sample for gen_sample in self.gen_dataset.load_dataset(N=self.N)]

        if self.enable_caching:
            self.load_cache()
        scores = []
        for i, candidate in enumerate(tqdm(x, desc="Computing Frugalscore")):
            score = self.get_elem_from_cache(candidate, generated_samples)
            if score is None: # the element does not appear to be in the cache
                score = max(self.frugalscore.compute(predictions=len(generated_samples) * [candidate],
                                         references=generated_samples,
                                         batch_size=batch_size, max_length=max_length, device="gpu")['scores'])
                self.add_elem_to_cache(candidate, generated_samples, float(score))
            scores.append(score)
            if self.enable_caching and (i+1) % self.save_cache_every == 0:
                self.save_cache()   # intermediate saving of cache to prevent losing progress.
        if not self.larger_is_better():
            scores = [-x for x in scores]
        if self.enable_caching:
            self.save_cache()
        return scores

class BERTScoreSignal(PairedMIASignal):
    def __init__(self, target_model: LanguageModel, sampling_args: SamplingArgs, N: int = 256, **kwargs):
        super().__init__(target_model, **kwargs)
        self.N = N
        self.gen_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)
        self.bertscore = evaluate._load("bertscore", cache_dir=os.path.expanduser("~/.cache/my_cache"))

    def get_cache_fp(self) -> str:
        return f"bertscore_{self.target_model.get_name()}_ckpt={self.target_model.get_checkpoint()}.json"

    def __str__(self):
        return f"BertScore, N={self.N}, caching={self.enable_caching}, Model={self.target_model.get_name()}"

    def get_name(self):
        return "BERT"

    def measure(self, x: List[str], batch_size=128, max_length=128) -> List[float]:
        generated_samples = [gen_sample for gen_sample in self.gen_dataset.load_dataset(N=self.N)]

        if self.enable_caching:
            self.load_cache()
        scores = []
        for i, candidate in enumerate(tqdm(x, desc="Computing BertScore")):
            score = self.get_elem_from_cache(candidate, generated_samples)
            if score is None: # the element does not appear to be in the cache
                score = max(self.bertscore.compute(predictions=len(generated_samples) * [candidate],
                                         references=generated_samples,
                                         batch_size=batch_size, model_type="distilbert-base-uncased", device="cuda")['precision'])
                self.add_elem_to_cache(candidate, generated_samples, score)
            scores.append(score)
            if self.enable_caching and (i+1) % self.save_cache_every == 0:
                self.save_cache()   # intermediate saving of cache to prevent losing progress.
        if not self.larger_is_better():
            scores = [-x for x in scores]
        if self.enable_caching:
            self.save_cache()
        return scores

class ROUGESignal(PairedMIASignal):
    def __init__(self, target_model: LanguageModel, sampling_args: SamplingArgs, N: int = 256, name="rouge1", **kwargs):
        super().__init__(target_model, **kwargs)
        self.N = N
        self.name = name
        self.gen_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)
        self.rouge = evaluate._load("rouge", cache_dir=os.path.expanduser("~/.cache/my_cache"))

    def get_cache_fp(self) -> str:
        return f"rougescore_{self.target_model.get_name()}_ckpt={self.target_model.get_checkpoint()}.json"

    def __str__(self):
        return f"ROUGEscore ({self.name}), N={self.N}, caching={self.enable_caching}, Model={self.target_model.get_name()}"

    def get_name(self):
        return "ROUGE"

    def measure(self, x: List[str], batch_size=128, max_length=128) -> List[float]:
        generated_samples = [gen_sample for gen_sample in self.gen_dataset.load_dataset(N=self.N)]

        if self.enable_caching:
            self.load_cache()
        scores = []
        for i, candidate in enumerate(tqdm(x, desc=f"Computing ROUGE")):
            score = self.get_elem_from_cache(candidate, generated_samples)
            if score is None: # the element does not appear to be in the cache
                score = self.rouge.compute(predictions=[candidate], references=[generated_samples])
                self.add_elem_to_cache(candidate, generated_samples, score)
            scores.append(score[self.name])
            if self.enable_caching and (i+1) % self.save_cache_every == 0:
                self.save_cache()   # intermediate saving of cache to prevent losing progress.
        if not self.larger_is_better():
            scores = [-x for x in scores]
        if self.enable_caching:
            self.save_cache()
        return scores

class ROUGE1Signal(ROUGESignal):
    def __init__(self, *args, name='rouge1', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def get_name(self):
        return "ROUGE1"

class ROUGE2Signal(ROUGESignal):
    def __init__(self, *args, name='rouge2', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def get_name(self):
        return "ROUGE2"

class ROUGELSignal(ROUGESignal):
    def __init__(self, *args, name='rougeL', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def get_name(self):
        return "ROUGE_L"

class ROUGELsumSignal(ROUGESignal):
    def __init__(self, *args, name='rougeLsum', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def get_name(self):
        return "ROUGE_LS"

class BLEUSignal(PairedMIASignal):
    def __init__(self, target_model: LanguageModel, sampling_args: SamplingArgs, N: int = 256,
                 **kwargs):
        super().__init__(target_model, **kwargs)
        self.N = N
        self.gen_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)
        self.bleu = evaluate._load("bleu", cache_dir=os.path.expanduser("~/.cache/my_cache"))

    def get_cache_fp(self) -> str:
        return f"bleuscore_{self.target_model.get_name()}_ckpt={self.target_model.get_checkpoint()}.json"

    def __str__(self):
        return f"BLEUscore, N={self.N}, Model={self.target_model.get_name()}"

    def get_name(self):
        return "BLEU"

    def measure(self, x: List[str]) -> List[float]:
        generated_samples = [gen_sample for gen_sample in self.gen_dataset.load_dataset(N=self.N)]

        if self.enable_caching:
            self.load_cache()
        scores = []
        for i, candidate in enumerate(tqdm(x, desc=f"Computing BLEUscore")):
            score = self.get_elem_from_cache(candidate, generated_samples)
            if score is None: # the element does not appear to be in the cache
                score = self.bleu.compute(predictions=[candidate], references=[generated_samples])
                self.add_elem_to_cache(candidate, generated_samples, score)
            scores.append(score['bleu'])
            if self.enable_caching and (i+1) % self.save_cache_every == 0:
                self.save_cache()   # intermediate saving of cache to prevent losing progress.
        if not self.larger_is_better():
            scores = [-x for x in scores]
        if self.enable_caching:
            self.save_cache()
        return scores

class PPLSignal(MIASignalNoCache):
    def larger_is_better(self):
        return False

    def __str__(self):
        return f"Perplexity Signal ({self.target_model.get_name()})"

    def get_name(self):
        return "Perplexity"

    def measure(self, x) -> List[float]:
        scores = [self.target_model.perplexity(x_i, verbose=False) for x_i in tqdm(x, desc="Measuring PPL")]
        if not self.larger_is_better():
            scores = [-x for x in scores]
        return scores

