import os
import sys

from tqdm import tqdm

from src.attacks.mia import ReconstructionResults, ReconstructionResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from typing import List

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.dataset.auto_dataset import AutoDatasetWrapper
from src.dataset.text_dataset import TextDatasetWrapper
from src.models.model_factory import ModelFactory
from src.models.language_model import LanguageModel
from src.ner import NERAnalyzer


import transformers

from src.arguments import SamplingArguments
from src.utils import print_highlighted

def parse_args():
    """ Reconstruction attack.
    The attacker has a set of sequences from the training dataset where a PII has been redacted.
    The target model has been trained on the non-redacted data. Can an attacker infer the PII?
    """
    parser = transformers.HfArgumentParser((ModelArguments, DatasetArgs, SamplingArguments))
    model_args, dataset_args, sampling_args = parser.parse_args_into_dataclasses()
    return model_args, dataset_args, sampling_args

def get_first_name(sentence: str, ner: NERAnalyzer, offset: int):
    result = [sentence[r.start:r.end] for r in ner.analyze(sentence) if r.start >= offset]
    if len(result) > 0:
        return result[0]
    return ""

def _reconstruct_pii_in_sample(target_model: LanguageModel, sample: str,
                               sampling_args: SamplingArguments, ner: NERAnalyzer, N=64, batch_size=32, fast=False):
    """ Extract a set of names from a target model. Idea: If top-1 generation is a name, return top-1 generation.
    Otherwise, autocomplete with high temperature, extract set of names and return most frequent name.
    """
    # Check if top-1 is a name.
    top1 = get_first_name(target_model.autocomplete(sample, seq_len = 20), ner=ner, offset=len(sample))
    if top1 is None and not fast:    # top token wasn't a name. do random sample.
        generated_sequences = target_model.generate(N=N, input_prompt=sample, seq_len=20, top_k=sampling_args.top_k,
                                   top_p=sampling_args.top_p, batch_size=batch_size, verbose=False)
        piis = [x for x in [get_first_name(generated_sequence, ner=ner, offset=len(sample)) for generated_sequence in tqdm(generated_sequences, desc="NER",
                                                                                              disable=True)] if x is not None]
        if len(piis) == 0:
            return None
        return max(set(piis), key=piis.count)
    return top1

def reconstruction_attack(model_args: ModelArguments,
                          dataset_args: DatasetArgs,
                          sampling_args: SamplingArguments):
    """ Given access to a redacted target dataset, reconstruct missing PIIs.
    """
    # -- Load the training data and all PIIs --
    real_train: TextDatasetWrapper = AutoDatasetWrapper.from_existing(dataset_args)
    real_piis: List[str] = real_train.get_unique_entities(entity_type='PERSON', shuffle=True, seed=42)
    candidates = real_piis[:1000]

    # -- Load the target model --
    target_model = ModelFactory.load_model(model_args)

    # -- Get pii-sentence pairs that we want to infer --
    entity_sample_pairs = []
    for pii in candidates:
        samples = real_train.search_for_samples_with_euii(pii)
        random.Random(42).shuffle(samples)
        for sample in samples[:5]:
            entity_sample_pairs.append((pii, sample))
    random.Random(42).shuffle(entity_sample_pairs)

    ner = NERAnalyzer()
    results = ReconstructionResults()
    mask_pii_fn = lambda entity, sample: sample.replace(entity, '[MASK]')
    for i, (pii_real, sample) in enumerate(entity_sample_pairs):
        masked_sample = mask_pii_fn(pii_real, sample)
        truncated_sample = masked_sample[:masked_sample.index('[MASK]')]
        if len(truncated_sample) < 5:
            continue
        truncated_sample = truncated_sample[:-1]    # add whitespace before PII

        pii_pred = _reconstruct_pii_in_sample(target_model, sample=truncated_sample, ner=ner, sampling_args=sampling_args)
        results.append(ReconstructionResult(pii_pred=pii_pred, pii_real=pii_real))
        results.print_stats()

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    print_highlighted(f"Running script with {args}")
    model_args, dataset_args, sampling_args = args

    reconstruction_attack(model_args=model_args, sampling_args=sampling_args, dataset_args=dataset_args)

# ----------------------------------------------------------------------------
