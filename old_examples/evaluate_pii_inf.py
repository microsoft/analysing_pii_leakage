import os
import random
import sys
from typing import List

import numpy as np
from tqdm import tqdm

from src.dataset.auto_dataset import AutoDatasetWrapper
from src.dataset.text_dataset import TextDatasetWrapper
from src.attacks.mia import PIIInferenceResults, PIIInferenceResult
from src.models.model_factory import ModelFactory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.arguments import SamplingArguments


def parse_args():
    parser = transformers.HfArgumentParser((SamplingArguments, DatasetArgs))

    parser.add_argument("--target", type=str, help="target model name")
    parser.add_argument("--model_arch", type=str, help="target model arch", default="gpt2")
    parser.add_argument("--shadow", type=str, help="shadow model name")
    parser.add_argument("--candidates", type=int, default=1000, help="number of candidates (random sample) to consider")

    sampling_args, dataset_args, other_args = parser.parse_args_into_dataclasses()
    return sampling_args, dataset_args, other_args

def get_acc_pii_inference(target_model, piis_to_samples, candidates, verbose=False, show_progress=True, fast=True):
    results = PIIInferenceResults()
    impute_fn = lambda pii, sample, new_pii: sample.replace(pii, new_pii)
    for pii, sample in tqdm(piis_to_samples, disable=not show_progress):
        pii_score = target_model.perplexity(sample, verbose=False)
        scores = []
        for candidate in candidates:
            imputed_sample = impute_fn(pii, sample, candidate)
            scores += [target_model.perplexity(imputed_sample, verbose=False)]
            if scores[-1] < pii_score and fast:
                break # early stopping, we found a better one
        best = np.argsort(scores)[0]
        results.append(PIIInferenceResult(pii_real=pii, pii_pred=candidates[best], score=scores[best]))
        if verbose:
            results.print_stats()
    return results.accuracy()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    sampling_args, dataset_args, other_args = args

    print(dataset_args)
    real_train: TextDatasetWrapper = AutoDatasetWrapper.from_existing(dataset_args)
    real_piis: List[str] = real_train.get_unique_entities(entity_type='PERSON', shuffle=True, seed=42)
    candidates = real_piis[:other_args.candidates]


    # Get one sentence where candidate PII occurs
    piis_to_samples = []
    for candidate in candidates:
        samples = real_train.search_for_samples_with_euii(entitiy_mention=candidate)
        random.Random(42).shuffle(samples)
        piis_to_samples += [[candidate, samples[0]]]
    random.Random(42).shuffle(piis_to_samples)

    target_model = ModelFactory.load_model(ModelArguments(model_name=other_args.target, model_arch=other_args.architecture))
    accuracy = get_acc_pii_inference(target_model, piis_to_samples[:50], candidates, verbose=True)
    print(f"Measured Acc for {target_model.get_name()}: {accuracy}")



# ----------------------------------------------------------------------------