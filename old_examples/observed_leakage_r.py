import os
import sys

from tqdm import tqdm

from old_examples.evaluate_ppl import mname_to_dname
from src.models.language_model import LanguageModel
from src.utils import print_highlighted

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from matplotlib import pyplot as plt

from src.dataset.auto_dataset import AutoDatasetWrapper
from src.dataset import GeneratedDatasetWrapper
from src.dataset.text_dataset import TextDatasetWrapper
from src.models.model_factory import ModelFactory

import transformers

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.arguments import SamplingArguments


def parse_args():
    parser = transformers.HfArgumentParser((ModelArguments, SamplingArguments, DatasetArgs))

    parser.add_argument("--model_names", nargs='+', help="model names to compute ppl over")
    parser.add_argument("--n_samples", type=int, default=15_000, help="number of samples to generate")

    model_args, sampling_args, dataset_args, other_args = parser.parse_args_into_dataclasses()
    return model_args, sampling_args, dataset_args, other_args


def plot_number_of_samples_vs_pr(target_models: List[LanguageModel],
                                 generated_datasets: List[GeneratedDatasetWrapper],
                                 real_dataset: TextDatasetWrapper,
                                 metric="Recall",
                                 entity_type="person"):
    """ Observed Leakage.
    Plots the number of samples versus the precision and recall.
    """
    real_entities: List[str] = real_dataset.get_unique_entities(entity_type=entity_type)

    for target_model, generated_dataset in zip(target_models, generated_datasets):
        generated_entities: dict = generated_dataset.load_entities(entity_type=entity_type, only_matches=False)

        print_highlighted(f"({target_model.get_name()} Found {len(real_entities)} unique PII in train and {len(generated_entities)} "
                          f"entities in generated!")

        x, precision, recall = [], [], []
        all_mentions = set()
        running_positives = 0
        for j, (_, entity_mentions) in enumerate(tqdm(generated_entities.items())):
            for entity_mention in list(set(entity_mentions)):
                l0 = len(all_mentions)
                all_mentions.add(entity_mention)
                if len(all_mentions) > l0:
                    running_positives += 1 if entity_mention in real_entities else 0
            x += [j*generated_dataset.seq_len]
            if len(real_entities) == 0:
                recall += [0]
            else:
                recall += [running_positives / len(real_entities)]
            if len(all_mentions) == 0:
                precision += [0]
            else:
                precision += [running_positives / len(all_mentions)]
        if metric.lower() == "recall":
            plt.plot(x, recall, label=f"{mname_to_dname(target_model.get_name())}", linestyle="--")
        else:
            plt.plot(x, precision, label=f"{mname_to_dname(target_model.get_name())}", linestyle="--")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """ This script generates all plots necessary to evaluate PII leakage in a given model.
    """
    args = parse_args()
    _, sampling_args, dataset_args, other_args = args
    sampling_args.N = other_args.n_samples  # overwrite sampling parameter

    # -- Load the training dataset and all entities -- #
    real_train: TextDatasetWrapper = AutoDatasetWrapper.from_existing(dataset_args)

    # -- Load the models -- #
    target_models: List[LanguageModel] = []
    generated_datasets: List[GeneratedDatasetWrapper] = []
    for model_name in other_args.model_names:
        model_args = ModelArguments(model_name=model_name)
        target_models += [ModelFactory.load_model(model_args)]
        generated_datasets += [GeneratedDatasetWrapper.from_sampling_args(model=target_models[-1],
                                                                          sampling_args=sampling_args)]
        print(f"Found {generated_datasets[-1].size()} elements in data from {target_models[-1].get_name()}")

    plot_number_of_samples_vs_pr(target_models, generated_datasets, real_train, metric="Recall")
    plot_number_of_samples_vs_pr(target_models, generated_datasets, real_train, metric="Precision")

# ----------------------------------------------------------------------------
