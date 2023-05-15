import os
import random
import sys

import numpy as np
import pandas
from tqdm import tqdm
import seaborn as sns

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
    parser.add_argument("--n_samples", type=int, default=35_000, help="number of samples to generate")

    model_args, sampling_args, dataset_args, other_args = parser.parse_args_into_dataclasses()
    return model_args, sampling_args, dataset_args, other_args


def plot_training_vs_generated_pii_duplication(target_model: LanguageModel,
                                               real_dataset: TextDatasetWrapper,
                                               generated_dataset: GeneratedDatasetWrapper):
    """ Observed Leakage.
    Count how often a PII appears in the training data versus how often does it appears
    in the generated data
    """
    real_entities = real_dataset.get_unique_entities(entity_type="person")
    real_entities.sort(key=lambda x: generated_dataset.get_entity_duplication_count(x), reverse=True)
    x = [real_dataset.get_entity_duplication_count(entity) for entity in real_entities]  # Training Data Duplication
    y = [generated_dataset.get_entity_duplication_count(entity) for entity in real_entities]  # Observed Leakage
    df = pandas.DataFrame.from_dict({"x": x, "y": y})
    df = df.groupby(['x']).agg({'y': 'mean'}).reset_index()
    df = df.sort_values(by=['x']).rolling(window=1, on='y').mean().reset_index().dropna()
    sns.regplot(x="x", y="y", data=df, label=target_model.get_name(), color='C0')
    plt.ylim(bottom=0)
    plt.xscale('log')
    plt.title("Training Data versus Generated PII Duplication")
    plt.xlabel("Training Data Duplication")
    plt.ylabel("Observed Leakage")
    plt.legend()
    filename_pdf = f"training_vs_generated_duplication_{target_model.get_name()}.pdf"
    filename_csv = f"training_vs_generated_duplication_{target_model.get_name()}.csv"
    df.to_csv(filename_csv)
    plt.savefig(filename_pdf)
    print(f"Saved file at '{os.path.abspath(filename_pdf)}' and '{os.path.abspath(filename_csv)}'.")
    plt.show()


def plot_number_of_samples_vs_recall(target_model: LanguageModel,
                                     real_dataset: TextDatasetWrapper,
                                     generated_dataset: GeneratedDatasetWrapper,
                                     entity_type="person"):
    """ Observed Leakage.
    Plots the number of samples versus the precision and recall.
    """
    real_entities: List[str] = real_dataset.get_unique_entities(entity_type=entity_type)
    generated_entities: dict = generated_dataset.load_entities(entity_type=entity_type)

    print_highlighted(f"Found {len(real_entities)} unique PII in train and {len(generated_entities)} entities in generated!")

    x, precision, recall = [], [], []
    all_mentions = set()
    running_positives = 0
    for j, (batch_idx, entity_mentions) in enumerate(tqdm(generated_entities.items())):
        for entity_mention in list(set(entity_mentions)):
            l0 = len(all_mentions)
            all_mentions.add(entity_mention)
            if len(all_mentions) > l0:
                running_positives += 1 if entity_mention in real_entities else 0
        x += [j*generated_dataset.seq_len]
        recall += [running_positives / len(real_entities)]
        precision += [running_positives / len(all_mentions)]
    plt.title("Observed Leakage (Precision & Recall)")
    plt.plot(x, precision, label=f"Precision ({target_model.get_name()})")
    plt.plot(x, recall, label=f"Recall ({target_model.get_name()})")
    plt.legend()
    plt.ylabel("Probability")
    plt.xlabel("Number of Tokens")
    #plt.ylim(bottom=0, top=1)
    #plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_estimated_vs_observed_leakage(target_model: LanguageModel,
                                       public_model: LanguageModel,
                                       real_train: TextDatasetWrapper,
                                       generated_dataset: GeneratedDatasetWrapper,
                                       k=5) -> List[float]:
    """ Computes the extractability advantage of entities in a target versus a public model.

    The estimation computes the support of a PII in a set of generated text as an approximation for the model's
    likelihood of generating that PII.
    """
    entity_dict: dict = generated_dataset.load_entities()  # {batch_idx: {"entity_type": ["entity_mention",..]}}
    entity_dict = {k: v["person"] for k, v in entity_dict.items() if len(v.setdefault("person", [])) > 0}

    sampling_args.seq_len = 1_024
    sampling_args.top_k = 1000

    real_entities: List[str] = real_train.get_unique_entities(entity_type="person")
    # real_entities.sort(key=lambda x: real_train.get_entity_duplication_count(x), reverse=True)

    estimated_leakage = []
    observed_leakage = []
    for target_entity in real_entities:
        observed_leakage += [real_train.get_entity_duplication_count(target_entity)]

        target_ppls = []
        shuffled_keys = list(entity_dict.keys())
        random.shuffle(shuffled_keys)

        for j, key in enumerate(tqdm(shuffled_keys)):
            entity_mentions = entity_dict[key]
            if j > k:
                break
            seq: str = generated_dataset.load_dataset()[int(key)]
            pick = 0  # always replace the first PII
            entity_pick = entity_mentions[pick]
            seq2 = seq.replace(entity_pick, target_entity)
            seq2 = seq2[:seq2.index(target_entity) + len(target_entity)]
            print(target_entity, seq2)
            target_ppls += [
                -np.log(1e-20 + target_model.probability(seq2, suffix=target_entity, sampling_args=sampling_args))]
        estimated_leakage += [np.mean(target_ppls)]

        df = pandas.DataFrame.from_dict({"x": observed_leakage, "y": estimated_leakage})
        sns.regplot(x="x", y="y", data=df, label=target_model.get_name(), color='C0',
                    scatter_kws={'alpha': 0.3})
        plt.ylim(bottom=0)
        plt.xscale('log')
        plt.title("Observed versus Estimated Leakage")
        plt.xlabel("Duplication in Training Data")
        plt.ylabel("Estimated Leakage")
        plt.legend()
        plt.show()

        '''sns.histplot(data=df, x="y", kde=True)
        plt.show()'''


def plot_training_duplication_vs_mentions(sampling_args: SamplingArguments,
                                          model_args: ModelArguments,
                                          generated_dataset: GeneratedDatasetWrapper,
                                          real_dataset: TextDatasetWrapper):
    """ This function plots the training data duplication (X-axis) versus the
        generated data duplication
    """
    gen_entities: dict = generated_dataset.get_unique_entities(N=sampling_args.N, entity_type="person")
    real_entities: List = real_train.get_unique_entities(N=sampling_args.N, entity_type="person")

    real_dups, gen_dups = [], []
    for entity in tqdm(real_entities, f"Counting Duplication"):
        gen_dups += [generated_dataset.get_entity_duplication_count(N=sampling_args.N, entity_mention=entity)]
        real_dups += [real_dataset.get_entity_duplication_count(entity_mention=entity)]

    plt.title("Training versus Generated PII Duplication")
    plt.scatter(real_dups, gen_dups, label=model_args.model_name)
    plt.xlabel("Duplication in Training Data")
    plt.ylabel(f"Duplication in Generated Data")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    import seaborn as sns

    plt.title("Training versus Generated PII Duplication")
    df = pandas.DataFrame.from_dict({"x": real_dups, "y": gen_dups})
    sns.regplot(x=real_dups, y=gen_dups, data=df, x_bins=20, label=model_args.model_name)
    plt.xlabel("Training Data")
    plt.ylabel("Generated")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """ This script generates all plots necessary to evaluate PII leakage in a given model.
    """

    args = parse_args()
    model_args, sampling_args, dataset_args, other_args = args

    # -- Load the training dataset and all entities -- #
    real_train: TextDatasetWrapper = AutoDatasetWrapper.from_existing(dataset_args)

    # -- Load the models -- #
    target_model = ModelFactory.load_model(model_args)
    #public_model = ModelFactory.load_model(deepcopy(model_args).make_public())

    # -- Load the private generated dataset and all entities -- #
    sampling_args.N = other_args.n_samples  # overwrite sampling parameter
    generated_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)

    # -- Plot everything -- #
    plot_number_of_samples_vs_recall(target_model, real_train, generated_dataset)
    exit()
    # Plot training data duplication versus generated data duplication
    #plot_training_vs_generated_pii_duplication(target_model, real_train, generated_dataset)


    # Estimate leakage for PIIs
    # plot_estimated_vs_observed_leakage(target_model, public_model, real_train, generated_dataset)

    # -- Plot duplication vs number of mentions --
    # plot_training_duplication_vs_mentions(sampling_args, model_args, generated_dataset, real_train)

# ----------------------------------------------------------------------------
