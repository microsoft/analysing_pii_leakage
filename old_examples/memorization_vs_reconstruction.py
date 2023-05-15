import ast
import os
import sys

from tqdm import tqdm

from src.dataset.auto_dataset import AutoDatasetWrapper
from old_examples.evaluate_ppl_iters import resolve_label
from old_examples.evaluate_reconstruction import get_first_name
from src.ner import NERAnalyzer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import transformers
from matplotlib import pyplot as plt

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.arguments import SamplingArguments
from src.attacks.mia import PPLSignal, BLEUSignal, ROUGE1Signal, ROUGE2Signal, ROUGELsumSignal, BERTScoreSignal, \
    FrugalScoreSignal
from src.models.model_factory import ModelFactory
from src.models.util import rnd_idx

def parse_args():
    """ This script evaluates a set of model's vulnerability to PII extraction
     """
    parser = transformers.HfArgumentParser((DatasetArgs, SamplingArguments))

    parser.add_argument("--target", type=str, help="target model name")
    parser.add_argument("--shadow", nargs='+', help="[optional] shadow model names")
    parser.add_argument("--attack", default="shadow", choices=["shadow", "population"], help="attack mode")
    parser.add_argument("--scores", nargs='+', help="which score to compute", default='ppl')

    parser.add_argument("--n_train", type=int, default=10000, help="number of members in the target data")

    dataset_args, sampling_args, other_args = parser.parse_args_into_dataclasses()
    return dataset_args, sampling_args, other_args


if __name__ == "__main__":
    args = parse_args()
    dataset_args, sampling_args, other_args = args

    dataset = AutoDatasetWrapper.from_existing(dataset_args, split='train').dataset
    seed = 42   # changing the seed will require re-computing the cache

    train_split = dataset.select(rnd_idx(len(dataset), seed=seed)[:other_args.n_train])

    model_args = ModelArguments(model_name=other_args.target)
    target_model = ModelFactory.load_model(model_args)

    signal_cls = {
        "ppl": PPLSignal,
        "bleu": BLEUSignal,
        "rouge1": ROUGE1Signal,
        "rouge2": ROUGE2Signal,
        "rougesum": ROUGELsumSignal,
        "frugal": FrugalScoreSignal,    # 0:13 min for 100 samples (at N=100)
        "bert": BERTScoreSignal         # 1:13 min for 100 samples (at N=100)
    }
    signals_classes = [signal_cls[score] for score in other_args.scores]
    target_signal = signals_classes[0](target_model, sampling_args=sampling_args, enable_caching=True)
    shadow_model = ModelFactory.load_model(ModelArguments(model_name=other_args.shadow[0]))
    shadow_signal = signals_classes[0](shadow_model, sampling_args=sampling_args, enable_caching=True)

    #t = target_signal.measure([x for x in train_split['text']])
    #s = shadow_signal.measure([x for x in train_split['text']])
    #mem_scores = [t_i - s_i for t_i, s_i in zip(t, s)]
    mem_scores = [0 for i in train_split]

    ner = NERAnalyzer()

    x, y = [], []
    for mem_score, data in zip(tqdm(mem_scores, disable=True), train_split):
        sample, persons = data['text'], ast.literal_eval(data['persons'])

        for person in list(set(persons)):
            p_occ = sample.index(person) - 1 # -1 for whitespace
            if p_occ < 10:
                continue    # we require at least 10 chars of context
            pii_pred = get_first_name(target_model.autocomplete(sample[:p_occ], seq_len=15), offset=p_occ, ner=ner)

            x.append(mem_score)
            if pii_pred is None or pii_pred.strip().lower() != person.strip().lower():
                y.append(0)
            else:
                print(f"Found a match!: {person}")
                y.append(1)
    theta = np.polyfit(np.asarray(x), np.asarray(y), 1)
    y_line = theta[1] + theta[0] * np.asarray(x)
    plt.plot(x, y_line, label=str(resolve_label(target_model.get_name())))

    bins = 10
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    def plot_window(x, y, window_size=10):
        x_new, y_new = [], []
        for i in range(len(y) - window_size):
            x_new.append(np.mean(x[i:i + window_size]))
            y_new.append(np.mean(y[i:i + window_size]))
        plt.plot(x_new, y_new, label=f"{window_size}")

    plot_window(x=x, y=y, window_size=10)
    print(f"TOTAL DATA POINTS: {len(x)}")


    plt.title("Memorization vs Reconstruction")
    plt.ylabel("Probability of Correct PII Reconstruction")
    plt.xlabel("Memorization Score")
    plt.legend()
    plt.show()

    x_range, y_val = [], []
    binsize = len(x) // bins
    for bin_i in range(bins):
        x_range += [(min(x[bin_i*binsize:(bin_i+1) * binsize]), max(x[bin_i*binsize:(bin_i+1) * binsize]))]
        y_val += [float(np.mean(y[bin_i*binsize:(bin_i+1) * binsize]))]


    x, y = [], []
    for (x_min, x_max), height in zip(x_range, y_val):
        x.append((x_max-x_min) / 2)
        y.append(height)
    plot_window(x=x, y=y, window_size=2)


    plt.title("Memorization versus PII Reconstruction")
    plt.ylabel("Probability of Correct PII Reconstruction")
    plt.xlabel("Memorization Score")
    plt.legend()
    plt.show()









