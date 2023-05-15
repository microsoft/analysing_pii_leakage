import os
import sys

from src.dataset.auto_dataset import AutoDatasetWrapper
from old_examples.evaluate_ppl_iters import resolve_label
from old_examples.evaluate_mia import shadow_model_attack, population_attack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """ This script plots the iterations on the x-axis and the ROC AUC on the y-axis for
    different attack modes (population and shadow)
     """
    parser = transformers.HfArgumentParser((DatasetArgs, SamplingArguments))

    parser.add_argument("--target", type=str, help="target model name")
    parser.add_argument("--shadow", nargs='+', help="shadow model names")
    parser.add_argument("--scores", nargs='+', help="which score to compute")

    parser.add_argument("--n_train", type=int, default=500, help="number of members in the target data")
    parser.add_argument("--n_test", type=int, default=500, help="number of non-members in the target data")

    dataset_args, sampling_args, other_args = parser.parse_args_into_dataclasses()
    return dataset_args, sampling_args, other_args

if __name__ == "__main__":
    args = parse_args()
    dataset_args, sampling_args, other_args = args

    dataset = AutoDatasetWrapper.from_existing(dataset_args, split='train').dataset
    dataset_test = AutoDatasetWrapper.from_existing(dataset_args, split='test').dataset
    seed = 42   # changing the seed will require re-computing the cache

    train_split = dataset.select(rnd_idx(len(dataset), seed=seed)[:other_args.n_train])
    test_split = dataset_test.select(rnd_idx(len(dataset_test), seed=seed)[:other_args.n_test])

    model_args = ModelArguments(model_name=other_args.target)
    target_model = ModelFactory.load_model(model_args, hollow=True)
    public_model = ModelFactory.load_model(ModelArguments(model_name="echr_public"))

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

    for signal_class in signals_classes:
        x, y_population, y_shadow = [], [], []

        shadow_model =ModelFactory.load_model(ModelArguments(model_name=other_args.shadow[0]))
        for ckpt in [0] + target_model.get_available_checkpoints():
            if ckpt == 0:
                model = public_model
            else:
                if not ckpt in shadow_model.get_available_checkpoints():
                    print("Cannot find a shadow model ckpt .. ")
                    continue
                shadow_model.load(ckpt=ckpt)
                target_model.load(ckpt=ckpt)
                model = target_model

            shadow_signals = [signal_class(shadow_model, sampling_args=sampling_args, enable_caching=True)]
            target_signal = signal_class(model, sampling_args=sampling_args, enable_caching=True)

            x.append(ckpt)
            y_population += [population_attack(train_split, test_split, target_signal, show_signal=False)]
            y_shadow += [shadow_model_attack(train_split, test_split, target_signal, shadow_signals, sampling_args,
                                             fast=True, show_signal=False)]
        plt.plot(x, y_shadow, label=resolve_label(f"Shadow Attack ({target_model.get_name()})"), marker='x')
        plt.plot(x, y_population, label=resolve_label(f"Population Attack ({target_model.get_name()})"), marker='x')

    plt.title("Training Iteration vs ROC AUC")
    plt.ylabel("ROC AUC")
    plt.ylim(0.4, 1.05)
    plt.xlabel("Training Iterations")
    plt.legend()
    plt.show()




