import math
import os
import sys
from typing import List

from old_examples.evaluate_ppl_iters import resolve_label

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.arguments import SamplingArguments
from src.dataset import GeneratedDatasetWrapper
from src.attacks.mia import PPLSignal, BLEUSignal, ROUGE1Signal, ROUGE2Signal, ROUGELsumSignal, BERTScoreSignal, \
    FrugalScoreSignal, MIASignal
from src.models.model_factory import ModelFactory
from src.models.util import rnd_idx

def parse_args():
    """ This script evaluates a set of model's vulnerability to PII extraction
     """
    parser = transformers.HfArgumentParser((DatasetArgs, SamplingArguments))

    parser.add_argument("--target", type=str, help="target model name")
    parser.add_argument("--shadow", nargs='+', help="[optional] shadow model names")
    parser.add_argument("--attack", default="shadow", choices=["shadow", "population"], help="attack mode")
    parser.add_argument("--scores", nargs='+', help="which score to compute")
    parser.add_argument("--fast", type=bool, default=False, help="whether to skip unnecessary generation of plots")

    parser.add_argument("--n_train", type=int, default=500, help="number of members in the target data")
    parser.add_argument("--n_test", type=int, default=500, help="number of non-members in the target data")

    dataset_args, sampling_args, other_args = parser.parse_args_into_dataclasses()
    return dataset_args, sampling_args, other_args

def plot_signal_hist(member_signals, nonmember_signals, title="Signal Histogram"):
    import seaborn as sn
    sn.histplot(
        data=pd.DataFrame({
            'Signal': member_signals + nonmember_signals,
            'Membership': ['Member'] * len(member_signals) + ['Non-member'] * len(nonmember_signals)
        }),
        x='Signal',
        hue='Membership',
        element='step',
        stat='density',
        kde=True
    )
    plt.grid()
    plt.xlabel('Signal value')
    plt.ylabel('Density')
    plt.title(resolve_label(title))
    savepath = f"signal_{target_model.get_name()}.svg"
    plt.savefig(savepath, dpi=300)
    print(f"Saved at {os.path.abspath(savepath)}")
    plt.show()

def plot_log_prob_vs_calibrated_loss(member_signals_x, member_signals_y,
                                     nonmember_signals_x, nonmember_signals_y,
                                     gen_signals_x, gen_signals_y):
    """ Plots the log probability (x) versus the calibrated loss (y)
    """
    import seaborn as sn

    def filter_outliers(x, y):
        merged = [(x_i, y_i) for x_i, y_i in zip(x, y)]
        idx = np.argsort(y)
        qmin, qmax = y[idx[len(y) // 7]], y[idx[ 8 * len(y) // 10]]
        xnew = [x_i[0] for x_i in merged if qmin <= x_i[1] <= qmax]
        ynew = [x_i[1] for x_i in merged if qmin <= x_i[1] <= qmax]
        return xnew, ynew
    gen_signals_x, gen_signals_y = filter_outliers(gen_signals_x, gen_signals_y)

    fig, ax = plt.subplots(figsize=(6, 6))
    df = pd.DataFrame({
        'x': member_signals_x,
        'y': member_signals_y
    })
    sn.scatterplot(data=df, x="x", y="y", color="green", ax=ax, label="Member", alpha=0.7)
    sn.kdeplot(data=df, x="x", y="y", levels=5, color="green", fill=True, alpha=0.4, cut=5, ax=ax)

    df = pd.DataFrame({
        'x': nonmember_signals_x,
        'y': nonmember_signals_y
    })
    sn.scatterplot(data=df, x="x", y="y", color="blue", ax=ax, label="Non-member", alpha=0.7)
    sn.kdeplot(data=df, x="x", y="y", levels=5,  color="blue", fill=True, alpha=0.4, cut=5, ax=ax)

    df = pd.DataFrame({
        'x': gen_signals_x,
        'y': gen_signals_y
    })
    sn.scatterplot(data=df, x="x", y="y", color="orange", ax=ax, label="Generated", alpha=0.7)
    sn.kdeplot(data=df, x="x", y="y", levels=5, color="orange", fill=True, alpha=0.4, cut=5, ax=ax)
    plt.ylim(-10,10)
    plt.title("Extractability versus Memorization")
    plt.xlabel("Log Probability")
    plt.ylabel("Calibrated Score")
    plt.legend()

    savepath = f"extractability_vs_mem_{target_model.get_name()}.svg"
    plt.savefig(savepath, dpi=300)
    print(f"Saved at {os.path.abspath(savepath)}")
    plt.show()

def plot_roc_auc(member_signals, nonmember_signals, title="ROC curve", model_name=None, show=True):
    """ Larger scores should be better!
    """
    y_test = [1] * len(member_signals) + [0] * len(nonmember_signals)
    y_score = member_signals + nonmember_signals

    idx = np.arange(len(y_score))
    np.random.shuffle(idx)
    y_test = [y_test[i] for i in idx]
    y_score = [y_score[i] for i in idx]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    if show:
        range01 = np.linspace(0, 1)
        plt.fill_between(fpr, tpr, alpha=0.15)
        plt.plot(fpr, tpr)
        plt.plot(range01, range01, '--', label='Random guess')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.legend()
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title(resolve_label(title))
        plt.text(
            0.7, 0.3,
            f'AUC = {roc_auc:.03f}',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.5)
        )

        savepath = f"roc_auc_{model_name}.svg"
        plt.savefig(savepath, dpi=300)
        print(f"Saved at {os.path.abspath(savepath)}")
        plt.show()
    return roc_auc


def shadow_model_attack(train_split, test_split,
                      target_signal: MIASignal,
                      shadow_signals: List[MIASignal],
                      sampling_args: SamplingArguments,
                      fast: bool = False,
                      show: bool = True,
                      show_signal: bool = True) -> float:
    """ Shadow model attack.
    """
    target_model = target_signal.get_target_model()

    member_signals = target_signal.measure(train_split['text'])
    nonmember_signals = target_signal.measure(test_split["text"])

    shadow_member_signals = shadow_signals[0].measure(train_split['text'])
    shadow_nonmember_signals = shadow_signals[0].measure(test_split['text'])

    calibrated_member_signal = [ts - ss for ts, ss in zip(member_signals, shadow_member_signals)]
    calibrated_nonmember_signal = [ts - ss for ts, ss in zip(nonmember_signals, shadow_nonmember_signals)]

    if not fast:
        dataset = GeneratedDatasetWrapper(model=target_model, top_k=sampling_args.top_k,
                                                   seq_len=sampling_args.seq_len, top_p=sampling_args.top_p,
                                                   prompted=sampling_args.prompted,
                                                   batch_size=sampling_args.sample_batch_size).load_dataset(N=len(member_signals))
        target_gen_signals = target_signal.measure(dataset)
        shadow_gen_signals = shadow_signals[0].measure(dataset)
        calibrated_gen_signals = [ts - ss for ts, ss in zip(target_gen_signals, shadow_gen_signals)]

        # -- Plotting --
        try:
            plot_log_prob_vs_calibrated_loss([-math.log(x) for x in member_signals], calibrated_member_signal,
                                             [-math.log(x) for x in nonmember_signals], calibrated_nonmember_signal,
                                             [-math.log(x) for x in target_gen_signals], calibrated_gen_signals)
        except:
            plot_log_prob_vs_calibrated_loss([-math.log(-x) for x in member_signals], calibrated_member_signal,
                                             [-math.log(-x) for x in nonmember_signals], calibrated_nonmember_signal,
                                             [-math.log(-x) for x in target_gen_signals], calibrated_gen_signals)
    roc_auc = plot_roc_auc(calibrated_member_signal, calibrated_nonmember_signal, show=show,
                           model_name=target_signal.get_target_model().get_name(),
                           title=f"Shadow ({str(target_signal)})")
    if show_signal:
        plot_signal_hist(calibrated_member_signal, calibrated_nonmember_signal, title=f"Shadow Attack with {str(target_signal)}")
    return roc_auc

def population_attack(train_split,
                      test_split,
                      target_signal,
                      show_signal: bool = True,
                      show=True) -> float:
    """ Measure the target model's signal on the different populations
    """
    member_signals = target_signal.measure(train_split['text'])
    nonmember_signals = target_signal.measure(test_split['text'])

    roc_auc = plot_roc_auc(member_signals, nonmember_signals,
                 title=f"Population Signal={str(target_signal)}"
                       f"N={len(member_signals)+len(nonmember_signals)}",
                           model_name=target_signal.get_target_model().get_name(), show=show)
    if show_signal:
        plot_signal_hist(member_signals, nonmember_signals, title=f"Population ({str(target_signal)})")
    return roc_auc

if __name__ == "__main__":
    args = parse_args()
    dataset_args, sampling_args, other_args = args

    print(dataset_args)
    dataset = load_dataset(dataset_args.dataset_path, name=dataset_args.dataset_mode, cache_dir=dataset_args.cache_dir(),
                           sample_duplication_rate=dataset_args.sample_duplication_rate)
    seed = 42   # changing the seed will require re-computing the cache
    print(f"Done loading dataset!")

    train_split = dataset["train"].select(rnd_idx(len(dataset["train"]), seed=seed)[:other_args.n_train])
    test_split = dataset["test"].select(rnd_idx(len(dataset["test"]), seed=42+seed)[:other_args.n_test])

    print(train_split['text'][0])
    print(test_split['text'][0])

    model_args = ModelArguments(model_name=other_args.target)
    target_model = ModelFactory.load_model(model_args)
    target_model.load(ckpt=target_model.get_available_checkpoints()[-1])

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
        target_signal = signal_class(target_model, sampling_args=sampling_args, enable_caching=True)
        if other_args.attack == "population":
            population_attack(train_split, test_split, target_signal)
        elif other_args.attack == "shadow":
            shadow_model = ModelFactory.load_model(ModelArguments(model_name=other_args.shadow[0]))
            shadow_model.load(ckpt=shadow_model.get_available_checkpoints()[-1])
            shadow_models = [signal_class(shadow_model, sampling_args=sampling_args)]
            shadow_model_attack(train_split, test_split, target_signal, shadow_models, sampling_args, fast=other_args.fast)


