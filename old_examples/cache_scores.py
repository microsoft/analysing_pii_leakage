import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from datasets import load_dataset

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.arguments import SamplingArguments
from src.attacks.mia import BLEUSignal, ROUGE1Signal, ROUGE2Signal, ROUGELsumSignal, BERTScoreSignal, \
    FrugalScoreSignal
from src.models.model_factory import ModelFactory
from src.models.util import rnd_idx

def parse_args():
    """ This precomputes caching scores for a model.
     """
    parser = transformers.HfArgumentParser((DatasetArgs, SamplingArguments))

    parser.add_argument("--model_names", nargs='+', type=str, help="target model names")

    parser.add_argument("--n_train", type=int, default=500, help="number of members in the target data")
    parser.add_argument("--n_test", type=int, default=500, help="number of non-members in the target data")

    dataset_args, sampling_args, other_args = parser.parse_args_into_dataclasses()
    return dataset_args, sampling_args, other_args

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


    for model_name in other_args.model_names:
        try:
            model_args = ModelArguments(model_name=model_name)
            target_model = ModelFactory.load_model(model_args)

            print(f"Perplexity of model: {target_model.perplexity(test_split)}")

            signals_classes = {
                # "ppl": PPLSignal, # not cachable
                "bleu": BLEUSignal,
                "rouge1": ROUGE1Signal,
                "rouge2": ROUGE2Signal,
                "rougesum": ROUGELsumSignal,
                "frugal": FrugalScoreSignal,    # 0:13 min for 100 samples (at N=100)
                "bert": BERTScoreSignal         # 1:13 min for 100 samples (at N=100)
            }

            for signal_cls in signals_classes.values():
                target_signal = signal_cls(target_model, sampling_args=sampling_args, enable_caching=True)
                target_signal.measure(train_split['text'])
        except:
            print(f"Could not load {model_name}")
            pass

