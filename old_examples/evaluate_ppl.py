import os
import sys
import random
from typing import List

import pandas
import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.models.model_factory import ModelFactory
from src.models.language_model import LanguageModel
from src.models.util import rnd_idx
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers

from src.utils import print_highlighted


def parse_args():
    """ This script evaluates the perplexity of a set of models  """
    parser = transformers.HfArgumentParser((DatasetArgs,))

    parser.add_argument("--target_models", nargs='+', help="model names to compute ppl over")
    parser.add_argument("--n_train", type=int, default=1000, help="number of members in the target data")

    dataset_args, other_args = parser.parse_args_into_dataclasses()
    return dataset_args, other_args

def mname_to_dname(model_name):
     name_dict = {
         "echr2_public": r"Public Model",
         "echr2_dp8": r"GPT2-s, $\epsilon=8$",
         "echr2m_dp8": r"GPT2-m, $\epsilon=8$",
         "echr2l_dp8": r"GPT2-l, $\epsilon=8$",
         "echr2xl_dp8": r"GPT2-xl, $\epsilon=8$",

         "echr2_nodp": "GPT2-s",
         "echr2m_nodp": "GPT2-m",
         "echr2l_nodp": "GPT2-l",
         "echr2xl_nodp": "GPT2-xl",

         "echrxl_nodp_scrubbed": "GPT2-xl (scrub)",
         "echrl_nodp_scrubbed": "GPT2-l (scrub)",
         "echrm_nodp_scrubbed": "GPT2-m (scrub)",
         "echr_nodp_scrubbed": "GPT2-s (scrub)",

         "yelpxl_nodp_scrubbed": "GPT2-xl (scrub)",
         "yelpl_nodp_scrubbed": "GPT2-l (scrub)",
         "yelpm_nodp_scrubbed": "GPT2-m (scrub)",
         "yelp_nodp_scrubbed": "GPT2-s (scrub)",

         "yelp_dp8": r"GPT2-s, $\epsilon=8$",
         "yelpm_dp8": r"GPT2-m, $\epsilon=8$",
         "yelpl_dp8": r"GPT2-l, $\epsilon=8$",
         "yelpxl_dp8": r"GPT2-xl, $\epsilon=8$",

         "yelp_nodp": "GPT2-s",
         "yelpm_nodp": "GPT2-m",
         "yelpl_nodp": "GPT2-l",
         "yelpxl_nodp": "GPT2-xl",
     }
     return name_dict.setdefault(model_name, model_name)

def perplexity_scatter(
                    train_split,
                    test_split,
                    target_models: List[LanguageModel]):
    """ Computes a scatter plot for the perplexity comparing training and testing perplexities
    """
    loss_dict: dict = {}
    for model in target_models:
        loss_dict['x'] = loss_dict.setdefault('x', []) + [model.perplexity(train_split["text"])]
        loss_dict['y'] = loss_dict.setdefault('y', []) + [model.perplexity(test_split["text"])]
        loss_dict['Millions of Parameters'] = loss_dict.setdefault('Millions of Parameters', []) + \
                                              [model.model.num_parameters() // 10 ** 6]
        loss_dict['name'] = loss_dict.setdefault('name', []) + [mname_to_dname(model.get_name())]
    print(loss_dict)

    fp = f"perplexity_scatter_{random.randint(0, 9**9)}.pdf"

    df = pandas.DataFrame.from_dict(loss_dict)
    sns.scatterplot(x='x', y='y', size='Millions of Parameters', data=df,
                    sizes=(124, 300))
    plt.plot([df['x'].min(), df['x'].max()], [df['x'].min(), df['x'].max()], linestyle='--')
    plt.grid()
    plt.title('Training vs Testing Perplexity on Enron')
    plt.xlabel('Training Perplexity')
    plt.ylabel('Testing Perplexity')

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'] + .02, point['y'], str(point['val']))

    label_point(df.x, df.y, df.name, plt.gca())
    print(f"Saved at {os.path.abspath(fp)}")
    plt.savefig(fp)
    plt.show()


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    print_highlighted(f"Running script with {args}")
    dataset_args, other_args = args

    dataset = load_dataset(dataset_args.dataset_path,
                           name=dataset_args.dataset_mode,
                           cache_dir=dataset_args.cache_dir(),
                           sample_duplication_rate=dataset_args.sample_duplication_rate)
    seed = 42  # changing the seed may require re-computing the cache
    test_split = dataset["test"].select(rnd_idx(len(dataset["test"]), seed=seed)[:other_args.n_train])
    train_split = dataset["train"].select(rnd_idx(len(dataset["train"]), seed=seed)[:other_args.n_train])

    target_models: List[LanguageModel] = []
    for target_model_name in other_args.target_models:
        model_args = ModelArguments(model_name=target_model_name)
        target_models += [ModelFactory.load_model(model_args)]
        print(f"Loading Model {target_model_name} with {target_models[-1].model.num_parameters()} params!")


    perplexity_scatter(train_split, test_split, target_models)



# ----------------------------------------------------------------------------
