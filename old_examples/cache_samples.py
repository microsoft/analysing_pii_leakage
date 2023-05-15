import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.dataset import GeneratedDatasetWrapper

import transformers

from src.arguments import ModelArguments
from src.arguments import SamplingArguments
from src.models.model_factory import ModelFactory

def parse_args():
    """ This precomputes caching scores for a model.
     """
    parser = transformers.HfArgumentParser((SamplingArguments))

    parser.add_argument("--model_names", nargs='+', type=str, help="target model names")

    sampling_args, other_args = parser.parse_args_into_dataclasses()
    return sampling_args, other_args

if __name__ == "__main__":
    args = parse_args()
    sampling_args, other_args = args

    for model_name in tqdm(other_args.model_names):
        try:
            model_args = ModelArguments(model_name=model_name)
            target_model = ModelFactory.load_model(model_args)

            gen_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)
            gen_dataset.load_dataset(N=sampling_args.N)
        except:
            print(f"Could not load {model_name}")
            pass

