import os
import sys

from torch.distributed import barrier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arguments import TrainingArguments

from src.dataset import GeneratedDatasetWrapper
from src.models.model_factory import ModelFactory

import transformers

from src.arguments import ModelArguments
from src.arguments import SamplingArguments


def parse_args():
    """ This script allows (i) generating text and (ii) deploying NER using multiple GPUs.

        Command:
        CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node 2 --master_port 1234
         examples/multi_gpu_generate_sequences.py --model_name echr2xl_nodp --N 55_000 --sample_batch_size 128
    """
    parser = transformers.HfArgumentParser((ModelArguments, SamplingArguments, TrainingArguments))

    parser.add_argument("--model_names", nargs='+', help="all model names to generate for")
    parser.add_argument("--reset_ner", action='store_true', help="reset ner tagging?")
    parser.add_argument("--all_checkpoints", default=False, action='store_true', help="generate text over all checkpoints")

    model_args, sampling_args, train_args, other_args = parser.parse_args_into_dataclasses()
    return model_args, sampling_args, train_args, other_args

def print_0(to_print):
    if train_args.local_rank == 0:
        print(to_print)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    model_args, sampling_args, train_args, other_args = args

    for model_name in other_args.model_names:
        model_args.model_name = model_name
        target_model = ModelFactory.load_model(model_args, hollow=True)

        ckpts = [None]
        if other_args.all_checkpoints:
            ckpts = target_model.get_available_checkpoints()

        for ckpt in ckpts:
            target_model.load(ckpt=ckpt)
            generated_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model, sampling_args=sampling_args)
            print_0(f"Process {train_args.local_rank}: Before Generation: Found {generated_dataset.size()} elements!")
            total_generate = sampling_args.N - generated_dataset.size()
            generate_per_gpu = max(generated_dataset.size() + total_generate // train_args.world_size, 0)
            sampling_args.N = generate_per_gpu
            generated_dataset = GeneratedDatasetWrapper.from_sampling_args(model=target_model,
                                                                           sampling_args=sampling_args)
            print_0(f"Process {train_args.local_rank}: Generate per GPU: {total_generate // train_args.world_size}")
            generated_dataset.load_dataset()  # generate samples
            print_0(f"Process {train_args.local_rank}: After Generation: Found {generated_dataset.size()} elements!")

            print_0(f"Going to sleep .. ")
            barrier()
            print_0(f"Woke up! Tagging samples!")

            # tag samples on multiple GPU
            num_gpus = train_args.world_size
            splits = [(i/num_gpus, (i+1)/num_gpus) for i in range(num_gpus)]
            generated_dataset.load_entities_multi_gpu(train_args.local_rank, split=splits[train_args.local_rank], reset=other_args.reset_ner)

            barrier()




# ----------------------------------------------------------------------------