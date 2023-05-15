import ast
import os
import random
import sys
from copy import deepcopy
from typing import List

from presidio_analyzer import RecognizerResult
from tqdm import tqdm

from src.models.language_model import LanguageModel
from src.ner import NERAnalyzer
from src.utils import print_highlighted

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import transformers
from datasets import load_dataset
from matplotlib import pyplot as plt

from src.arguments.dataset_args import DatasetArgs
from src.arguments import ModelArguments
from src.arguments import SamplingArguments
from src.models.model_factory import ModelFactory
from src.models.util import rnd_idx


def parse_args():
    """ This script evaluates the risk of PII reconstruction.
     """
    parser = transformers.HfArgumentParser((DatasetArgs, SamplingArguments))

    parser.add_argument("--target_models", nargs='+', help="target model names")
    parser.add_argument("--n_train", type=int, default=50_000, help="number of members in the target data")
    parser.add_argument("--attack", type=str, default="ours",
                        choices=["tab", "ours"], help="which attack to choose")

    dataset_args, sampling_args, other_args = parser.parse_args_into_dataclasses()
    return dataset_args, sampling_args, other_args

def anonymize(sample: str, target_entity: str, all_entities: List[str], replacement=""):
    """ This function anonymizes a sample and replaces the !first! mention of a target
    entity with '[MASK]' and mentions of all other entities with 'someone'
    """
    copied_sample = deepcopy(sample)
    copied_sample = copied_sample.replace(target_entity, "__")  # replace all mentions of target entity with mask

    for other_entity in all_entities:
        copied_sample = copied_sample.replace(other_entity, replacement)
    try:
        prefix_idx = copied_sample.index("__")
    except:
        print_highlighted(f"[ERROR] {copied_sample}, {target_entity}")
        exit()
    prefix = copied_sample[:max(0, prefix_idx - 1)]  # account for whitespace
    suffix = copied_sample[max(0, prefix_idx + len("__")):].replace("[MASK]", replacement)
    if len(prefix) > 0:
        anon = f"{prefix} [MASK]{suffix}"
    else:
        anon = f"[MASK]{suffix}"
    return anon, prefix, suffix

class MatchCounter:
    def __init__(self):
        self.corrects = 0
        self.total = 0
        self.log = []

    def precision(self):
        if self.total == 0:
            return 0
        return self.corrects / self.total

    def tick(self, match: bool = False, entity = None, seq = None):
        if match:
            self.corrects += 1
        self.total += 1
        if entity is not None and seq is not None:
            self.log.append({'entity': entity, 'seq': seq, 'match': match, 'precision': self.precision()})

def tab_attack(train_split,
               target_model: LanguageModel,
               entity_type='person') -> MatchCounter:
    """ Takes the top-1 reconstructed sample based on the prefix (TAB-attack).
    """
    matches = MatchCounter()
    with tqdm(train_split) as pbar:
        for data in pbar:
            sample, entities = data['text'], ast.literal_eval(data[entity_type])
            entities = list(set(entities))

            for target_entity in entities:
                anon, prefix, suffix = anonymize(sample, target_entity, entities)
                if len(prefix) < 10 or len(prefix) > 2500:
                    continue
                gen_seq = target_model.autocomplete(prefix, 20)

                if f" {target_entity} " in gen_seq:    # check for verbatim match (Problem: this matches substrings, which overrepresent matches by tab)
                    matches.tick(True, entity=target_entity, seq=sample)
                    print_highlighted(f"Leaked '{target_entity}' in '{prefix}'")
                else:
                    matches.tick(False, entity=target_entity, seq=sample)
                pbar.set_description(f"(TAB, {target_model.get_name()}) Precision: {100 * matches.precision():.2f}% ({matches.corrects}/{matches.total})")
    return matches

def anonymize_bert(nlp, sample, entities):
    # replace all entities with mask token
    masked_sample = deepcopy(sample)
    for entity in entities:
        masked_sample = masked_sample.replace(entity, "__")
    masked_sample = masked_sample.replace("__", nlp.tokenizer.mask_token)

    for _ in range(masked_sample.count(nlp.tokenizer.mask_token)):
        first_mask_idx = masked_sample.index(nlp.tokenizer.mask_token)

        if masked_sample.count(nlp.tokenizer.mask_token) >= 2:
            second_mask_idx = masked_sample[first_mask_idx + len(nlp.tokenizer.mask_token) - 1:].index(
                nlp.tokenizer.mask_token) + (first_mask_idx + len(nlp.tokenizer.mask_token) - 1)
            # print_highlighted(masked_sample[:second_mask_idx])
            out = nlp(masked_sample[:second_mask_idx], top_k=1)
            masked_sample = out[0]['sequence'] + masked_sample[second_mask_idx:]
        else:  # go to the end
            masked_sample = nlp(masked_sample, top_k=1)[0]['sequence']
            break
    return masked_sample

def ours_attack(train_split,
                target_model: LanguageModel,
                public_model: LanguageModel,
                entity_type='person') -> MatchCounter:
    """ Here we reconstruct PII using (i) candidate generation and (ii) suffix
    """

    matches = MatchCounter()
    matches2 = MatchCounter()    # count how often it was in candidates, but didnt match
    ner_analyzer = NERAnalyzer()

    from transformers import pipeline
    nlp = pipeline('fill-mask')

    with tqdm(train_split) as pbar:
        for data in pbar:
            sample, entities = data['text'], ast.literal_eval(data[entity_type])
            entities = list(set(entities))

            if len(entities) == 0 or len(sample) > 1500:
                continue

            for target_entity in entities:
                pbar.set_description(
                    f"(Ours, {target_model.get_name()}) Precision: {100 * matches.precision():.2f}% ({matches.corrects}/{matches.total})."
                    f"Precision_Candidates={100 * matches2.precision():.2f}%")
                #anon, prefix, suffix = anonymize(sample, target_entity, entities)

                anon = anonymize_bert(nlp, sample.replace(target_entity, "**", 1), entities)
                anon = anon.replace("**", "[MASK]")
                prefix = anon[:anon.index("[MASK]")-1]
                if len(prefix) < 10 or len(prefix) > 2_500:
                    print(f"Skipping because len(prefix)={len(prefix)}")
                    continue

                #print_highlighted(f"{target_entity} --- {anon} ")
                #print_highlighted(prefix)
                #print_highlighted(sample)

                # 1.) generate sequences from the prefix
                gen_seqs = target_model.generate(N=64, input_prompt=prefix, top_k=5, seq_len=20)
                gen_seqs += [target_model.autocomplete(prefix, 20)]

                # idea let public model autocomplete.
                # subtract specificty

                # 2.) extract unique candidate PII
                candidates = []
                for seq in gen_seqs:
                    results: List[RecognizerResult] = ner_analyzer.analyze(seq, entity_classes=["PERSON"])
                    candidates += [seq[r.start:r.end] for r in results]
                candidates = list(set(candidates))

                print_highlighted(f"{target_entity} {target_entity in candidates} ||||| {candidates} ||||| {prefix}")

                # 3.) check if true entity is in candidates. count as negative if not
                if target_entity not in candidates:
                    matches.tick(False, entity=target_entity, seq=sample)
                    matches2.tick(False, entity=target_entity, seq=sample)
                    continue
                matches2.tick(True, entity=target_entity, seq=sample)

                correct_idx = candidates.index(target_entity)
                unmasked_seq = [anon.replace("[MASK]", candidate) for candidate in candidates]

                ppls = [float(x.item()) for x in
                        target_model.perplexity(unmasked_seq, apply_exp=False, return_as_list=True, verbose=False, norm=False)]
                if np.argmin(ppls) == correct_idx:
                    matches.tick(True, entity=target_entity, seq=sample)
                    print_highlighted(f"Leaked {target_entity} in '{anon}'")
                else:
                    matches.tick(False, entity=target_entity, seq=sample)
                    print_highlighted(f"No Leakage! {target_entity} lost to {candidates[np.argmin(ppls)]}")

    return matches


def plot_likelihood_prefix_histogram(train_split,
                                     target_models: List[LanguageModel],
                                     entity_type: str = "persons"):
    """ Estimate leakage through TAB attack.
    """
    ctr = 0
    fn = f"reconstruction_tab_{random.randint(0, 9**9)}.pdf"
    with tqdm(train_split) as pbar:
        loss_dict = {}
        for data in pbar:
            sample, entities = data['text'], ast.literal_eval(data[entity_type])
            entities: List[str] = list(set(entities))

            if len(entities) == 0:  # skip if there are no PII
                continue

            # Step 2: Compute likelihood of generating entity on sample
            # For now, only compute likelihood for the FIRST PII
            for entity in entities[:1]:
                for target_model in target_models:
                    score = target_model.substring_perplexity(sample, substring=entity)
                    loss_dict[target_model.get_name()] = loss_dict.setdefault(target_model.get_name(), []) + [score]
                ctr += 1

            if ctr % 100 == 0:
                for target_model_name, losses in loss_dict.items():
                    values, base = np.histogram(losses, bins=100)
                    cumulative = np.cumsum(values)
                    plt.plot(base[:-1], cumulative / len(losses), label=f"{target_model_name}")
                plt.title("Risk of PII Reconstruction")
                plt.grid()
                plt.xlabel("Perplexity")
                plt.ylabel("Proportion of Training Data")
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()
                plt.savefig(fn)
                print(f"Saved '{os.path.abspath(fn)}'")
                plt.show()


if __name__ == "__main__":
    args = parse_args()
    dataset_args, sampling_args, other_args = args

    dataset = load_dataset(dataset_args.dataset_path,
                           name=dataset_args.dataset_mode,
                           cache_dir=dataset_args.cache_dir(),
                           sample_duplication_rate=dataset_args.sample_duplication_rate)
    seed = 42  # changing the seed may require re-computing the cache
    train_split = dataset["train"].select(rnd_idx(len(dataset["train"]), seed=seed)[:other_args.n_train])

    target_models: List[LanguageModel] = []
    public_models: List[LanguageModel] = []
    for target_model_name in other_args.target_models:
        model_args = ModelArguments(model_name=target_model_name)
        target_models += [ModelFactory.load_model(model_args)]
        print(f"Loading Model {target_model_name} with {target_models[-1].model.num_parameters()} params!")
        public_models += [ModelFactory.load_model(deepcopy(model_args).make_public())]

    for target_model, public_model in zip(target_models, public_models):
        if other_args.attack.lower() == "ours":
            ours_attack(train_split, target_model, public_model)
        elif other_args.attack.lower() == "tab":
            tab_attack(train_split, target_model)
        else:
            raise ValueError(other_args.attack.lower())


