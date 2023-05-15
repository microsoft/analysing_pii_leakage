import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data


def intersection(a: dict, b: dict):
    intersect1d = {}
    for entity_type in a.keys():
        intersect1d[entity_type] = [x for x in a[entity_type] if x in b[entity_type]]
    return intersect1d

def difference(a: dict, b: dict):
    diff1d = {}
    for entity_type in a.keys():
        diff1d[entity_type] = [x for x in a[entity_type] if x not in b[entity_type]]
    return diff1d

def union(a: dict, b: dict):
    union1d = {}
    for entity_type in list(set(list(a.keys()) + list(b.keys()))):
        union1d[entity_type] = [x for x in a.setdefault(entity_type, [])] + [x for x in b.setdefault(entity_type, [])]
    return union1d

def fetch_tensor_from_dataset(dataset):
    """ Converts a dataset to a tensor """
    loader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(loader))

def _compute_average_sentence_length(dataset):
    """ Computes average length of each element in a dataset """
    ctr = 0
    for sentence in dataset:
        ctr += len(sentence[0])
    return {ctr / len(dataset)}

def plot_tokens_in_dataset_cdf(model, dataset):
    tokens, = model.tokenize_datasets([dataset["train"]])
    l = [len(x["input_ids"]) for x in tokens]
    plt.title("TAB Number of Tokens")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Density")
    plt.grid()
    plt.xlim(0, 1.1 * max(l))
    count, bins_count = np.histogram(l, bins=70)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()

def truncate_dataset(model, dataset):
    """ Truncates sequences in a dataset to the maximum length of a model """
    tokenized_train, tokenized_test = model.tokenize_datasets([dataset["train"], dataset["test"]])
    return tokenized_train, tokenized_train