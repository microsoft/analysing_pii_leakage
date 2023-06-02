## Analyzing Leakage of Personally Identifiable Information in Language Models

<p>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/1.11-PyTorch-orange">
    </a>
    <a href="https://github.com/pytorch/opacus">
            <img alt="Build" src="https://img.shields.io/badge/1.12-opacus-orange">
    </a>
</p>

This repository contains the official code for our IEEE S&P 2023 paper using GPT-2 language models and
Flair Named Entity Recognition (NER) models.
It allows fine-tuning (i) undefended, (ii) differentially-private and (iii) scrubbed language models
on ECHR and Enron and attacking them using the attacks presented in our paper.


## Publication

> **Analyzing Leakage of Personally Identifiable Information in Language Models.**
> Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz and Santiago Zanella-Béguelin.
> Symposium on Security and Privacy (S&P '23). San Francisco, CA, USA.
>
> [![arXiv](https://img.shields.io/badge/arXiv-2302.00539-green)](https://arxiv.org/abs/2302.00539)


## Build & Run

We recommend setting up a conda environment for this project.
```shell
$ conda create -n pii-leakage python=3.10
$ conda activate pii-leakage
$ pip install -e .
```


## Usage

We explain the following functions. The scripts are in the ```./examples``` folder and
run configurations are in the ```./configs``` folder.
* **Fine-Tune**: Fine-tune a pre-trained LM on a dataset (optionally with DP or scrubbing).
* **PII Extraction**: Given a fine-tuned LM, return a set of PII.
* **PII Reconstruction**: Given a fine-tuned LM and a masked sentence, reconstruct the most likely PII candidate for the masked tokens.
* **PII Inference**: Given a fine-tuned LM, a masked sentence and a set of PII candidates, choose the most likely candidate.


## Fine-Tuning

We demonstrate how to fine-tune a ```GPT-2 small``` ([Huggingface](https://huggingface.co/gpt2)) model on the [ECHR](https://huggingface.co/datasets/ecthr_cases) dataset
(i) without defenses, (ii) with scrubbing and (iii) with differentially private training (ε=8).

**No Defense**
```shell
$ python fine_tune.py --config_path ../configs/fine-tune/echr-gpt2-small-undefended.yml
```

**With Scrubbing**

_Note_: All PII will be scrubbed from the dataset. Scrubbing is a one-time operation that requires tagging all PII in the dataset first
which can take many hours depending on your setup. We do not provide tagged datasets.
```
$ python fine_tune.py --config_path ../configs/fine-tune/echr-gpt2-small-scrubbed.yml
```

**With DP (ε=8.0)**

_Note_: We use the [dp-transformers](https://github.com/microsoft/dp-transformers) wrapper around PyTorch's [opacus](https://github.com/pytorch/opacus) library.
 ```
$ python fine_tune.py --config_path ../configs/fine-tune/echr-gpt2-small-dp8.yml
```


## Attacks

Assuming your fine-tuned model is located at ```../echr_undefended``` run the following attacks.
Otherwise, you can edit the ```model_ckpt``` attribute in the ```../configs/<ATTACK>/echr-gpt2-small-undefended.yml``` file to point to the location of the model.

**PII Extraction**

This will extract PII from the model's generated text.
```shell
$ python extract_pii.py --config_path ../configs/pii-extraction/echr-gpt2-small-undefended.yml
```

**PII Reconstruction**

This will reconstruct PII from the model given a target sequence.
```shell
$ python reconstruct_pii.py --config_path ../configs/pii-reconstruction/echr-gpt2-small-undefended.yml
```

**PII Inference**

This will infer PII from the model given a target sequence and a set of PII candidates.
```shell
$ python reconstruct_pii.py --config_path ../configs/pii-inference/echr-gpt2-small-undefended.yml
```


## Evaluation

Use the ```evaluate.py``` script to evaluate our privacy attacks against the LM.
```shell
$ python evaluate.py --config_path ../configs/evaluate/pii-extraction.yml
```
This will compute the precision/recall for PII extraction and accuracy for PII reconstruction/inference attacks.


## Datasets

The provided ECHR dataset wrapper already tags all PII in the dataset.
The PII tagging is done using the Flair NER modules and can take several hours depending on your setup, but is a one-time operation
that will be cached in subsequent runs.


## Fine-Tuned Models

Unfortunately, we do not provide fine-tuned model checkpoints.
This repository does support loading models remotely, which can be done by providing a URL instead of a local path
in the configuration files for the ```model_ckpt``` attribute.


## Citation

Please consider citing the following paper if you found our work useful.

```
@InProceedings{lukas2023analyzing,
  title      = {Analyzing Leakage of Personally Identifiable Information in Language Models},
  author     = {Lukas, Nils and Salem, Ahmed and Sim, Robert and Tople, Shruti and Wutschitz, Lukas and Zanella-B{\'e}guelin, Santiago},
  booktitle  = {2023 IEEE Symposium on Security and Privacy (SP)},
  year       = {2023},
  publisher  = {IEEE Computer Society},
  pages      = {346-363},
  doi        = {10.1109/SP46215.2023.00154}
}
```
