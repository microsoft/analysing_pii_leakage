
## Analyzing Leakage of Personally Identifiable Information in Language Models

<p>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-orange">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/1.11-PyTorch-orange">
    </a>
    <a href="https://github.com/pytorch/opacus">
            <img alt="Build" src="https://img.shields.io/badge/1.12-opacus-orange">
    </a>
</p>

Official implementation of the following paper to appear at the
IEEE Symposium on Security and Privacy (S&P '23).


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
$ pip install -r requirements.txt
```

## Usage 

We explain the following functions. The scripts are in the ```./examples``` folder and
run configurations are in the ```./configs``` folder.  
* **Fine-Tune**: Fine-tune a pre-trained LM on a dataset (optionally with DP or scrubbing). 
* **PII Extraction**: Given a pre-trained LM, return a set of PII.

* **PII Reconstruction**: Given a pre-trained LM and a masked sentence, reconstruct the most likely PII candidate for the masked tokens.
* **PII Inference**: Given a pre-trained LM, a masked sentence and a set of PII candidates, choose the most likely candidate.
* **Membership Inference**: Given a pre-trained LM and a set of sentences, determine whether a sentence was part of its training data.


## Fine-Tuning
We demonstrate how to fine-tune a ```GPT-2 small``` ([Huggingface](https://huggingface.co/gpt2)) model on the [ECHR](https://huggingface.co/datasets/ecthr_cases) dataset
(i) without defenses, (ii) with scrubbing and (iii) with differential private training (ε=8).

**No Defense**
```shell
$ python examples/fine-tune.py --config_file ../configs/fine-tune/echr-gpt2-small-undefended.yml
```
**With Scrubbing**

_Note_: All PII will be scrubbed from the dataset. Scrubbing is a one-time operation that requires tagging all PII in the dataset first
which can take many hours depending on your setup. We do not provide tagged datasets. 
```
$ python examples/fine-tune.py --config_file ../configs/fine-tune/echr-gpt2-small-scrubbed.yml
```
**With DP (ε=8.0)**

_Note_: We use the [dp-transformers](https://github.com/microsoft/dp-transformers) wrapper around PyTorch's [opacus](https://github.com/pytorch/opacus) library. 
 ```
$ python examples/fine-tune.py --config_file ../configs/fine-tune/echr-gpt2-small-dp8.yml
```

## Attacks 
Assuming your pre-trained model is located at ```~/echr_undefended``` you can edit the model_ckpt
in the ```../configs/<ATTACK>/echr-gpt2-small-undefended.yml``` file and run the following attacks.

**PII Extraction**

```shell
$ python examples/attack.py --config_file ../configs/pii-extraction/echr-gpt2-small-undefended.yml
``` 
**PII Reconstruction**

```shell
$ python examples/attack.py --config_file ../configs/pii-reconstruction/echr-gpt2-small-undefended.yml
``` 
**PII Inference**

```shell
$ python examples/attack.py --config_file ../configs/pii-inference/echr-gpt2-small-undefended.yml
``` 

## Metrics

To reproduce the Figures in our paper, please run the following command (this may take a while).

**ToDo**

## Bibtex
Please consider citing the following papers if you found our work useful.  
```
@article{lukas2023analyzing,
  title = {Analyzing Leakage of Personally Identifiable Information in Language Models},
  author = {Lukas, Nils and Salem, Ahmed and Sim, Robert and Tople, Shruti and Wutschitz, Lukas and Zanella-B{\'e}guelin, Santiago},
  journal = {2023 IEEE Symposium on Security and Privacy (SP)},
  year = {2023},
}
```