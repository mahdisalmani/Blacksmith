# Blacksmith

## Cite Us

This repository contains code and matrials for our paper "Blacksmith: Fast Adversarial Training of Vision Transformers via a Mixture of Single-step and Multi-step Methods" which is now available on arXiv. Please cite us as below.

```bibtex
@misc{2023blacksmith,
      title={Blacksmith: Fast Adversarial Training of Vision Transformers via a Mixture of Single-step and Multi-step Methods}, 
      author={Mahdi Salmani and Alireza Dehghanpour Farashah and Mohammad Azizmalayeri and Mahdi Amiri and Navid Eslami and Mohammad Taghi Manzuri and Mohammad Hossein Rohban},
      year={2023},
      eprint={2310.18975},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Requirements
You may create a conda environment from `requirements.txt`
```
conda create --name <env> --file requirements.txt
```
## Usage
You may perform Blacksmith Adversarial Training with CIFAR10 and CIFAR100 on model ViT-Base and ViT-Small. For instance, to train on CIFAR10 with ViT-Base:
```
python train.py --heat-rate 0.66 --epsilon 8 --alpha 8 --validation-early-stop --pretrained_vit --dataset CIFAR10 --architecture VIT_BASE --out-dir path/to/results --root-model-dir /path/to/trained/models/
```
