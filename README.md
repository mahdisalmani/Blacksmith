# blacksmith

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
