import argparse
import copy
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from architectures.deit import deit_tiny_patch16_224
from architectures.vit import vit_base_patch16_224_in21k

from utils.data_utils import CIFAR10Utils, CIFAR100Utils
from utils.attack_utils import AttackUtils
from utils.utils import TrapezoidLR

from autoattack import AutoAttack

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../', type=str)
parser.add_argument('--epsilon', default=16, type=int,
                    help='''Epsilon to be used at test time (only for final model,
                       if computing loss during training epsilon train is used).
                       If set to None, default, the same args.epsilon will be used for test and train.''')
parser.add_argument('--model-path', type=str, default='/kaggle/input/vit-n-fgsm/ex1/ex1/model.pth')
parser.add_argument('--pretrain-pos-only', action='store_true')
parser.add_argument('--patch', type=int, default=4)

args = parser.parse_args()


class MyModel(torch.nn.Module):
    def __init__(self, model, mu, std):
        super().__init__()
        self.mu = mu
        self.sigma = std
        self.model = model

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.model(x)


def _pgd_whitebox(model, X, y, adversary):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = adversary.run_standard_evaluation(X, y, bs=X.size(0))
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader, adverary):
    model.model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, adverary)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print("robust_err_total: ", robust_err_total)
    print("natural_err_total: ", natural_err_total)


if __name__ == '__main__':
    model_test = vit_base_patch16_224_in21k(pretrained=False,
                                            img_size=32,
                                            pretrain_pos_only=False,
                                            patch_size=4, num_classes=100, args=args).cuda()

    model_test.load_state_dict(torch.load(args.model_path))
    model_test.float()
    model_test.eval()

    data_utils = CIFAR100Utils(normalize=False)
    attack_utils = AttackUtils(data_utils.lower_limit, data_utils.upper_limit, data_utils.std)
    (train_loader, test_loader, robust_test_loader,
     valid_loader, train_idx, valid_idx) = data_utils.get_indexed_loaders(args.data_dir,
                                                                          128,
                                                                          valid_size=0,
                                                                          robust_test_size=-1)

    dset_mean =  (0.5071, 0.4865, 0.4409)
    dset_std = (0.2673, 0.2564, 0.2762)

    mu = torch.tensor(dset_mean).view(3, 1, 1).cuda()
    std = torch.tensor(dset_std).view(3, 1, 1).cuda()
    my_model = MyModel(model_test, mu, std)

    print("start evaluating...")
    test_loss, test_acc = attack_utils.evaluate_standard(test_loader, my_model)
    print("test acc: ", test_acc)
    epsilon = args.epsilon / 255.0
    open('./attack_logs.out', 'w').close()
    adversary = AutoAttack(my_model, norm='Linf', eps=epsilon, version='standard',
                           log_path="./attack_logs.out")
    adversary.seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_adv_test_whitebox(my_model, device, test_loader, adversary)
