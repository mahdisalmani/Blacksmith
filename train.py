import argparse
import copy
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from architectures.preact_resnet import PreActResNet18
from architectures.wide_resnet import Wide_ResNet
from architectures.deit import deit_tiny_patch16_224
from architectures.vit import vit_base_patch16_224_in21k

from utils.data_utils import CIFAR10Utils, CIFAR100Utils
from utils.attack_utils import AttackUtils

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Architecture settings
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='One of: CIFAR10, CIFAR100')
    parser.add_argument('--architecture', default='PreActResNet18', type=str)


    # Vision transform
    parser.add_argument('--pretrained_vit', default=False, action='store_true',
                        help='Use pretrained vision transformer or not')
    parser.add_argument('--pretrain-pos-only', action='store_true')
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--vit-depth', type=int, default=12)

    # Wide resnet settings (in case of using wideresnet)
    parser.add_argument('--wide_resnet_depth', default=28, type=int, help='WideResNet depth')
    parser.add_argument('--wide_resnet_width', default=10, type=int, help='WideResNet width')
    parser.add_argument('--wide_resnet_dropout_rate', default=0.3, type=float, help='WideResNet dropout rate')

    # Training data settings
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)


    # Learning rate settings
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--lr-decay-milestones', type=int, nargs='+', default=[15, 18])
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    # Method settings
    parser.add_argument('--method', type=str, default='blacksmith', choices=['blacksmith', 'pgd', 'fgsm'])
    
    # Blacksmith settings
    parser.add_argument('--lr-max-heat', default=0.2, type=float)

    # PGD training settings
    parser.add_argument('--attack-iters', type=int, default=2) 
    parser.add_argument('--pgd-alpha', type=float, default=-1.0)

    # Adversarial training settings
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--unif', default='1.0', type=float,
                        help='''Magnitude of the uniform noise relative to epsilon.
                                - k -> U(-k*eps, k*eps),
                                - 0 -> No noise,
                                - Default is 1 -> U(-1eps, 1eps).
                                In NFGSM it would be set to 2.
                        ''')
    parser.add_argument('--clip', default=1, type=float,
                        help='''Radius of the inf ball where to clip the perturbations.
                                Relative to epsilon: i.e. 1 means clip(-eps, eps).
                                By default it is set to -1 (no clipping)
                                In Fast Adv Training it would be set to 1.
                                In NFGSM it would be set to -1.
                        ''')
    parser.add_argument('--validation-early-stop', action='store_true',
                        help='Store best epoch via validation')
    


    # Evaluation settings
    parser.add_argument('--robust_test_size', default=-1, type=int,
                        help='Number of samples to be used for robust testing, Default: -1 will use all samples')
    parser.add_argument('--epsilon_test', default=None, type=int,
                        help='''Epsilon to be used at test time (only for final model,
                        if computing loss during training epsilon train is used).
                        If set to None, default, the same args.epsilon will be used for test and train.''')
    parser.add_argument('--pgd-attack-iters', type=int, default=30)
    parser.add_argument('--attack-restarts', type=int, default=3)


    # Config paths
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--out-dir', default='/path/to/results/',
                        type=str, help='Output directory')
    parser.add_argument('--root-model-dir',
                        default='/path/to/trained/models/',
                        type=str, help='Models directory')

    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    print('Defining data object')
    num_classes = 10
    if args.dataset.upper() == 'CIFAR10':
        data_utils = CIFAR10Utils()
    elif args.dataset.upper() == 'CIFAR100':
        data_utils = CIFAR100Utils()
        num_classes = 100
    else:
        raise ValueError('Unsupported dataset.')

    # If args.epsilon_test is None, use the same epsilon than during training.
    if args.epsilon_test is None:
        args.epsilon_test = args.epsilon

    print('Defining attack object')
    attack_utils = AttackUtils(data_utils.lower_limit, data_utils.upper_limit, data_utils.std)

    # Set-up results paths
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    model_path = os.path.join(args.root_model_dir, args.out_dir.split('/')[-1])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.txt'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.validation_early_stop:
        best_pgd_val_acc = 0
        valid_size = 1000
    else:
        valid_size = 0

    (train_loader, test_loader, robust_test_loader,
     valid_loader, train_idx, valid_idx) = data_utils.get_indexed_loaders(args.data_dir,
                                                                          args.batch_size,
                                                                          valid_size=valid_size,
                                                                          robust_test_size=args.robust_test_size)

    # Making sure that data is in the supported format
    if (data_utils.img_size != (32, 32)):
        raise RuntimeError('Data is not in the supported format input image size (32x32)')

    # Adv training and test settings
    epsilon = (args.epsilon / 255.) / data_utils.std
    alpha = (args.alpha / 255.) / data_utils.std

    # Set pgd_alpha relative to alpha
    pgd_alpha = args.pgd_alpha * alpha


    if args.pgd_alpha == -1.0:
        pgd_alpha = (max(args.alpha / args.attack_iters, 2.) / 255.) / data_utils.std

    # Define architecture
    args.num_classes = data_utils.max_label + 1  # Labels start from 0
    if args.architecture.upper() == 'PREACTRESNET18':
        model = PreActResNet18(num_classes=args.num_classes).cuda()

    elif args.architecture.upper() in 'WIDERESNET':
        model = Wide_ResNet(args.wide_resnet_depth,
                            args.wide_resnet_width,
                            args.wide_resnet_dropout_rate,
                            num_classes=args.num_classes).cuda()

    elif args.architecture.upper() == 'VIT_BASE':
        model = vit_base_patch16_224_in21k(pretrained=args.pretrained_vit,
                                           img_size=32,
                                           pretrain_pos_only=args.pretrain_pos_only,
                                           patch_size=args.patch, num_classes=num_classes, args=args).cuda()
    elif args.architecture.upper() == 'DEIT_TINY':
        model = deit_tiny_patch16_224(pretrained=args.pretrained_vit,
                                      img_size=32,
                                      pretrain_pos_only=args.pretrain_pos_only,
                                      patch_size=args.patch, num_classes=num_classes, args=args).cuda()
    else:
        raise ValueError('Unknown architecture.')

    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)

    elif args.lr_schedule == 'multistep':
        steps_per_epoch = len(train_loader)
        milestones = list(np.array(args.lr_decay_milestones) * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    start_train_time = time.time()
    if args.validation_early_stop:
        val_acc_hist = []
        robust_val_acc_hist = []

    print('Start training')
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    train_steps = 0

    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y, batch_idx) in enumerate(tqdm(train_loader)):
            # rate = (total_steps - train_steps) / total_steps
            rate = 0.5
            X, y = X.cuda(), y.cuda()
            eta = torch.zeros_like(X).cuda()
            if args.unif > 0:
                for j in range(len(epsilon)):
                    eta[:, j, :, :].uniform_(-args.unif * epsilon[j][0][0].item(),
                                             args.unif * epsilon[j][0][0].item())
                eta = attack_utils.clamp(eta, attack_utils.lower_limit - X, attack_utils.upper_limit - X)

            
            if args.method == 'blacksmith':
                p = 1 if np.random.random() > rate else 0
                end = args.vit_depth if p == 1 else int(rate * args.vit_depth)
                start = 0 if p == 0 else int(rate * args.vit_depth)
                steps = 1 if p == 1 else 2

                for j in range(steps):
                    eta.requires_grad = True
                    output = model(X + eta, end=end)
                    loss = F.cross_entropy(output, y)
                    grad = torch.autograd.grad(loss, eta)[0].detach()
                    delta = attack_utils.clamp(eta + (alpha / steps) * torch.sign(grad), -epsilon, epsilon)
                    delta = attack_utils.clamp(delta, attack_utils.lower_limit - X, attack_utils.upper_limit - X)
                    eta = delta.detach()
                
                model.freeze_except(start=start, end=end)
                delta = delta.detach()
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                opt.zero_grad()
                loss.backward()
                
                if args.architecture.upper() == "VIT_BASE":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                elif args.architecture.upper() == "DEIT_TINY":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
                opt.step()
                scheduler.step()

                model.freeze_except()

            elif args.method == 'pgd':
                eta.requires_grad = True
                for _ in range(args.attack_iters):
                    output = model(X + eta)
                    loss = F.cross_entropy(output, y)
                    grad = torch.autograd.grad(loss, eta)[0].detach()
                    eta.data = attack_utils.clamp(eta + pgd_alpha * torch.sign(grad), -epsilon, epsilon)
                    eta.data = attack_utils.clamp(eta, attack_utils.lower_limit - X, attack_utils.upper_limit - X)

                eta = eta.detach()
                output = model(X + eta)
                loss = F.cross_entropy(output, y)
                opt.zero_grad()
                loss.backward()
                
                if args.architecture.upper() == "VIT_BASE":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                elif args.architecture.upper() == "DEIT_TINY":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
                opt.step()
                scheduler.step()

            elif args.method == 'fgsm':
                eta.requires_grad = True
                output = model(X + eta)
                loss = F.cross_entropy(output, y)
                grad = torch.autograd.grad(loss, eta)[0].detach()
                if args.clip > 0:
                    eta.data = attack_utils.clamp(eta + alpha * torch.sign(grad), -epsilon, epsilon)
                else:
                    eta.data = eta + alpha * torch.sign(grad)
                eta.data = attack_utils.clamp(eta, attack_utils.lower_limit - X, attack_utils.upper_limit - X)

                eta = eta.detach()
                output = model(X + eta)
                loss = F.cross_entropy(output, y)
                opt.zero_grad()
                loss.backward()
            
                if args.architecture.upper() == "VIT_BASE":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                elif args.architecture.upper() == "DEIT_TINY":
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
                opt.step()
                scheduler.step()
            else:
                raise ValueError

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            train_steps += 1


        if args.validation_early_stop:
            pgd_loss, pgd_acc = attack_utils.evaluate_pgd(valid_loader, model, 10, 1, epsilon=args.epsilon)
            test_loss, test_acc = attack_utils.evaluate_standard(valid_loader, model)
            print("Validation pgd10 test-acc, pgd-acc")
            print(test_acc, pgd_acc)

            model.train()
            if pgd_acc >= best_pgd_val_acc:
                best_pgd_val_acc = pgd_acc
                best_state_dict = copy.deepcopy(model.state_dict())
                val_acc_hist.append(test_acc)
                robust_val_acc_hist.append(pgd_acc)

        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f', epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)
        print(epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)

    train_time = time.time()
    if args.validation_early_stop:
        torch.save(best_state_dict, os.path.join(model_path, f'best_model.pth'))

    final_state_dict = model.state_dict()
    torch.save(final_state_dict, os.path.join(model_path, 'model.pth'))

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

    if args.validation_early_stop:
        np.save(os.path.join(model_path, 'val_acc_hist.npy'), val_acc_hist)
        np.save(os.path.join(model_path, 'robust_val_acc.npy'), robust_val_acc_hist)

    if args.robust_test_size != 0:
        print('Training finished, starting evaluation')
        args.num_classes = data_utils.max_label + 1
        if args.architecture.upper() == 'PREACTRESNET18':
            model_test = PreActResNet18(num_classes=args.num_classes).cuda()
        elif args.architecture.upper() in 'WIDERESNET':
            model_test = Wide_ResNet(args.wide_resnet_depth,
                                     args.wide_resnet_width,
                                     args.wide_resnet_dropout_rate,
                                     num_classes=args.num_classes).cuda()
        elif args.architecture.upper() == 'VIT_BASE':
            model_test = vit_base_patch16_224_in21k(pretrained=args.pretrained_vit,
                                                    img_size=32,
                                                    pretrain_pos_only=args.pretrain_pos_only,
                                                    patch_size=args.patch, num_classes=num_classes, args=args).cuda()
        elif args.architecture.upper() == 'DEIT_TINY':
            model_test = deit_tiny_patch16_224(pretrained=args.pretrained_vit,
                                               img_size=32,
                                               pretrain_pos_only=args.pretrain_pos_only,
                                               patch_size=args.patch, num_classes=num_classes, args=args).cuda()

        
        model_test.load_state_dict(final_state_dict)
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = attack_utils.evaluate_pgd(robust_test_loader, model_test, args.pgd_attack_iters,
                                                      args.attack_restarts, epsilon=args.epsilon_test)
        test_loss, test_acc = attack_utils.evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        print(test_loss, test_acc, pgd_loss, pgd_acc)
        print('Evaluating final model finished')


if __name__ == "__main__":
    main()