import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg


class TINYIMAGENETUtils(object):
    def __init__(self, normalize=True):
        if normalize:
            self.tiny_mean = (0.4802, 0.4481, 0.3975)
            self.tiny_std = (0.2302, 0.2265, 0.2262)
        else:
            self.tiny_mean = (0., 0., 0.)
            self.tiny_std = (1., 1., 1.)

        self.mu = torch.tensor(self.tiny_mean).view(3, 1, 1).cuda()
        self.std = torch.tensor(self.tiny_std).view(3, 1, 1).cuda()

        self.upper_limit = ((1 - self.mu) / self.std)
        self.lower_limit = ((0 - self.mu) / self.std)

    def get_indexed_loaders(self, dir_, batch_size=128, batch_size_test=None, shuffle=True, valid_size=1000,
                            robust_test_size=-1):
        if batch_size_test is None:
            batch_size_test = batch_size

        train_dataset, val_dataset, test_dataset = load_tiny_data(dir_)
        self.img_size = (64, 64)
        self.max_label = 199
        num_workers = 0
        num_train = len(train_dataset)
        indices = list(range(num_train))
        if shuffle:
            np.random.shuffle(indices)
        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
        train_sampler = SubsetRandomSampler(train_idx)
        if valid_size > 0:
            valid_sampler = SubsetRandomSampler(valid_idx)

        assert robust_test_size <= len(test_dataset)
        if robust_test_size < 0:
            robust_test_size = len(test_dataset)
        robust_test_sampler = SubsetRandomSampler(list(range(robust_test_size)))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            sampler=train_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size_test,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
        )
        if valid_size > 0:
            valid_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=batch_size_test,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                sampler=valid_sampler
            )
        else:
            valid_loader = None

        robust_test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size_test,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            sampler=robust_test_sampler
        )
        return train_loader, test_loader, robust_test_loader, valid_loader, train_idx, valid_idx


class TinyImageNet(VisionDataset):
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=True, indexed=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))
        self.indexed = indexed

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = self.make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        image, target = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.indexed:
            return image, target, index
        return image, target

    def __len__(self):
        return len(self.data)

    def make_dataset(self, root, base_folder, dirname, class_to_idx):
        images = []
        dir_path = os.path.join(root, base_folder, dirname)

        if dirname == 'train':
            for fname in sorted(os.listdir(dir_path)):
                cls_fpath = os.path.join(dir_path, fname)
                if os.path.isdir(cls_fpath):
                    cls_imgs_path = os.path.join(cls_fpath, 'images')
                    for imgname in sorted(os.listdir(cls_imgs_path)):
                        path = os.path.join(cls_imgs_path, imgname)
                        image = self.loader(path)
                        item = (image, class_to_idx[fname])
                        images.append(item)
        else:
            imgs_path = os.path.join(dir_path, 'images')
            imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split('\t'), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                image = self.loader(path)
                item = (image, class_to_idx[cls_map[imgname]])
                images.append(item)

        return images


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def load_tiny_data(dir="./tiny-imagenet"):
    print("Loading Tiny-ImageNet data from ")
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = TinyImageNet(dir, split='train', download=True, transform=train_transform,
                                 indexed=True)
    val_dataset = TinyImageNet(dir, split='train', download=True, transform=val_transform)
    test_dataset = TinyImageNet(dir, split='val', download=False, transform=val_transform)

    return train_dataset, val_dataset, test_dataset
