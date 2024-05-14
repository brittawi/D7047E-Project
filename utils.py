from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import WeightedRandomSampler
from torchvision.utils import make_grid 
import torch

import lightning as L

import numpy as np
import matplotlib.pyplot as plt

TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 1 - TRAIN_FRACTION


def loadData(batchSize, numWorkers, dataDir = "./chest_xray", customSplit = True, useAugment = False, useSampler = False, showAnalytics = False):
    """

    :param batchSize:
    :param numWorkers:
    :param dataDir:
    :param customSplit:
    :param useAugment:
    :param useSampler:
    :param showAnalytics:
    :return: train_loader, val_loader, test_loader
    """
    data_dir_train = dataDir + "/train"
    data_dir_val = dataDir + "/val"
    data_dir_test = dataDir + "/test"

    assert not customSplit or not useAugment, "custom split and augment are mutually exclusive"

    # Some desired transforms for ResNet50
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 
        ]
    )

    dataset_train = datasets.ImageFolder(data_dir_train, transform)
    dataset_val = datasets.ImageFolder(data_dir_val, transform)

    if customSplit:
        generator1 = torch. Generator().manual_seed(42)
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val])
        dataset_train, dataset_val = torch.utils.data.random_split(dataset,
                                                                [TRAIN_FRACTION, VALIDATION_FRACTION],
                                                                generator=generator1)
    elif useAugment:
        # The Effectiveness of Image Augmentation in Deep Learning Networks for Detecting COVID-19: A Geometric Transformation Perspective
        # https://www.frontiersin.org/articles/10.3389/fmed.2021.629134/full
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.NEAREST_EXACT),
            #    transforms.RandomResizedCrop(size=(256, 256), scale=(0.1, 0.9), ratio=(1, 1)),
            transform,
        ])

        dataset_train = datasets.ImageFolder(data_dir_train, train_transform)

    dataset_test = datasets.ImageFolder(data_dir_test, transform)

    if showAnalytics:
        analytics(dataset_train, dataset_val, dataset_test)

    if useSampler:
        labels = extract_targets(dataset_train)
        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in np.unique(labels)])
        
        # use WeightedRandomSampler as dataset is not balanced
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
        train_loader = DataLoader(
            dataset_train, 
            batch_size=batchSize, 
            shuffle=False, # has to be False when using WeightedRandomSampler
            num_workers=numWorkers, 
            persistent_workers= True,
            sampler = sampler
            )
    else:
        train_loader = DataLoader(
            dataset_train, 
            batch_size=batchSize, 
            shuffle=True, 
            num_workers=numWorkers, 
            persistent_workers= True
            )
        
    val_loader = DataLoader(
        dataset_val, 
        batch_size=batchSize, 
        shuffle=False, 
        num_workers=numWorkers, 
        persistent_workers= True
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batchSize, 
        shuffle=False, 
        num_workers=numWorkers, 
        persistent_workers= True
    )

    return train_loader, val_loader, test_loader


def analytics(dataset_train, dataset_val, dataset_test):
    
    # print sizes of datasets
    print(f'Size of train dataset: {len(dataset_train)}')
    print(f'Size of validation dataset: {len(dataset_val)}')
    print(f'Size of test dataset: {len(dataset_test)}')

    # plot distribution of each data set
    labels = ["normal", "pneumonia"]
    labels_num = [0,1]
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))

    # Bar for training
    train_targets = extract_targets(dataset_train)
    pneumonia = np.count_nonzero(train_targets)
    normal = len(train_targets) - pneumonia
    ax1.bar(labels_num, [normal,pneumonia])
    ax1.set_xticks(labels_num, labels)
    ax1.set_title("Distribution of training set")
    ylim = ax1.get_ylim()
    
    # Bar for validation
    val_targets = extract_targets(dataset_val)
    pneumonia = np.count_nonzero(val_targets)
    normal = len(val_targets) - pneumonia
    ax2.bar(labels_num, [normal,pneumonia])
    ax2.set_xticks(labels_num, labels)
    ax2.set_title("Distribution of validation set")
    ax2.set_ylim(ylim)

    # Bar for testing
    test_targets = extract_targets(dataset_test)
    pneumonia = np.count_nonzero(test_targets)
    normal = len(test_targets) - pneumonia
    ax3.bar(labels_num, [normal, pneumonia])
    ax3.set_xticks(labels_num, labels)
    ax3.set_title("Distribution of test set")
    ax3.set_ylim(ylim)


def extract_targets(dataset):
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        return [target for _, target in dataset]
    return dataset.targets


def plotExamples(train_loader):
    examples = next(iter(train_loader))
    images, labels = examples
    grid = make_grid(images[:9], nrow=3)
    plt.imshow(grid.permute(1, 2, 0))

def set_reproducibility(seed=None):
    if seed is None:
        seed = 42
    np.random.seed(seed)
    L.seed_everything(42)
    torch.manual_seed(seed)
