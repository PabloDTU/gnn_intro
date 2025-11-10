from copy import deepcopy

import torch
import torch.utils.data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import create_dataset, get_loader


def prepare_data(train_dataset, train_size: int):
    val_dataset = deepcopy(train_dataset)
    unlabbeled_dataset = deepcopy(train_dataset)
    val_split = 0.1  # % of training data used for validation
    val_size = int(val_split * len(train_dataset))
    indexes = torch.randperm(len(train_dataset)).tolist()
    indexes_train = indexes[:train_size]
    indexes_val = indexes[train_size : (train_size + val_size)]
    indexes_unlabelled = indexes[(train_size + val_size) :]
    train_dataset.targets = train_dataset.targets[indexes_train]
    train_dataset.data = train_dataset.data[indexes_train]
    val_dataset.targets = val_dataset.targets[indexes_val]
    val_dataset.data = val_dataset.data[indexes_val]
    unlabbeled_dataset.targets = unlabbeled_dataset.targets[indexes_unlabelled]
    unlabbeled_dataset.data = unlabbeled_dataset.data[indexes_unlabelled]
    return train_dataset, val_dataset, unlabbeled_dataset


def create_loader(unsupervised_train_fraction=0.0):
    """Create data loader object.

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    train_dataset = dataset

    # Split training data into supervised and unsupervised portions
    total_train_size = len(train_dataset)
    supervised_size = int(total_train_size * (1 - unsupervised_train_fraction))

    # Create indices for supervised and unsupervised splits
    indices = torch.randperm(total_train_size).tolist()
    supervised_indices = indices[:supervised_size]
    unsupervised_indices = indices[supervised_size:]

    # Create supervised training loader
    supervised_dataset = torch.utils.data.Subset(train_dataset, supervised_indices)
    loaders = [get_loader(supervised_dataset, cfg.train.sampler, cfg.train.batch_size, shuffle=True)]

    # Create unsupervised training loader
    unsupervised_dataset = torch.utils.data.Subset(train_dataset, unsupervised_indices)
    if len(unsupervised_dataset) != 0:
        unsupervised_loader = get_loader(
            unsupervised_dataset, cfg.train.sampler, cfg.train.batch_size, shuffle=True
        )
    else:
        unsupervised_loader = None
    loaders.append(unsupervised_loader)

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == "graph":
            split_names = ["val_graph_index", "test_graph_index"]
            split_indices = dataset.data[split_names[i]]
            loaders.append(get_loader(dataset[split_indices], cfg.val.sampler, cfg.train.batch_size, shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(get_loader(dataset, cfg.val.sampler, cfg.train.batch_size, shuffle=False))

    print(f"Size of supervised_dataset: {len(supervised_dataset)}")
    print(f"Size of unsupervised_dataset: {len(unsupervised_dataset)}")
    if len(loaders) > 2:
        print(f"Size of validation dataset: {len(loaders[2].dataset)}")
    if len(loaders) > 3:
        print(f"Size of test dataset: {len(loaders[3].dataset)}")
        print("Test")
    return loaders
