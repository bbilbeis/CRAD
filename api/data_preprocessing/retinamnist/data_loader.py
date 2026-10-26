import logging
import math
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from .datasets import RetinaMNIST_truncated

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(5):  # 5 classes for RetinaMNIST
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append(tmp)
    return net_cls_counts


def _count_classes(y_train, dataidxs, num_classes=5):
    unq, unq_cnt = np.unique(y_train[np.array(dataidxs)], return_counts=True)
    counts = []
    for label in range(num_classes):
        if label in unq:
            counts.append(unq_cnt[np.argwhere(unq == label)][0, 0])
        else:
            counts.append(0)
    return counts


def _split_train_val_indices(dataidxs, y_train, val_ratio, seed, num_classes=5):
    dataidxs = np.array(dataidxs, dtype=np.int64)
    if val_ratio <= 0.0 or len(dataidxs) < 2:
        return dataidxs, np.array([], dtype=np.int64)

    rng = np.random.RandomState(seed)
    train_idxs = []
    val_idxs = []
    labels = y_train[dataidxs]

    for cls in range(num_classes):
        cls_idxs = dataidxs[labels == cls]
        if len(cls_idxs) < 2:
            train_idxs.extend(cls_idxs.tolist())
            continue

        shuffled = rng.permutation(cls_idxs)
        n_val = max(1, int(math.floor(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        val_idxs.extend(shuffled[:n_val].tolist())
        train_idxs.extend(shuffled[n_val:].tolist())

    if len(val_idxs) == 0 and len(dataidxs) >= 2:
        shuffled = rng.permutation(dataidxs)
        n_val = max(1, int(math.floor(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        val_idxs = shuffled[:n_val].tolist()
        train_idxs = shuffled[n_val:].tolist()

    return (
        np.array(train_idxs, dtype=np.int64),
        np.array(val_idxs, dtype=np.int64),
    )


def _data_transforms_retinamnist():
    # Standard MedMNIST normalization using 0.5 mean/std for each channel
    RETINAMNIST_MEAN = [0.5, 0.5, 0.5]
    RETINAMNIST_STD = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(RETINAMNIST_MEAN, RETINAMNIST_STD),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(RETINAMNIST_MEAN, RETINAMNIST_STD),
    ])

    return train_transform, valid_transform


def load_retinamnist_data(datadir):
    train_transform, test_transform = _data_transforms_retinamnist()
    train_ds = RetinaMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    test_ds = RetinaMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = train_ds.data, train_ds.target
    X_test, y_test = test_ds.data, test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(datadir, partition, n_nets, alpha, logger):
    X_train, y_train, X_test, y_test = load_retinamnist_data(datadir)
    n_cls = 5

    if partition == 'n_cls':
        n_client = n_nets
        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_client):
            cls_priors[i][random.sample(range(n_cls), int(alpha))] = 1.0 / alpha

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = np.random.randint(0, len(idx_list[cls_label]))
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

    elif partition == 'dir':
        n_client = n_nets
        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.random.dirichlet(alpha=[alpha] * n_cls, size=n_client)
        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    available_classes = [i for i, amt in enumerate(cls_amount) if amt > 0]
                    if not available_classes:
                        break
                    cls_label = np.random.choice(available_classes)
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

    elif partition == 'my_part':
        n_shards = int(alpha)
        n_client = n_nets
        n_data_per_clnt = len(y_train) / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(y_train)).astype(int)
        cls_priors = np.zeros(shape=(n_client, n_cls))

        cls_priors_tmp = np.random.dirichlet(alpha=[0.3] * n_cls, size=int(n_shards))

        for i in range(n_client):
            cls_priors[i] = cls_priors_tmp[int(i / int(n_client / n_shards))]

        prior_cumsum = np.cumsum(cls_priors, axis=1)

        idx_list = [np.where(y_train == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]
        net_dataidx_map = {}
        for j in range(n_client):
            net_dataidx_map[j] = []

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = len(idx_list[cls_label])
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader_retinamnist(datadir, train_bs, test_bs, dataidxs=None, test_idxs=None,
                            val_idxs=None, cache_train_data_set=None,
                            cache_test_data_set=None, logger=None):
    transform_train, transform_test = _data_transforms_retinamnist()
    dataidxs = np.array(dataidxs)
    train_ds = RetinaMNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True, cache_data_set=cache_train_data_set)
    test_ds = RetinaMNIST_truncated(datadir, dataidxs=test_idxs, train=False, transform=transform_test, download=True, cache_data_set=cache_test_data_set)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)
    if val_idxs is None:
        return train_dl, test_dl

    val_ds = RetinaMNIST_truncated(
        datadir,
        dataidxs=np.array(val_idxs),
        train=True,
        transform=transform_test,
        download=True,
        cache_data_set=cache_train_data_set,
    )
    val_dl = data.DataLoader(dataset=val_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    return train_dl, test_dl, val_dl


def load_partition_data_retinamnist(data_dir, partition_method, partition_alpha,
                                 client_number, batch_size, logger,
                                 val_ratio=0.0, seed=0):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        data_dir, partition_method, client_number, partition_alpha, logger
    )

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    val_data_local_dict = dict()
    train_cls_counts_after_val = []
    transform_train, transform_test = _data_transforms_retinamnist()

    # Pre-cache local dataset references
    cache_train_data_set = RetinaMNIST_truncated(data_dir, train=True, transform=transform_train, download=True)
    cache_test_data_set = RetinaMNIST_truncated(data_dir, train=False, transform=transform_test, download=True)

    idx_test = [[] for i in range(5)]
    for label in range(5):
        idx_test[label] = np.where(y_test == label)[0]

    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    for client_idx in range(client_number):
        for label in range(5):
            label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num)
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]])
                )

        dataidxs = net_dataidx_map[client_idx]
        train_idxs, val_idxs = _split_train_val_indices(
            dataidxs,
            y_train,
            val_ratio,
            seed + client_idx,
            num_classes=5,
        )
        train_data_local, test_data_local, val_data_local = get_dataloader_retinamnist(
            data_dir,
            batch_size,
            batch_size,
            train_idxs,
            test_dataidxs[client_idx],
            val_idxs=val_idxs,
            cache_train_data_set=cache_train_data_set,
            cache_test_data_set=cache_test_data_set,
            logger=logger,
        )
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        val_data_local_dict[client_idx] = val_data_local
        train_cls_counts_after_val.append(_count_classes(y_train, train_idxs, 5))

    return None, None, None, None, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, \
           val_data_local_dict, train_cls_counts_after_val
