from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.pathmnist.data_loader import load_partition_data_pathmnist
from api.data_preprocessing.bloodmnist.data_loader import load_partition_data_bloodmnist
from api.data_preprocessing.retinamnist.data_loader import load_partition_data_retinamnist
from api.data_preprocessing.dermamnist.data_loader import load_partition_data_dermamnist
from api.data_preprocessing.organamnist.data_loader import load_partition_data_organamnist
from api.data_preprocessing.tissuemnist.data_loader import load_partition_data_tissuemnist
from api.data_preprocessing.mnist.data_loader import load_partition_data_mnist
from api.data_preprocessing.fashionmnist.data_loader import load_partition_data_fmnist
from api.data_preprocessing.svhn.data_loader import load_partition_data_svhn
from api.data_preprocessing.emnist.data_loader import load_partition_data_emnist
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets.cifar as cifar_module

# Override default Toronto links with fast BrainChip mirror
cifar_module.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
cifar_module.CIFAR100.url = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"


def get_benchmark_global_test_dataset(dataset_name, data_dir, batch_size=64):
    """
    Loads and returns (test_loader_global, test_dataset_global, global_targets)
    for the benchmark's official test split.
    """
    dname = dataset_name.lower()
    if dname == "organmnist":
        dname = "organamnist"

    normalized = os.path.normpath(data_dir)
    if os.path.basename(normalized) == dname:
        base_data_dir = os.path.dirname(normalized)
    else:
        base_data_dir = normalized
    target_data_dir = os.path.join(base_data_dir, dname)

    if dname == "cifar10":
        from torchvision.datasets import CIFAR10
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        ds = CIFAR10(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.targets)

    elif dname == "cifar100":
        from torchvision.datasets import CIFAR100
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        ds = CIFAR100(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.targets)

    elif dname == "pathmnist":
        from api.data_preprocessing.pathmnist.datasets import PathMNIST_truncated
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        ds = PathMNIST_truncated(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.target).ravel()

    elif dname == "bloodmnist":
        from api.data_preprocessing.bloodmnist.datasets import BloodMNIST_truncated
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        ds = BloodMNIST_truncated(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.target).ravel()

    elif dname == "retinamnist":
        from api.data_preprocessing.retinamnist.datasets import RetinaMNIST_truncated
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        ds = RetinaMNIST_truncated(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.target).ravel()

    elif dname == "dermamnist":
        from api.data_preprocessing.dermamnist.datasets import DermaMNIST_truncated
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        ds = DermaMNIST_truncated(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.target).ravel()

    elif dname == "organamnist":
        from api.data_preprocessing.organamnist.datasets import OrganAMNIST_truncated
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        ds = OrganAMNIST_truncated(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.target).ravel()

    elif dname == "tissuemnist":
        from api.data_preprocessing.tissuemnist.datasets import TissueMNIST_truncated
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        ds = TissueMNIST_truncated(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.target).ravel()

    elif dname == "mnist":
        from torchvision.datasets import MNIST
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = MNIST(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.targets).ravel()

    elif dname == "fashionmnist":
        from torchvision.datasets import FashionMNIST
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        ds = FashionMNIST(target_data_dir, train=False, download=True, transform=ts)
        targets = np.array(ds.targets).ravel()

    elif dname == "svhn":
        from torchvision.datasets import SVHN
        SVHN_MEAN = [0.4377, 0.4438, 0.4728]
        SVHN_STD = [0.1980, 0.2010, 0.1970]
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        ds = SVHN(target_data_dir, split='test', download=True, transform=ts)
        targets = np.array(ds.labels).ravel()

    elif dname == "emnist":
        from torchvision.datasets import EMNIST
        ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
        ])
        ds = EMNIST(target_data_dir, split='byclass', train=False, download=True, transform=ts)
        targets = np.array(ds.targets).ravel()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if targets.ndim > 1:
        targets = targets.ravel()

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return dl, ds, targets


def load_data(args, dataset_name, logger):
    seed = getattr(args, 'seed', 1024)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if dataset_name == "organmnist":
        dataset_name = "organamnist"

    logger.info("Preparing the dataset {} with non-iid {} and alpha {}".format(dataset_name,args.partition_method,args.partition_alpha))

    # Strip the dataset name suffix if args.data_dir was already mutated (e.g. in subprocesses/workers)
    normalized_path = os.path.normpath(args.data_dir)
    if os.path.basename(normalized_path) == dataset_name:
        args.data_dir = os.path.dirname(normalized_path)

    # Pre-fetch the benchmark's official test DataLoader & dataset
    test_data_global, global_test_dataset, global_targets = get_benchmark_global_test_dataset(
        dataset_name, args.data_dir, batch_size=getattr(args, 'batch_size', 64)
    )
    test_data_num = len(global_test_dataset)

    if dataset_name == "cifar10":
        args.data_dir = os.path.join(args.data_dir, "cifar10")
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.num_clients, args.batch_size, logger)
    elif dataset_name == "cifar100":
        args.data_dir = os.path.join(args.data_dir, "cifar100")
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         class_num = load_partition_data_cifar100(args.data_dir, args.partition_method,
                                                args.partition_alpha, args.num_clients, args.batch_size, logger)
    elif dataset_name == "pathmnist":
        args.data_dir = os.path.join(args.data_dir, "pathmnist")
        res = load_partition_data_pathmnist(args.data_dir, args.partition_method,
                                            args.partition_alpha, args.num_clients, args.batch_size, logger)
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         _, class_num = res
    elif dataset_name == "bloodmnist":
        args.data_dir = os.path.join(args.data_dir, "bloodmnist")
        res = load_partition_data_bloodmnist(args.data_dir, args.partition_method,
                                            args.partition_alpha, args.num_clients, args.batch_size, logger)
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         _, class_num = res
    elif dataset_name == "retinamnist":
        args.data_dir = os.path.join(args.data_dir, "retinamnist")
        res = load_partition_data_retinamnist(args.data_dir, args.partition_method,
                                            args.partition_alpha, args.num_clients, args.batch_size, logger)
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         _, class_num = res
    elif dataset_name == "dermamnist":
        args.data_dir = os.path.join(args.data_dir, "dermamnist")
        res = load_partition_data_dermamnist(args.data_dir, args.partition_method,
                                            args.partition_alpha, args.num_clients, args.batch_size, logger)
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         _, class_num = res
    elif dataset_name == "organamnist":
        args.data_dir = os.path.join(args.data_dir, "organamnist")
        res = load_partition_data_organamnist(args.data_dir, args.partition_method,
                                            args.partition_alpha, args.num_clients, args.batch_size, logger)
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         _, class_num = res
    elif dataset_name == "tissuemnist":
        args.data_dir = os.path.join(args.data_dir, "tissuemnist")
        res = load_partition_data_tissuemnist(args.data_dir, args.partition_method,
                                            args.partition_alpha, args.num_clients, args.batch_size, logger)
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         _, class_num = res
    elif dataset_name == "mnist":
        args.data_dir = os.path.join(args.data_dir, "mnist")
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         class_num = load_partition_data_mnist(args.data_dir, args.partition_method,
                                              args.partition_alpha, args.num_clients, args.batch_size, logger)
    elif dataset_name == "fashionmnist":
        args.data_dir = os.path.join(args.data_dir, "fashionmnist")
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         class_num = load_partition_data_fmnist(args.data_dir, args.partition_method,
                                               args.partition_alpha, args.num_clients, args.batch_size, logger)
    elif dataset_name == "svhn":
        args.data_dir = os.path.join(args.data_dir, "svhn")
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         class_num = load_partition_data_svhn(args.data_dir, args.partition_method,
                                             args.partition_alpha, args.num_clients, args.batch_size, logger)
    elif dataset_name == "emnist":
        args.data_dir = os.path.join(args.data_dir, "emnist")
        train_data_num, _, train_data_global, _,         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,         class_num = load_partition_data_emnist(args.data_dir, args.partition_method,
                                               args.partition_alpha, args.num_clients, args.batch_size, logger)

    # Perform split / resampling of local test set into local_val_set and local_test_set
    val_data_local_dict, test_data_local_dict, val_cls_counts_dict = split_val_test_datasets(
        train_data_local_dict, test_data_local_dict, test_data_global, args, logger
    )

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
               val_data_local_dict, val_cls_counts_dict]

    logger.info("Finished data preparation in {} clients".format(args.num_clients))
    return dataset


def split_val_test_datasets(train_data_local_dict, test_data_local_dict, test_data_global, args, logger):
    """
    Splits/configures each client's local_val_set and local_test_set based on protocol:
    - Protocol 'option_a' (shard split): split original local test set 50/50 into val and test.
      Asserts zero index overlap between val and test.
    - Protocol 'option_b' (resampled): entire original local test set becomes val; local test set
      is sampled without replacement from benchmark's official test split matching client's
      validation class distribution and sized to match.
    - Saves and loads split indices to/from disk ('data/splits/...') to guarantee identical data splits across methods.
    """
    val_test_ratio = getattr(args, 'val_test_ratio', 0.5)
    seed = getattr(args, 'seed', 1024)
    split_enabled = getattr(args, 'split_val_test', False)
    protocol = getattr(args, 'val_test_protocol', 'option_a').lower()
    dataset_name = getattr(args, 'dataset', 'dataset')
    partition_method = getattr(args, 'partition_method', 'dir')
    partition_alpha = getattr(args, 'partition_alpha', 0.3)
    num_clients_cfg = getattr(args, 'num_clients', len(test_data_local_dict))

    splits_dir = os.path.join("data", "splits")
    os.makedirs(splits_dir, exist_ok=True)

    val_data_local_dict = {}
    new_test_data_local_dict = {}
    val_cls_counts_dict = {}

    num_clients = len(test_data_local_dict)

    # Extract global test targets for Option B
    global_test_dataset = getattr(test_data_global, 'dataset', None) if test_data_global is not None else None
    if global_test_dataset is not None:
        if hasattr(global_test_dataset, 'target') and global_test_dataset.target is not None:
            global_targets = np.array(global_test_dataset.target)
        elif hasattr(global_test_dataset, 'targets') and global_test_dataset.targets is not None:
            global_targets = np.array(global_test_dataset.targets)
        elif hasattr(global_test_dataset, 'labels') and global_test_dataset.labels is not None:
            global_targets = np.array(global_test_dataset.labels)
        else:
            global_targets = np.array([global_test_dataset[i][1] for i in range(len(global_test_dataset))])
        if global_targets.ndim > 1:
            global_targets = global_targets.ravel()
    else:
        global_targets = None

    for client_idx in range(num_clients):
        test_loader = test_data_local_dict[client_idx]
        train_loader = train_data_local_dict.get(client_idx, None)
        N_train = len(train_loader.dataset) if train_loader is not None and hasattr(train_loader, 'dataset') else 0

        if not split_enabled:
            # Legacy mode: local_val_set is identical to local_test_set
            val_data_local_dict[client_idx] = test_loader
            new_test_data_local_dict[client_idx] = test_loader
            N_val = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0
            N_test = N_val
            val_cls_counts_dict[client_idx] = {}
            msg = f"[Data Split] Client {client_idx}: Train={N_train}, Val={N_val}, Test={N_test} (Legacy mode)"
            if logger is not None:
                logger.info(msg)
            print(msg)
            continue

        # Standardized split filename incorporating partition parameters
        split_file = os.path.join(
            splits_dir,
            f"{dataset_name}_{partition_method}_alpha{partition_alpha}_clients{num_clients_cfg}_{protocol}_seed{seed}_client{client_idx}.pt"
        )
        legacy_split_file = os.path.join(
            splits_dir,
            f"{dataset_name}_{protocol}_seed{seed}_client{client_idx}.pt"
        )

        dataset = test_loader.dataset
        N_total_test = len(dataset)

        # Check if split file exists on disk
        split_loaded = False
        target_file_to_load = split_file if os.path.exists(split_file) else (legacy_split_file if os.path.exists(legacy_split_file) else None)
        if target_file_to_load is not None:
            try:
                split_data = torch.load(target_file_to_load, weights_only=False)
                val_indices = [int(i) for i in split_data['val_indices'] if int(i) < N_total_test]
                is_global_test = split_data.get('is_global_test', False)
                if is_global_test and global_test_dataset is not None:
                    test_indices = [int(i) for i in split_data['test_indices'] if int(i) < len(global_test_dataset)]
                else:
                    test_indices = [int(i) for i in split_data['test_indices'] if int(i) < N_total_test]
                val_counts = split_data.get('val_counts', {})
                if len(val_indices) > 0 and len(test_indices) > 0:
                    split_loaded = True
            except Exception as e:
                print(f"⚠️ [Data Split] Could not load split file {target_file_to_load} ({e}), regenerating...")
                if os.path.exists(target_file_to_load):
                    try:
                        os.remove(target_file_to_load)
                    except Exception:
                        pass

        if not split_loaded:
            # Extract targets / labels of local test dataset
            if hasattr(dataset, 'target') and dataset.target is not None:
                targets = np.array(dataset.target)
            elif hasattr(dataset, 'targets') and dataset.targets is not None:
                targets = np.array(dataset.targets)
            elif hasattr(dataset, 'labels') and dataset.labels is not None:
                targets = np.array(dataset.labels)
            else:
                targets = np.array([dataset[i][1] for i in range(N_total_test)])

            if targets.ndim > 1:
                targets = targets.ravel()

            unique_classes = np.unique(targets)
            rng = np.random.RandomState(seed + client_idx)

            if protocol == 'option_b' and global_targets is not None:
                # Option B: entire original partition becomes validation set
                val_indices = list(range(N_total_test))
                is_global_test = True
                test_indices = []

                # Compute per-class sample counts in validation partition
                val_counts = {}
                for idx in val_indices:
                    lbl = int(targets[idx])
                    val_counts[lbl] = val_counts.get(lbl, 0) + 1

                # Sample without replacement within client from global test set matching client's val class proportions
                for c, n_needed in val_counts.items():
                    global_cls_idxs = np.where(global_targets == c)[0]
                    if len(global_cls_idxs) == 0:
                        continue
                    n_draw = min(n_needed, len(global_cls_idxs))
                    sampled = rng.choice(global_cls_idxs, size=n_draw, replace=False)
                    test_indices.extend(sampled.tolist())

                # Hard assertion for Option B ensuring NO duplicate images within client
                assert len(test_indices) == len(set(test_indices)), (
                    f"❌ [Data Split Error] Duplicate test indices sampled for Client {client_idx} in Option B!"
                )
            else:
                # Option A: 50/50 split of original local test partition
                is_global_test = False
                val_indices = []
                test_indices = []

                if N_total_test < 2:
                    val_indices = list(range(N_total_test))
                    test_indices = list(range(N_total_test))
                else:
                    for c in unique_classes:
                        cls_idxs = np.where(targets == c)[0]
                        if len(cls_idxs) == 0:
                            continue
                        if len(cls_idxs) == 1:
                            val_indices.append(int(cls_idxs[0]))
                            continue

                        perm = rng.permutation(cls_idxs)
                        n_val = int(round(len(perm) * val_test_ratio))
                        n_val = max(1, min(len(perm) - 1, n_val))
                        val_indices.extend(perm[:n_val].tolist())
                        test_indices.extend(perm[n_val:].tolist())

                    if len(val_indices) == 0 or len(test_indices) == 0:
                        perm = rng.permutation(N_total_test)
                        n_val = max(1, min(N_total_test - 1, int(round(N_total_test * val_test_ratio))))
                        val_indices = perm[:n_val].tolist()
                        test_indices = perm[n_val:].tolist()

                # Compute per-class sample counts for Option A validation set
                val_counts = {}
                for idx in val_indices:
                    lbl = int(targets[idx])
                    val_counts[lbl] = val_counts.get(lbl, 0) + 1

                # Hard assertion for Option A ensuring ZERO index overlap
                set_val = set(val_indices)
                set_test = set(test_indices)
                assert len(set_val.intersection(set_test)) == 0, (
                    f"❌ [Data Split Error] Index overlap detected between local_val_set and local_test_set for Client {client_idx}!"
                )

            # Save split indices atomically to prevent race conditions across parallel jobs
            split_data = {
                'val_indices': val_indices,
                'test_indices': test_indices,
                'is_global_test': is_global_test,
                'val_counts': val_counts,
                'protocol': protocol,
                'seed': seed,
                'dataset_name': dataset_name,
                'client_idx': client_idx,
            }
            tmp_file = f"{split_file}.tmp.{os.getpid()}"
            try:
                torch.save(split_data, tmp_file)
                os.replace(tmp_file, split_file)
            except Exception:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except Exception:
                        pass
                torch.save(split_data, split_file)

        # Construct Subsets & DataLoaders
        val_sub = Subset(dataset, val_indices)
        if is_global_test and global_test_dataset is not None:
            test_sub = Subset(global_test_dataset, test_indices)
        else:
            test_sub = Subset(dataset, test_indices)

        bs = getattr(test_loader, 'batch_size', 64) or 64
        val_loader = DataLoader(val_sub, batch_size=bs, shuffle=False, drop_last=False)
        test_loader_new = DataLoader(test_sub, batch_size=bs, shuffle=False, drop_last=False)

        val_data_local_dict[client_idx] = val_loader
        new_test_data_local_dict[client_idx] = test_loader_new
        val_cls_counts_dict[client_idx] = val_counts

        N_val = len(val_indices)
        N_test = len(test_indices)
        msg = f"[Data Split] Client {client_idx}: Train={N_train}, Val={N_val}, Test={N_test} (Protocol: {protocol.upper()})"
        print(msg)
        if logger is not None:
            logger.info(msg)

    return val_data_local_dict, new_test_data_local_dict, val_cls_counts_dict
