"""
Parallel Client Batch Runner Subprocess & Worker.

Manages persistent workers and parallel subprocesses for multi-GPU training
in decentralized knowledge distillation.
"""
import copy
import logging
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataprepare_util import load_data
from experiments.my_model_trainer import MyModelTrainer
from experiments.client import Client
from utils.hetero_models import create_hetero_model


def build_peer_neighbor_info(n_idx, args, w_per_mdls_lstrd, client_archs, classwise_accs, class_counts, client_val_class_counts):
    """Constructs peer distillation neighbor info tuple for CRAD or Uniform KD."""
    if getattr(args, 'idea', 'crad') in ('baseline', 'uniform'):
        return (w_per_mdls_lstrd[n_idx], client_archs[n_idx], None, n_idx)

    use_val = getattr(args, 'fix_eq6', True)
    if use_val and client_val_class_counts is not None:
        count_vec = torch.nan_to_num(client_val_class_counts[n_idx].float(), nan=0.0)
    else:
        count_vec = torch.nan_to_num(torch.tensor(class_counts[n_idx]).float(), nan=0.0)

    acc_vec = torch.nan_to_num(classwise_accs[n_idx].float(), nan=0.0)
    eps = getattr(args, 'eps_var', 1e-6)
    tilde_a = (acc_vec * count_vec + 2.0) / (count_vec + 4.0)
    scaled_rel = 1.0 / ((tilde_a * (1.0 - tilde_a)) / (count_vec + 4.0) + eps)
    zero_mask = (count_vec == 0)
    k_zero_strat = getattr(args, 'k_zero_strategy', 'drop')
    if k_zero_strat == 'drop':
        scaled_rel[zero_mask] = 0.0
    else:
        prior_weight = 1.0 / (0.0625 + eps)
        scaled_rel[zero_mask] = prior_weight
    scaled_rel = torch.nan_to_num(scaled_rel, nan=0.0, posinf=0.0, neginf=0.0)
    train_counts = torch.nan_to_num(torch.tensor(class_counts[n_idx]).float(), nan=0.0)
    return (w_per_mdls_lstrd[n_idx], client_archs[n_idx], scaled_rel, train_counts, n_idx)


def np_benefit_choose(round_idx, clnt_idx, num_clients, client_num_per_round, cs):
    if cs == "full":
        return np.array([i for i in range(num_clients) if i != clnt_idx])
    else:
        np.random.seed(round_idx + clnt_idx)
        return np.random.choice([i for i in range(num_clients) if i != clnt_idx], client_num_per_round - 1, replace=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CRAD Client Batch Runner Subprocess")
    parser.add_argument("--gpu_id", type=int, required=True, help="CUDA GPU ID to use")
    parser.add_argument("--client_ids", type=str, required=True, help="Comma-separated client indices to train")
    parser.add_argument("--round_idx", type=int, required=True, help="Current communication round index")
    parser.add_argument("--in_path", type=str, required=True, help="Path to input round data")
    parser.add_argument("--out_path", type=str, required=True, help="Path to save output results")
    args_sub = parser.parse_args()

    device = torch.device(f"cuda:{args_sub.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    round_data = torch.load(args_sub.in_path, map_location=None)
    args = round_data['args']
    w_per_mdls_lstrd = round_data['w_per_mdls_lstrd']
    client_class_weights = round_data['client_class_weights']
    classwise_accs = round_data['classwise_accs']
    client_val_class_counts = round_data.get('client_val_class_counts', None)
    compare_local = round_data.get('compare_local', False)
    w_standalone_per_mdls = round_data.get('w_standalone_per_mdls', None)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"Subprocess_GPU_{args_sub.gpu_id}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_data(args, args.dataset, logger)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset[:8]
    val_data_local_dict = dataset[8] if len(dataset) >= 9 else test_data_local_dict
    client_archs = round_data['client_archs']
    client_ids = [int(x) for x in args_sub.client_ids.split(',')]

    results = {}
    for clnt_idx in client_ids:
        # Per-client RNG seed isolation
        client_round_seed = args.seed + (args_sub.round_idx * 10000) + clnt_idx
        random.seed(client_round_seed)
        np.random.seed(client_round_seed)
        torch.manual_seed(client_round_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(client_round_seed)

        arch = client_archs[clnt_idx]
        num_classes = len(class_counts[0]) if class_counts else 10
        in_channels = getattr(args, 'in_channels', 3)
        model = create_hetero_model(arch, num_classes=num_classes, in_channels=in_channels)
        trainer = MyModelTrainer(model, args, logger)
        trainer._arch_name = arch

        val_data = val_data_local_dict[clnt_idx] if clnt_idx in val_data_local_dict else test_data_local_dict[clnt_idx]
        client = Client(
            clnt_idx,
            train_data_local_dict[clnt_idx],
            test_data_local_dict[clnt_idx],
            train_data_local_num_dict[clnt_idx],
            args,
            device,
            trainer,
            logger,
            arch_name=arch,
            local_val_data=val_data,
        )

        print(f"  Round: {args_sub.round_idx} Client: {clnt_idx}/{args.num_clients} [{arch}] (on cuda:{args_sub.gpu_id}) Training...")
        start_time = time.perf_counter()

        client_num_per_round = int(args.num_clients * args.frac)
        nei_indexs = np_benefit_choose(args_sub.round_idx, clnt_idx, args.num_clients, client_num_per_round, args.cs)
        if args.num_clients != client_num_per_round:
            nei_indexs = np.append(nei_indexs, clnt_idx)
        nei_indexs = sorted(list(set(nei_indexs)))

        neighbor_info = []
        if args.algorithm not in ("FedAvg", "Standalone"):
            for n_idx in nei_indexs:
                if n_idx != clnt_idx:
                    info_tuple = build_peer_neighbor_info(
                        n_idx, args, w_per_mdls_lstrd, client_archs, classwise_accs, class_counts, client_val_class_counts
                    )
                    neighbor_info.append(info_tuple)

        class_weights = client_class_weights[clnt_idx]
        w_local_mdl, p_test_before_metric = client.train(
            copy.deepcopy(w_per_mdls_lstrd[clnt_idx]),
            neighbor_info,
            args_sub.round_idx,
            class_weights=class_weights,
        )

        comm_params = sum(
            sum(p.numel() for p in entry[0].values())
            for entry in neighbor_info
        )

        standalone_acc = 0.0
        w_standalone_new = None
        if compare_local and w_standalone_per_mdls is not None:
            w_stand_new, stand_metrics = client.train(
                copy.deepcopy(w_standalone_per_mdls[clnt_idx]),
                None,
                args_sub.round_idx,
            )
            w_standalone_new = {k: v.cpu() for k, v in w_stand_new.items()}
            standalone_acc = (
                stand_metrics['test_correct'] / stand_metrics['test_total']
                if stand_metrics['test_total'] > 0 else 0.0
            )

        elapsed = time.perf_counter() - start_time
        print(f"    Client {clnt_idx} [{arch}] done in {elapsed:.1f}s")

        results[clnt_idx] = {
            'clnt_idx': clnt_idx,
            'w_local_mdl': {k: v.cpu() for k, v in w_local_mdl.items()},
            'p_test_before_metric': p_test_before_metric,
            'comm_params': comm_params,
            'standalone_acc': standalone_acc,
            'w_standalone_new': w_standalone_new,
        }

    torch.save(results, args_sub.out_path)


def persistent_worker(gpu_id, task_queue, result_queue):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"  [Worker GPU {gpu_id}] Starting on {device}...")
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
            torch.cuda.synchronize(device)
            print(f"  [Worker GPU {gpu_id}] CUDA context initialized successfully.")
        except Exception as e:
            print(f"  [Worker GPU {gpu_id}] Failed to initialize CUDA: {e}")
            result_queue.put({'type': 'error', 'error': f"CUDA init failed: {str(e)}"})
            return

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"Worker_GPU_{gpu_id}")

    args = None
    client_archs = None
    clients = {}
    class_counts = None

    print(f"  [Worker GPU {gpu_id}] Ready for tasks.")

    while True:
        try:
            task = task_queue.get()
        except Exception as e:
            logger.error(f"Error getting task from queue: {e}")
            break

        if task is None:
            break

        task_type = task.get('type')
        if task_type == 'init':
            print(f"  [Worker GPU {gpu_id}] Received init task...")
            try:
                args = task['args']
                client_archs = task['client_archs']
                client_ids = task['client_ids']

                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)

                print(f"  [Worker GPU {gpu_id}] Loading data partition...")
                dataset = load_data(args, args.dataset, logger)
                [train_data_num, test_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset[:8]
                val_data_local_dict = dataset[8] if len(dataset) >= 9 else test_data_local_dict

                num_classes = len(class_counts[0]) if class_counts else 10
                print(f"  [Worker GPU {gpu_id}] Data loaded. Setting up {len(client_ids)} clients...")

                for clnt_idx in client_ids:
                    arch = client_archs[clnt_idx]
                    in_channels = getattr(args, 'in_channels', 3)
                    model = create_hetero_model(arch, num_classes=num_classes, in_channels=in_channels)
                    trainer = MyModelTrainer(model, args, logger)
                    trainer._arch_name = arch

                    val_data = val_data_local_dict[clnt_idx] if clnt_idx in val_data_local_dict else test_data_local_dict[clnt_idx]
                    client = Client(
                        clnt_idx,
                        train_data_local_dict[clnt_idx],
                        test_data_local_dict[clnt_idx],
                        train_data_local_num_dict[clnt_idx],
                        args,
                        device,
                        trainer,
                        logger,
                        arch_name=arch,
                        local_val_data=val_data,
                    )
                    clients[clnt_idx] = client

                print(f"  [Worker GPU {gpu_id}] Init complete.")
                result_queue.put({'type': 'init_done'})
            except Exception as e:
                print(f"  [Worker GPU {gpu_id}] Error during init: {e}")
                logger.error(f"Error during init task: {e}")
                result_queue.put({'type': 'error', 'error': str(e)})

        elif task_type == 'train':
            try:
                round_idx = task['round_idx']
                w_per_mdls_lstrd = task['w_per_mdls_lstrd']
                client_class_weights = task['client_class_weights']
                classwise_accs = task['classwise_accs']
                client_val_class_counts = task.get('client_val_class_counts', None)
                compare_local = task.get('compare_local', False)
                w_standalone_per_mdls = task.get('w_standalone_per_mdls', None)
                client_ids = task['client_ids']

                results = {}
                for clnt_idx in client_ids:
                    client_round_seed = args.seed + (round_idx * 10000) + clnt_idx
                    random.seed(client_round_seed)
                    np.random.seed(client_round_seed)
                    torch.manual_seed(client_round_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(client_round_seed)

                    client = clients[clnt_idx]
                    arch = client_archs[clnt_idx]

                    client.model_trainer.model.load_state_dict(w_per_mdls_lstrd[clnt_idx])

                    client_num_per_round = int(args.num_clients * args.frac)
                    nei_indexs = np_benefit_choose(round_idx, clnt_idx, args.num_clients, client_num_per_round, args.cs)
                    if args.num_clients != client_num_per_round:
                        nei_indexs = np.append(nei_indexs, clnt_idx)
                    nei_indexs = sorted(list(set(nei_indexs)))

                    neighbor_info = []
                    if args.algorithm not in ("FedAvg", "Standalone"):
                        for n_idx in nei_indexs:
                            if n_idx != clnt_idx:
                                info_tuple = build_peer_neighbor_info(
                                    n_idx, args, w_per_mdls_lstrd, client_archs, classwise_accs, class_counts, client_val_class_counts
                                )
                                neighbor_info.append(info_tuple)

                    class_weights = client_class_weights[clnt_idx]
                    start_time = time.perf_counter()
                    print(f"  Round: {round_idx} Client: {clnt_idx}/{args.num_clients} [{arch}] (on cuda:{gpu_id}) Training...")

                    w_local_mdl, p_test_before_metric = client.train(
                        copy.deepcopy(w_per_mdls_lstrd[clnt_idx]),
                        neighbor_info,
                        round_idx,
                        class_weights=class_weights,
                    )

                    comm_params = sum(
                        sum(p.numel() for p in entry[0].values())
                        for entry in neighbor_info
                    )

                    standalone_acc = 0.0
                    w_standalone_new = None
                    if compare_local and w_standalone_per_mdls is not None:
                        w_stand_new, stand_metrics = client.train(
                            copy.deepcopy(w_standalone_per_mdls[clnt_idx]),
                            None,
                            round_idx,
                        )
                        w_standalone_new = {k: v.cpu() for k, v in w_stand_new.items()}
                        standalone_acc = (
                            stand_metrics['test_correct'] / stand_metrics['test_total']
                            if stand_metrics['test_total'] > 0 else 0.0
                        )

                    elapsed = time.perf_counter() - start_time
                    print(f"    Client {clnt_idx} [{arch}] done in {elapsed:.1f}s")

                    results[clnt_idx] = {
                        'clnt_idx': clnt_idx,
                        'w_local_mdl': {k: v.cpu() for k, v in w_local_mdl.items()},
                        'p_test_before_metric': p_test_before_metric,
                        'comm_params': comm_params,
                        'standalone_acc': standalone_acc,
                        'w_standalone_new': w_standalone_new,
                    }

                result_queue.put({'type': 'train_done', 'results': results})
            except Exception as e:
                logger.error(f"Error during train task in round {round_idx}: {e}")
                result_queue.put({'type': 'error', 'error': str(e)})


if __name__ == "__main__":
    main()
