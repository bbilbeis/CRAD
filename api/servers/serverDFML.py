import copy
import math
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from experiments.client import Client
from experiments.my_model_trainer import MyModelTrainer, dfml_local_train, dfml_run_mutual_learning
from utils.hetero_models import create_hetero_model, assign_architectures
import torch.nn as nn
import torch.nn.functional as F

class ServerDFML(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args   = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
         class_counts] = dataset[:8]

        self.val_data_local_dict = dataset[8] if len(dataset) >= 9 else test_data_local_dict
        self.val_cls_counts_dict = dataset[9] if len(dataset) >= 10 else {}
        from utils.resource_tracker import ResourceTracker
        method_name = getattr(args, 'name', '') or getattr(args, 'exp_tag', '') or "DFML"
        self.resource_tracker = ResourceTracker(
            method=method_name,
            dataset=getattr(args, 'dataset', 'cifar10'),
            enabled=getattr(args, 'track_costs', True),
        )

        self.train_global            = train_data_global
        self.test_global             = test_data_global
        self.client_list             = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict     = train_data_local_dict
        self.test_data_local_dict      = test_data_local_dict
        self.class_counts            = class_counts
        self.model_trainer           = model_trainer  # kept for FLOPs reference only

        self.use_global_test = getattr(args, 'use_global_test', False)

        # Build global IID test loader if needed
        if self.use_global_test and self.test_global is None:
            if args.dataset == 'pathmnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
                data_dir = getattr(args, 'data_dir', './data/pathmnist')
                from api.data_preprocessing.pathmnist.datasets import PathMNIST_truncated
                raw = PathMNIST_truncated(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'bloodmnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
                data_dir = getattr(args, 'data_dir', './data/bloodmnist')
                from api.data_preprocessing.bloodmnist.datasets import BloodMNIST_truncated
                raw = BloodMNIST_truncated(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'retinamnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
                data_dir = getattr(args, 'data_dir', './data/retinamnist')
                from api.data_preprocessing.retinamnist.datasets import RetinaMNIST_truncated
                raw = RetinaMNIST_truncated(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'dermamnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
                data_dir = getattr(args, 'data_dir', './data/dermamnist')
                from api.data_preprocessing.dermamnist.datasets import DermaMNIST_truncated
                raw = DermaMNIST_truncated(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'organamnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                data_dir = getattr(args, 'data_dir', './data/organamnist')
                from api.data_preprocessing.organamnist.datasets import OrganAMNIST_truncated
                raw = OrganAMNIST_truncated(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'tissuemnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                data_dir = getattr(args, 'data_dir', './data/tissuemnist')
                from api.data_preprocessing.tissuemnist.datasets import TissueMNIST_truncated
                raw = TissueMNIST_truncated(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'svhn':
                SVHN_MEAN = [0.4377, 0.4438, 0.4728]
                SVHN_STD = [0.1980, 0.2010, 0.1970]
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(SVHN_MEAN, SVHN_STD),
                ])
                data_dir = getattr(args, 'data_dir', './data/svhn')
                raw = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=ts)
            elif args.dataset == 'emnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1751,), (0.3267,)),
                ])
                data_dir = getattr(args, 'data_dir', './data/emnist')
                raw = torchvision.datasets.EMNIST(root=data_dir, split='byclass', train=False, download=True, transform=ts)
            elif args.dataset == 'mnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ])
                data_dir = getattr(args, 'data_dir', './data/mnist')
                raw = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=ts)
            elif args.dataset == 'fashionmnist':
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,)),
                ])
                data_dir = getattr(args, 'data_dir', './data/fashionmnist')
                raw = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=ts)
            else:
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD  = [0.24703233, 0.24348505, 0.26158768]
                ts = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ])
                data_dir = getattr(args, 'data_dir', './data/cifar10')
                d_cls = torchvision.datasets.CIFAR100 if args.dataset == 'cifar100' else torchvision.datasets.CIFAR10
                raw = d_cls(root=data_dir, train=False, download=True, transform=ts)
            self.test_global = DataLoader(raw, batch_size=256, shuffle=False, num_workers=0)

        # Architecture assignment
        self.client_archs = assign_architectures(args.num_clients)
        
        self.num_classes = len(self.class_counts[0]) if self.class_counts else 10

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)
        self.init_stat_info()
        self._analyze_data_distribution()
        
        # DFML Specific: compute class weights (beta_c) for WSM loss for each client
        self.client_class_weights = self._compute_class_weights()

        # DFML Specific: Peak Models and local alpha
        self.peak_models = [copy.deepcopy(self.client_list[i].model_trainer.get_model_params()) for i in range(args.num_clients)]
        self.local_alphas = [0.0 for _ in range(args.num_clients)]

    def _analyze_data_distribution(self):
        self.client_present_classes = {i: set() for i in range(self.args.num_clients)}
        for client_idx in range(self.args.num_clients):
            for x, labels in self.train_data_local_dict[client_idx]:
                for y in labels:
                    self.client_present_classes[client_idx].add(int(y.item()))

    def _compute_class_weights(self):
        client_class_weights = []
        for client_idx in range(self.args.num_clients):
            class_counts = torch.zeros(self.num_classes)
            total = 0
            for x, labels in self.train_data_local_dict[client_idx]:
                for y in labels:
                    class_counts[int(y.item())] += 1
                    total += 1
            if total > 0:
                class_weights = class_counts / total
            else:
                class_weights = torch.ones(self.num_classes) / self.num_classes
            client_class_weights.append(class_weights)
        return client_class_weights

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        num_classes = self.num_classes
        for client_idx in range(self.args.num_clients):
            arch = self.client_archs[client_idx]
            model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
            trainer = MyModelTrainer(model, self.args, self.logger)
            trainer._arch_name = arch
            val_data = self.val_data_local_dict[client_idx] if hasattr(self, 'val_data_local_dict') and client_idx in self.val_data_local_dict else test_data_local_dict[client_idx]
            c = Client(
                client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx], self.args, self.device, trainer, self.logger, arch_name=arch,
                local_val_data=val_data
            )
            self.client_list.append(c)

    def train(self):
        import time
        w_per_mdls = [
            self.client_list[i].model_trainer.get_model_params()
            for i in range(self.args.num_clients)
        ]
        
        alpha_min = getattr(self.args, 'alpha_min', 0.0)
        alpha_max = getattr(self.args, 'alpha_max', 1.0)
        T_max = self.args.num_rounds
        K = getattr(self.args, 'dfml_K', 1)
        T_distill = getattr(self.args, 'T', 4.0)
        training_start = time.perf_counter()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.best_val_acc = 0.0
        self.best_val_round = 0
        self.best_local_test_acc_at_val = 0.0
        self.best_global_test_acc_at_val = 0.0
        self.best_missing_acc_at_val = 0.0
        self.best_present_acc_at_val = 0.0
        self.best_local_acc = 0.0
        self.best_global_acc = 0.0

        for round_idx in range(self.args.num_rounds):
            round_start = time.perf_counter()
            print(f"Communication round : {round_idx}")
            self.logger.info(f"#### Communication round : {round_idx}")

            # 1. Select Aggregator and Senders
            if round_idx == 0:
                aggregator_idx = 0
            else:
                aggregator_idx = random.randint(0, self.args.num_clients - 1)
            
            num_senders = int(self.args.num_clients * self.args.frac)
            num_senders = min(num_senders, self.args.num_clients - 1)
            available_senders = [i for i in range(self.args.num_clients) if i != aggregator_idx]
            senders_idx = random.sample(available_senders, num_senders)
            
            participants_idx = senders_idx + [aggregator_idx]
            print(f"  Aggregator: {aggregator_idx}, Senders: {senders_idx}")

            # 2. Local Training for all participants (Algorithm 1, Client Side)
            for ci in participants_idx:
                client = self.client_list[ci]
                client.model_trainer.set_model_params(w_per_mdls[ci])
                
                print(f'  Round: {round_idx} Client: {ci}/{self.args.num_clients} [{self.client_archs[ci]}] Local Training')
                dfml_local_train(
                    trainer=client.model_trainer,
                    train_data=client.local_training_data,
                    device=self.device,
                    args=self.args,
                    round_idx=round_idx,
                    class_weights=self.client_class_weights[ci]
                )
                w_per_mdls[ci] = client.model_trainer.get_model_params()

            # 3. Aggregator Mutual Learning Phase
            # Cyclical alpha scheduler (Cosine Annealing with Warm Restarts, starting from 0)
            T_0 = 10
            T_mult = 2
            cycle = 0
            T_cur = round_idx
            T_i = T_0
            while T_cur >= T_i:
                T_cur -= T_i
                cycle += 1
                T_i *= T_mult
                
            # Starts at alpha_min, peaks at alpha_max at the end of the period
            alpha_t = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 - math.cos(math.pi * T_cur / max(1, T_i - 1)))
            
            # Prepare models for aggregator
            w_list = [w_per_mdls[ci] for ci in participants_idx]
            arch_list = [self.client_archs[ci] for ci in participants_idx]
            aggregator_train_data = self.client_list[aggregator_idx].local_training_data
            aggregator_class_weights = self.client_class_weights[aggregator_idx]

            print(f'  Round: {round_idx} Aggregator Mutual Learning (alpha={alpha_t:.3f})')
            updated_weights = dfml_run_mutual_learning(
                w_list=w_list,
                arch_list=arch_list,
                agg_train_data=aggregator_train_data,
                device=self.device,
                args=self.args,
                round_idx=round_idx,
                alpha_t=alpha_t,
                class_weights=aggregator_class_weights,
                K=K,
                T=T_distill,
                num_classes=self.num_classes
            )

            # 4. Update Regular and Peak Models
            for local_i, global_i in enumerate(participants_idx):
                w_per_mdls[global_i] = updated_weights[local_i]
                
                if alpha_t >= self.local_alphas[global_i]:
                    self.peak_models[global_i] = copy.deepcopy(updated_weights[local_i])
                self.local_alphas[global_i] = alpha_t
                
            # Evaluation on validation partition and local test partition
            p_val_metrics = {'num_samples': [], 'num_correct': [], 'losses': [], 'per_class_correct': [], 'per_class_total': []}
            p_test_metrics = {'num_samples': [], 'num_correct': [], 'losses': [], 'per_class_correct': [], 'per_class_total': []}
            
            for ci in range(self.args.num_clients):
                client = self.client_list[ci]
                # Evaluate peak models on validation partition
                v_metrics = client.local_val(self.peak_models[ci])
                p_val_metrics['num_samples'].append(v_metrics['test_total'])
                p_val_metrics['num_correct'].append(v_metrics['test_correct'])
                p_val_metrics['losses'].append(v_metrics['test_loss'])
                p_val_metrics['per_class_correct'].append(v_metrics.get('per_class_correct', {}))
                p_val_metrics['per_class_total'].append(v_metrics.get('per_class_total', {}))

                # Evaluate peak models on local test partition
                t_metrics = client.local_test(self.peak_models[ci], b_use_test_dataset=True)
                p_test_metrics['num_samples'].append(t_metrics['test_total'])
                p_test_metrics['num_correct'].append(t_metrics['test_correct'])
                p_test_metrics['losses'].append(t_metrics['test_loss'])
                p_test_metrics['per_class_correct'].append(t_metrics.get('per_class_correct', {}))
                p_test_metrics['per_class_total'].append(t_metrics.get('per_class_total', {}))

            p_val_acc = sum(np.array(p_val_metrics['num_correct']) / np.maximum(1, np.array(p_val_metrics['num_samples']))) / self.args.num_clients
            p_val_loss = sum(np.array(p_val_metrics['losses']) / np.maximum(1, np.array(p_val_metrics['num_samples']))) / self.args.num_clients
            p_local_test_acc = sum(np.array(p_test_metrics['num_correct']) / np.maximum(1, np.array(p_test_metrics['num_samples']))) / self.args.num_clients
            p_local_test_loss = sum(np.array(p_test_metrics['losses']) / np.maximum(1, np.array(p_test_metrics['num_samples']))) / self.args.num_clients
            
            do_global_eval = (
                self.use_global_test
                and self.test_global is not None
                and (
                    round_idx == 0
                    or round_idx == self.args.num_rounds - 1
                    or round_idx % 5 == 4
                )
            )

            if do_global_eval:
                global_per_class = self._eval_global_per_class(self.peak_models)
                missing_acc, present_acc, overall_acc = self._compute_missing_present_acc(
                    global_per_class, source="global"
                )
                self.stat_info["last_missing_acc"] = missing_acc
                self.stat_info["last_present_acc"] = present_acc
                self.stat_info["last_overall_acc"] = overall_acc
            elif self.use_global_test:
                missing_acc = self.stat_info.get("last_missing_acc", 0.0)
                present_acc = self.stat_info.get("last_present_acc", 0.0)
                overall_acc = self.stat_info.get("last_overall_acc", 0.0)
            else:
                missing_acc, present_acc, overall_acc = self._compute_missing_present_acc_local(
                    p_val_metrics['per_class_correct'], 
                    p_val_metrics['per_class_total']
                )
            
            # Per-client class accuracy table (round 0, last, every 5th)
            _skip_until = getattr(self.args, 'skip_eval_until_round', 0)
            is_print = (
                round_idx >= _skip_until and (
                    round_idx == 0 or
                    round_idx == self.args.num_rounds - 1 or
                    (self.args.num_rounds > 10 and (round_idx + 1) % 5 == 0) or
                    (self.args.num_rounds <= 10 and round_idx == self.args.num_rounds // 2)
                )
            )
            if is_print:
                self._print_class_table(
                    p_val_metrics,
                    round_idx,
                    global_per_class=global_per_class if (self.use_global_test and do_global_eval) else None
                )

            self.stat_info["person_val_acc_before"].append(p_val_acc)
            self.stat_info["person_test_acc_before"].append(p_local_test_acc)
            self.stat_info["missing_class_acc"].append(missing_acc)
            self.stat_info["present_class_acc"].append(present_acc)
            self.stat_info["global_class_acc"].append(overall_acc)

            # Checkpoint Selection Rule (S1.1 & E1b):
            # Best checkpoint = highest average local validation accuracy across clients
            if not np.isnan(p_val_acc) and p_val_acc > self.best_val_acc:
                self.best_val_acc = p_val_acc
                self.best_val_round = round_idx
                self.best_local_test_acc_at_val = p_local_test_acc
                self.best_global_test_acc_at_val = overall_acc
                self.best_missing_acc_at_val = missing_acc
                self.best_present_acc_at_val = present_acc
                self.best_local_acc = p_val_acc
                self.best_global_acc = overall_acc

                if round_idx >= 200 or self.args.num_rounds <= 50 or (round_idx + 1) == self.args.num_rounds:
                    self._save_checkpoint(w_per_mdls, round_idx, suffix="best_val")
                    self._save_checkpoint(w_per_mdls, round_idx, suffix="best_local")
            
            if (round_idx + 1) % 50 == 0:
                self._save_checkpoint(w_per_mdls, round_idx)

            # Comm overhead: senders send to aggregator and get back
            # Size = sum(param_count) * 2
            comm_params = 0
            for ci in senders_idx:
                arch = self.client_archs[ci]
                comm_params += sum(p.numel() for p in create_hetero_model(arch, num_classes=self.num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3)).parameters()) * 2
            
            self.stat_info["comm_overhead_params"] += comm_params
            self.stat_info["comm_overhead_mb"]     += (comm_params * 4) / (1024**2)
            self.resource_tracker.add_payload_bytes(comm_params * 4)

            if self.use_global_test:
                global_eval_tag = " [cached]" if not do_global_eval else ""
                print(
                    "Round: {:3d} | Local Acc: {:.4f} | Global Acc (IID){}: {:.4f} | Local Test: {:.4f} "
                    "| Present: {:.2f}% | Missing: {:.2f}% | Loss: {:.4f} | Comm: {:.2f}MB".format(
                        round_idx,
                        p_val_acc,
                        global_eval_tag,
                        overall_acc,
                        p_local_test_acc,
                        present_acc * 100,
                        missing_acc * 100,
                        p_val_loss,
                        self.stat_info["comm_overhead_mb"],
                    )
                )
            else:
                print(
                    "Round: {:3d} | Local Acc: {:.4f} | Local Test: {:.4f} "
                    "| Present: {:.2f}% | Missing: {:.2f}% | Loss: {:.4f} | Comm: {:.2f}MB".format(
                        round_idx,
                        p_val_acc,
                        p_local_test_acc,
                        present_acc * 100,
                        missing_acc * 100,
                        p_val_loss,
                        self.stat_info["comm_overhead_mb"],
                    )
                )

            # ── Round stopwatch ───────────────────────────────────────────
            round_elapsed = time.perf_counter() - round_start
            if not hasattr(self, 'round_durations'):
                self.round_durations = []
            self.round_durations.append(round_elapsed)

            if round_elapsed >= 60:
                time_str = f"{round_elapsed / 60:.1f}min"
            else:
                time_str = f"{round_elapsed:.1f}s"

            if (round_idx + 1) % 5 == 0:
                completed = round_idx + 1
                remaining = self.args.num_rounds - completed
                avg_time = sum(self.round_durations) / completed
                eta_s = avg_time * remaining
                if eta_s >= 3600:
                    eta_str = f"{eta_s / 3600:.1f}h ({eta_s / 60:.0f}min)"
                elif eta_s >= 60:
                    eta_str = f"{eta_s / 60:.1f}min"
                else:
                    eta_str = f"{eta_s:.1f}s"
                time_str += f" | ETA: {eta_str} (for {self.args.num_rounds} rounds)"

            print(f"⏱ {time_str}")

        self.logger.info("DFML Training Finished.")
        self.logger.info(f'all_val_acc={self.stat_info["person_val_acc_before"]}')
        self.logger.info(f'all_test_acc={self.stat_info["person_test_acc_before"]}')
        
        total_elapsed = time.perf_counter() - training_start
        total_min = total_elapsed / 60
        if total_elapsed >= 3600:
            print(f"⏱ Total Training Time: {total_min / 60:.1f}h ({total_min:.0f}min)")
        else:
            print(f"⏱ Total Training Time: {total_min:.1f}min ({total_elapsed:.0f}s)")
        
        # ── Save Final Checkpoint ─────────────────────────────────────────────
        self._save_checkpoint(w_per_mdls, self.args.num_rounds - 1, is_final=True)

        self.logger.info({
            'best_val_round': self.best_val_round,
            'best_val_acc': self.best_val_acc * 100,
            'local_test_at_best_val': self.best_local_test_acc_at_val * 100,
            'global_test_at_best_val': self.best_global_test_acc_at_val * 100,
            'best_missing_acc': self.best_missing_acc_at_val * 100,
            'best_present_acc': self.best_present_acc_at_val * 100,
            'total_comm_mb': self.stat_info["comm_overhead_mb"],
        })

        eval_source = "global IID test" if self.use_global_test else "local partitioned test"
        print("\n" + "=" * 80)
        print(" 🏆 BEST CHECKPOINT SELECTION SUMMARY (Validation-based S1.1)")
        print("=" * 80)
        print(f" Best Round (Highest Avg Local Val Acc): Round {self.best_val_round}")
        print(f"  1. Local Validation Accuracy : {self.best_val_acc * 100:.2f}%")
        print(f"  2. Local Test Accuracy       : {self.best_local_test_acc_at_val * 100:.2f}% (at Best Val Round {self.best_val_round})")
        print(f"  3. Global Oracle Accuracy    : {self.best_global_test_acc_at_val * 100:.2f}% (at Best Val Round {self.best_val_round}) [{eval_source}]")
        print(f"  4. Missing Class Accuracy    : {self.best_missing_acc_at_val * 100:.2f}% (at Best Val Round {self.best_val_round}) [{eval_source}]")
        print(f"  5. Present Class Accuracy    : {self.best_present_acc_at_val * 100:.2f}% (at Best Val Round {self.best_val_round}) [{eval_source}]")
        print("=" * 80)

        print(f"best person_test_acc_before {self.best_val_acc * 100:.3f}%")
        print(f" -> Validation Accuracy (Best Round {self.best_val_round}): {self.best_val_acc * 100:.2f}%")
        print(f" -> Local Test Accuracy (at Best Val Round): {self.best_local_test_acc_at_val * 100:.2f}%")
        print(f" -> Global Oracle Test Accuracy: {self.best_global_test_acc_at_val * 100:.2f}%  [{eval_source}]")
        print(f" -> Missing Class Accuracy: {self.best_missing_acc_at_val * 100:.2f}%  [{eval_source}] (KD evidence)")
        print(f"Total Communication: {self.stat_info['comm_overhead_mb']:.2f} MB")
        self.resource_tracker.save_json()
        self.resource_tracker.print_summary_table()
        print("over")

    def _print_class_table(self, p_test_metrics_before, round_idx, global_per_class=None):
        if self.args.dataset == 'pathmnist':
            class_names = ['adi', 'bg', 'deb', 'lym', 'muc', 'mus', 'norm', 'str', 'ca']
            num_classes = 9
        elif self.args.dataset == 'cifar100':
            class_names = [str(i) for i in range(100)]
            num_classes = 100
        elif self.args.dataset == 'bloodmnist':
            class_names = ['baso', 'eosi', 'eryt', 'ig', 'lymp', 'mono', 'neut', 'plat']
            num_classes = 8
        elif self.args.dataset == 'retinamnist':
            class_names = ['norm', 'mild', 'mod', 'sev', 'prol']
            num_classes = 5
        elif self.args.dataset == 'dermamnist':
            class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
            num_classes = 7
        elif self.args.dataset == 'organamnist':
            class_names = ['blad', 'f-l', 'f-r', 'heart', 'k-l', 'k-r', 'liver', 'lu-l', 'lu-r', 'pan', 'spl']
            num_classes = 11
        elif self.args.dataset == 'tissuemnist':
            class_names = [f"Cls {c}" for c in range(8)]
            num_classes = 8
        elif self.args.dataset == 'cifar10':
            class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            num_classes = 10
        else:
            num_classes = len(self.class_counts[0]) if self.class_counts else 10
            class_names = [f"Cls {c}" for c in range(num_classes)]

        name_prefix = f"[{self.args.name}] " if getattr(self.args, 'name', '') else ""
        print("\n" + "=" * 115)
        is_global = global_per_class is not None
        test_type = "global test" if is_global else "local test"
        print(f"{name_prefix}--- Per-Client Class Accuracy (Round {round_idx}) [{test_type}] ---")
        header = f"{'Client':<8} | {'Arch':<12} | " + " | ".join([f"{name:>6}" for name in class_names]) + " |"
        print(header)
        print("-" * len(header))

        for ci in range(self.args.num_clients):
            arch = self.client_archs[ci]
            row = f"{ci:<8} | {arch:<12} | "
            for cls in range(num_classes):
                if is_global:
                    corr = global_per_class[ci]['correct']
                    tot = global_per_class[ci]['total']
                else:
                    corr = p_test_metrics_before['per_class_correct'][ci]
                    tot = p_test_metrics_before['per_class_total'][ci]

                t = tot.get(cls, 0)
                row += (f"{(corr.get(cls, 0)/t*100):>5.1f}% | " if t > 0
                        else f"{'N/A':>6} | ")
            print(row)
        print("=" * 115 + "\n")

    def _save_checkpoint(self, w_per_mdls, round_idx, is_final=False, suffix=None):
        import os
        import time
        if hasattr(self, 'timestamp'):
            timestamp = self.timestamp
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(self.args.result_path, 'checkpoints', self.args.dataset)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if suffix is None:
            suffix = "final" if is_final else f"round{round_idx}"
        checkpoint_name = f"{getattr(self.args, 'exp_tag', self.args.algorithm)}_{timestamp}_{suffix}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        torch.save({
            'round': round_idx,
            'client_weights': w_per_mdls,
            'peak_weights': self.peak_models,
            'client_archs': self.client_archs,
            'args': self.args,
            'stat_info': self.stat_info,
            'best_val_acc': getattr(self, 'best_val_acc', 0.0),
            'best_val_round': getattr(self, 'best_val_round', 0),
            'best_local_test_acc_at_val': getattr(self, 'best_local_test_acc_at_val', 0.0),
            'best_global_test_acc_at_val': getattr(self, 'best_global_test_acc_at_val', 0.0),
            'best_local_acc': getattr(self, 'best_val_acc', 0.0),
            'best_global_acc': getattr(self, 'best_global_test_acc_at_val', 0.0),
        }, checkpoint_path)
        
        tag_print = suffix.upper() if suffix in ["best_val", "best_local", "best_global"] else ('FINAL' if is_final else 'CHECKPOINT')
        print(f"[{tag_print}] Saved to: {checkpoint_path}")
        self.logger.info(f"Checkpoint saved at round {round_idx} to {checkpoint_path}")

    def init_stat_info(self):
        self.stat_info = {
            "person_val_acc_before":  [],
            "person_test_acc_before": [],
            "missing_class_acc":      [],
            "present_class_acc":      [],
            "global_class_acc":       [],
            "comm_overhead_params":   0,
            "comm_overhead_mb":       0,
            "last_missing_acc":       0.0,
            "last_present_acc":       0.0,
            "last_overall_acc":       0.0,
        }

    def _eval_global_per_class(self, peak_models: list) -> dict:
        device = self.device
        num_classes = self.num_classes
        global_per_class = {}

        for ci in range(self.args.num_clients):
            arch = self.client_archs[ci]
            model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
            model.load_state_dict(peak_models[ci], strict=True)
            model.to(device)
            model.eval()

            per_class_correct = {}
            per_class_total   = {}

            with torch.no_grad():
                for x, target in self.test_global:
                    x, target = x.to(device), target.to(device)
                    pred = model(x)
                    _, predicted = torch.max(pred, -1)

                    for t, p in zip(target.view(-1), predicted.view(-1)):
                        t_val = int(t.item())
                        per_class_correct[t_val] = (
                            per_class_correct.get(t_val, 0) + int((t == p).item())
                        )
                        per_class_total[t_val] = per_class_total.get(t_val, 0) + 1

            model.cpu()
            del model
            global_per_class[ci] = {
                'correct': per_class_correct,
                'total':   per_class_total,
            }

        torch.cuda.empty_cache()
        return global_per_class

    def _compute_missing_present_acc(self, global_per_class: dict, source: str = "global"):
        total_missing_corr = total_missing_samp = 0
        total_present_corr = total_present_samp = 0

        for ci in range(self.args.num_clients):
            present_cls = self.client_present_classes[ci]
            corr = global_per_class[ci]['correct']
            tot  = global_per_class[ci]['total']

            for cls, total in tot.items():
                c = corr.get(cls, 0)
                if cls in present_cls:
                    total_present_corr += c
                    total_present_samp += total
                else:
                    total_missing_corr += c
                    total_missing_samp += total

        missing_acc = total_missing_corr / max(1, total_missing_samp)
        present_acc = total_present_corr / max(1, total_present_samp)
        overall_acc = (total_present_corr + total_missing_corr) / max(1, total_present_samp + total_missing_samp)
        return missing_acc, present_acc, overall_acc

    def _compute_missing_present_acc_local(self, local_corr_list: list, local_tot_list: list):
        total_missing_corr = total_missing_samp = 0
        total_present_corr = total_present_samp = 0

        for ci in range(self.args.num_clients):
            present_cls = self.client_present_classes[ci]
            client_corr = local_corr_list[ci]
            client_tot  = local_tot_list[ci]

            for cls, total in client_tot.items():
                corr = client_corr.get(cls, 0)
                if cls in present_cls:
                    total_present_corr += corr
                    total_present_samp += total
                else:
                    total_missing_corr += corr
                    total_missing_samp += total

        missing_acc = total_missing_corr / max(1, total_missing_samp)
        present_acc = total_present_corr / max(1, total_present_samp)
        overall_acc = (total_present_corr + total_missing_corr) / max(1, total_present_samp + total_missing_samp)
        return missing_acc, present_acc, overall_acc

