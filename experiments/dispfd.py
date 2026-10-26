"""
Decentralized Federated Learning (DFL) Orchestrator.

This module (dispfd) manages the overall lifecycle of the decentralized training 
process, including:
1. Client initialization and heterogeneous architecture assignment.
2. Communication round coordination and neighbor selection (topology).
3. Metric aggregation and logging (accuracy, loss, communication overhead).
4. Missing vs. Present class accuracy evaluation (KD evidence metrics).
"""
import copy
import sys
import os
import gc
import logging
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from experiments.client import Client
from experiments.my_model_trainer import MyModelTrainer
from utils.hetero_models import create_hetero_model, assign_architectures


try:
    import ray
    
    @ray.remote(num_gpus=1)
    class RayTrainerActor:
        def __init__(self, gpu_id, args, client_archs, client_ids):
            import os
            import sys
            import torch
            import random
            import numpy as np
            import logging
            
            try:
                # Add root directory to python path to allow imports
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
                from utils.dataprepare_util import load_data
                from experiments.my_model_trainer import MyModelTrainer
                from experiments.client import Client
                from experiments.main import create_hetero_model
                
                self.gpu_id = gpu_id
                self.args = args
                self.client_archs = client_archs
                self.client_ids = client_ids
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                # Setup logging
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(f"RayWorker_GPU_{gpu_id}")
                
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
                    
                # Load partition data
                print(f"[Ray Worker {gpu_id}] Loading partition data on worker...")
                dataset = load_data(args, args.dataset, logger)
                [train_data_num, test_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset[:8]
                self.val_data_local_dict = dataset[8] if len(dataset) >= 9 else test_data_local_dict
                self.val_cls_counts_dict = dataset[9] if len(dataset) >= 10 else {}
                 
                self.class_counts = class_counts
                self.num_classes = len(class_counts[0]) if class_counts else 10
                self.clients = {}
                
                # Setup clients for this actor
                for clnt_idx in client_ids:
                    arch = client_archs[clnt_idx]
                    in_channels = getattr(args, 'in_channels', 3)
                    model = create_hetero_model(arch, num_classes=self.num_classes, tiny_experts=args.tiny_experts, in_channels=in_channels)
                    trainer = MyModelTrainer(model, args, logger)
                    trainer._arch_name = arch
                    val_data = self.val_data_local_dict[clnt_idx] if hasattr(self, 'val_data_local_dict') and clnt_idx in self.val_data_local_dict else test_data_local_dict[clnt_idx]

                    client = Client(
                        clnt_idx,
                        train_data_local_dict[clnt_idx],
                        test_data_local_dict[clnt_idx],
                        train_data_local_num_dict[clnt_idx],
                        args,
                        self.device,
                        trainer,
                        logger,
                        arch_name=arch,
                        local_val_data=val_data,
                    )
                    self.clients[clnt_idx] = client
                print(f"[Ray Worker {gpu_id}] Setup complete for client IDs: {client_ids}")
            except Exception as e:
                import traceback
                print(f"❌ [Ray Worker {gpu_id}] CRITICAL ERROR during initialization:")
                traceback.print_exc()
                raise e

            # Build and cache global test batches on the worker if use_global_test is enabled
            self.test_global_batches = []
            if getattr(args, 'use_global_test', False):
                import torchvision
                from torchvision import transforms
                from torch.utils.data import DataLoader
                
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
                        transforms.Normalize([0.5]),
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
                    if args.dataset == 'cifar100':
                        d_cls = torchvision.datasets.CIFAR100
                    else:
                        d_cls = torchvision.datasets.CIFAR10
                    raw = d_cls(root=data_dir, train=False, download=True, transform=ts)
                
                test_global_loader = DataLoader(raw, batch_size=256, shuffle=False, num_workers=0)
                for x, target in test_global_loader:
                    self.test_global_batches.append((x.to(self.device), target.to(self.device)))

        def health_check(self):
            return True

        def evaluate_clients(self, client_weights_dict):
            import torch
            from utils.hetero_models import create_hetero_model
            
            results = {}
            for clnt_idx, w_model in client_weights_dict.items():
                arch = self.client_archs[clnt_idx]
                
                model = create_hetero_model(arch, num_classes=self.num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
                model.to(self.device)
                model.load_state_dict(w_model, strict=True)
                model.eval()
                
                correct_by_class = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
                total_by_class = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    for x, target in self.test_global_batches:
                        pred = model(x)
                        _, predicted = torch.max(pred, -1)
                        
                        target_flat = target.view(-1)
                        predicted_flat = predicted.view(-1)
                        
                        total_by_class += torch.bincount(target_flat, minlength=self.num_classes).float()
                        correct_mask = predicted_flat.eq(target_flat)
                        if correct_mask.any():
                            correct_by_class += torch.bincount(target_flat[correct_mask], minlength=self.num_classes).float()
                            
                corr_cpu = correct_by_class.cpu()
                tot_cpu = total_by_class.cpu()
                per_class_correct = {cls: int(corr_cpu[cls].item()) for cls in range(self.num_classes)}
                per_class_total   = {cls: int(tot_cpu[cls].item()) for cls in range(self.num_classes)}
                
                results[clnt_idx] = {
                    'correct': per_class_correct,
                    'total':   per_class_total,
                }
            return results

        def train_clients(self, round_idx, w_per_mdls_lstrd, client_class_weights, classwise_accs, compare_local, w_standalone_per_mdls,
                          client_val_class_counts=None, client_val_class_correct=None, client_test_class_counts=None, client_test_class_correct=None):
            import copy
            import torch
            import time
            import numpy as np
            
            # Local replica of np_benefit_choose
            def local_benefit_choose(r_idx, c_idx, num_clnts, clnt_num_rnd, cs):
                if cs == "full":
                    return np.array([i for i in range(num_clnts) if i != c_idx])
                else:
                    np.random.seed(r_idx + c_idx)
                    return np.random.choice([i for i in range(num_clnts) if i != c_idx], clnt_num_rnd - 1, replace=False)

            results = {}
            for clnt_idx in self.client_ids:
                client = self.clients[clnt_idx]
                arch = self.client_archs[clnt_idx]
                
                # Load weights
                client.model_trainer.model.load_state_dict(w_per_mdls_lstrd[clnt_idx])
                
                client_num_per_round = int(self.args.num_clients * self.args.frac)
                nei_indexs = local_benefit_choose(round_idx, clnt_idx, self.args.num_clients, client_num_per_round, self.args.cs)
                if self.args.num_clients != client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)
                nei_indexs = sorted(list(set(nei_indexs)))
                
                neighbor_info = []
                if self.args.algorithm not in ("FedAvg", "Standalone"):
                    for n_idx in nei_indexs:
                        if n_idx != clnt_idx:
                            if getattr(self.args, 'idea', 'crad') in ('baseline', 'uniform'):
                                neighbor_info.append((w_per_mdls_lstrd[n_idx], self.client_archs[n_idx], None, n_idx))
                            else:
                                count_vec = torch.tensor(self.class_counts[n_idx])
                                acc_vec = torch.nan_to_num(classwise_accs[n_idx].float(), nan=0.0)
                                count_vec = torch.nan_to_num(count_vec.float(), nan=0.0)
                                eps = getattr(self.args, 'eps_var', 1e-6)
                                tilde_a = (acc_vec * count_vec + 2.0) / (count_vec + 4.0)
                                scaled_rel = 1.0 / ((tilde_a * (1.0 - tilde_a)) / (count_vec + 4.0) + eps)
                                zero_mask = (count_vec == 0)
                                k_zero_strat = getattr(self.args, 'k_zero_strategy', 'drop')
                                if k_zero_strat == 'drop':
                                    scaled_rel[zero_mask] = 0.0
                                else:
                                    prior_weight = 1.0 / (0.0625 + eps)
                                    scaled_rel[zero_mask] = prior_weight
                                scaled_rel = torch.nan_to_num(scaled_rel, nan=0.0, posinf=0.0, neginf=0.0)
                                neighbor_info.append((w_per_mdls_lstrd[n_idx], self.client_archs[n_idx], scaled_rel, count_vec.float(), n_idx))

                class_weights = client_class_weights[clnt_idx]
                
                start_time = time.perf_counter()
                print(f"  Round: {round_idx} Client: {clnt_idx}/{self.args.num_clients} [{arch}] (on Ray GPU worker) Training...")
                
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
                print(f"    ⏱ Client {clnt_idx} [{arch}] done in {elapsed:.1f}s")
                
                results[clnt_idx] = {
                    'clnt_idx': clnt_idx,
                    'w_local_mdl': {k: v.cpu() for k, v in w_local_mdl.items()},
                    'p_test_before_metric': p_test_before_metric,
                    'comm_params': comm_params,
                    'standalone_acc': standalone_acc,
                    'w_standalone_new': w_standalone_new,
                }
                
            return results
except ImportError:
    pass


class dispfd(object):

    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args   = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
         class_counts] = dataset[:8]

        self.val_data_local_dict = dataset[8] if len(dataset) >= 9 else test_data_local_dict
        self.val_cls_counts_dict = dataset[9] if len(dataset) >= 10 else {}

        self.train_global            = train_data_global
        self.test_global             = test_data_global
        self.val_global              = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total  = test_data_num
        self.client_list             = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict     = train_data_local_dict
        self.test_data_local_dict      = test_data_local_dict
        self.class_counts            = class_counts
        self.num_classes             = len(self.class_counts[0]) if self.class_counts else 10
        from utils.resource_tracker import ResourceTracker
        method_name = getattr(args, 'name', '') or getattr(args, 'exp_tag', '') or getattr(args, 'algorithm', 'TypicalKD')
        if hasattr(args, 'idea') and args.idea and method_name == getattr(args, 'algorithm', 'TypicalKD'):
            method_name = f"{args.algorithm}_{args.idea}"
        self.resource_tracker = ResourceTracker(
            method=method_name,
            dataset=getattr(args, 'dataset', 'cifar10'),
            enabled=getattr(args, 'track_costs', True),
        )

        # Whether to evaluate missing-class accuracy on the global test set.
        # This is an "Oracle" evaluation: it measures how well knowledge has 
        # generalized globally, but this data is NEVER used for training.
        # Opt-in flag: preserves fully decentralised semantics by default.
        self.use_global_test = getattr(args, 'use_global_test', False)
        self.test_global_batches = None

        # ── Build global IID test loader if needed ────────────────────────────
        # The data_loader always returns None for train/test_global (it only
        # builds per-client partitioned loaders). We build the full IID test
        # set here from the already-downloaded data so --use_global_test works.
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
                if args.dataset == 'cifar100':
                    d_cls = torchvision.datasets.CIFAR100
                else:
                    d_cls = torchvision.datasets.CIFAR10
                raw = d_cls(root=data_dir, train=False, download=True, transform=ts)
            self.test_global = DataLoader(
                raw, batch_size=256, shuffle=False, num_workers=0
            )
            logger.info(
                f"[GlobalTest] Built IID test loader from {data_dir} "
                f"({len(raw)} samples, dataset={args.dataset})"
            )
            print(f"[GlobalTest] IID test loader ready: {len(raw)} samples")

        # Pre-load all batches directly to GPU VRAM to eliminate DataLoader CPU/GPU transfer overhead
        if self.use_global_test and getattr(args, 'cache_global_test', False) and self.test_global is not None:
            eval_loader = self.test_global
            if hasattr(self.test_global, 'dataset'):
                eval_loader = DataLoader(self.test_global.dataset, batch_size=256, shuffle=False, num_workers=0)
            self.test_global_batches = []
            print(f"🚀 [GlobalTest] Pre-loading all test batches directly to {self.device} memory...")
            for x, target in eval_loader:
                self.test_global_batches.append((x.to(self.device), target.to(self.device)))
            print(f"✅ [GlobalTest] Pre-loading complete. Cached {len(self.test_global_batches)} batches in VRAM.")

        if self.use_global_test:
            logger.info(
                "[INFO] Global test evaluation ENABLED — accuracy measured "
                "on the full IID test set (research eval only; does not leak into training)."
            )
        else:
            logger.info(
                "[INFO] Global test evaluation DISABLED — accuracy measured "
                "on local partitioned test set only."
            )

        # Architecture assignment (round-robin across ARCH_POOL)
        if getattr(args, 'algorithm', '') == 'FedAvg':
            self.client_archs = [args.model] * args.num_clients
            logger.info(f"FedAvg mode detected. Forcing homogeneous architecture: {args.model}")
        else:
            arch_pool = getattr(args, 'arch_pool', ["resnet18", "resnet18_half", "cnn6"])
            self.client_archs = [arch_pool[i % len(arch_pool)] for i in range(args.num_clients)]

        logger.info(f"Heterogeneous architecture assignment: {self.client_archs}")

        # Check if we have a checkpoint to load weights from
        self.checkpoint_data = None
        self.start_round = 0
        if getattr(args, 'load_checkpoint', ''):
            print(f"📥 Loading checkpoint weights from: {args.load_checkpoint}")
            self.checkpoint_data = torch.load(args.load_checkpoint, map_location=self.device)
            # If the checkpoint has client_archs, restore client_archs as well
            if 'client_archs' in self.checkpoint_data:
                self.client_archs = list(self.checkpoint_data['client_archs'])
                if len(self.client_archs) < args.num_clients:
                    for i in range(len(self.client_archs), args.num_clients):
                        self.client_archs.append(arch_pool[i % len(arch_pool)])
                print(f"🔄 Restored and extended client architectures: {self.client_archs}")
            
            # Setup resume parameters if requested
            if getattr(args, 'resume', False):
                saved_round = self.checkpoint_data.get('round', -1)
                self.start_round = saved_round + 1
                print(f"📶 Resuming from round {self.start_round}")

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
        )

        if getattr(args, 'resume', False) and self.checkpoint_data is not None:
            if 'stat_info' in self.checkpoint_data:
                self.stat_info = self.checkpoint_data['stat_info']
                # Trim lists to exactly start_round to avoid trailing items if we are re-running from an intermediate checkpoint
                for key in ["person_test_acc_before", "missing_class_acc", "present_class_acc"]:
                    if key in self.stat_info and isinstance(self.stat_info[key], list):
                        self.stat_info[key] = self.stat_info[key][:self.start_round]
            else:
                self.init_stat_info()
        else:
            self.init_stat_info()

        self._analyze_data_distribution()

        # Side-by-side comparison support
        self.compare_local = getattr(args, 'compare_local', False)
        self.w_standalone_per_mdls = None
        if self.compare_local:
            # Each client maintains a second 'Local Only' model state
            self.w_standalone_per_mdls = [
                self.client_list[i].model_trainer.get_model_params()
                for i in range(self.args.num_clients)
            ]
            logger.info("[CompareLocal] Side-by-side training ENABLED. Tracking shadow 'Local Only' models.")

        # Parallel client training support
        self.parallel = getattr(args, 'parallel', False)
        if self.parallel:
            logger.info("[Parallel] Parallel client training ENABLED via ThreadPoolExecutor.")
            print("[Parallel] ⚡ Parallel client training ENABLED")

    # ── setup ─────────────────────────────────────────────────────────────────

    def _analyze_data_distribution(self):
        num_classes = self.num_classes
        self.client_class_counts = torch.zeros(self.args.num_clients, num_classes)
        self.client_val_class_counts = torch.zeros(self.args.num_clients, num_classes)
        self.client_present_classes = {i: set() for i in range(self.args.num_clients)}
        for client_idx in range(self.args.num_clients):
            for x, labels in self.train_data_local_dict[client_idx]:
                for y in labels:
                    c = int(y.item())
                    self.client_present_classes[client_idx].add(c)
                    self.client_class_counts[client_idx, c] += 1
            present = sorted(list(self.client_present_classes[client_idx]))
            self.logger.info(f"Client {client_idx} [{self.client_archs[client_idx]}] "
                             f"Present Classes: {present}")

        # Compute validation sample counts per class for Task 2 (Eq. 6 m_j^c)
        for client_idx in range(self.args.num_clients):
            if hasattr(self, 'val_cls_counts_dict') and client_idx in self.val_cls_counts_dict and self.val_cls_counts_dict[client_idx]:
                for c, count in self.val_cls_counts_dict[client_idx].items():
                    if c < num_classes:
                        self.client_val_class_counts[client_idx, c] = count
            elif hasattr(self, 'val_data_local_dict') and client_idx in self.val_data_local_dict:
                for x, labels in self.val_data_local_dict[client_idx]:
                    for y in labels:
                        c = int(y.item())
                        if c < num_classes:
                            self.client_val_class_counts[client_idx, c] += 1
            else:
                self.client_val_class_counts[client_idx] = self.client_class_counts[client_idx].clone()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict,
                       test_data_local_dict):
        self.logger.info("############ setup_clients (START) ############")
        num_classes = self.num_classes

        for client_idx in range(self.args.num_clients):
            arch = self.client_archs[client_idx]
            model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
            
            # If checkpoint has client weights, load them
            if self.checkpoint_data is not None and 'client_weights' in self.checkpoint_data:
                if getattr(self.args, 'reset_active_client', False) and client_idx == self.args.num_clients - 1:
                    print(f"🔄 [SANDBOX] Resetting client {client_idx} ({arch}) to randomly initialized weights.")
                elif client_idx < len(self.checkpoint_data['client_weights']):
                    weights = self.checkpoint_data['client_weights'][client_idx]
                    model.load_state_dict(weights)
                    print(f" Loaded weights for client {client_idx} ({arch})")
                else:
                    print(f" No checkpoint weights for client {client_idx} ({arch}), leaving randomly initialized.")

            trainer = MyModelTrainer(model, self.args, self.logger)
            trainer._arch_name = arch

            val_data = self.val_data_local_dict[client_idx] if hasattr(self, 'val_data_local_dict') and client_idx in self.val_data_local_dict else test_data_local_dict[client_idx]

            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                trainer,
                self.logger,
                arch_name=arch,
                local_val_data=val_data,
            )
            self.client_list.append(c)

        self.logger.info("############ setup_clients (END) ############")

    # ── main training loop ───────────────────────────────────────────────────

    def _train_single_client(self, clnt_idx, round_idx, w_per_mdls_lstrd, num_classes, class_weights=None):
        """Train a single client for one round. Returns all outputs needed by the orchestrator.
        
        This method is designed to be called both sequentially and in parallel.
        It is self-contained: it reads from w_per_mdls_lstrd (frozen snapshot) and
        returns results without mutating any shared state.
        """
        client = self.client_list[clnt_idx]
        if self.parallel:
            torch.set_num_threads(1)
            print(f"  Round: {round_idx} Client: {clnt_idx}/{self.args.num_clients} [{self.client_archs[clnt_idx]}] (on {client.device}) Training...")
        start_time = time.perf_counter()
        client_num_per_round = int(self.args.num_clients * self.args.frac)
        nei_indexs = self._benefit_choose(
            round_idx, clnt_idx,
            self.args.num_clients, client_num_per_round, self.args.cs
        )
        if self.args.num_clients != client_num_per_round:
            nei_indexs = np.append(nei_indexs, clnt_idx)
        nei_indexs = np.sort(nei_indexs)

        comm_params = 0  # Communication overhead for this client

        if self.args.algorithm == "FedAvg":
            # ── Decentralized Parameter Averaging (FedAvg) ────────────
            w_local_mdl = self._aggregate_func(
                clnt_idx, self.args.num_clients,
                client_num_per_round, nei_indexs, w_per_mdls_lstrd
            )
            actual_neighbors = [n_idx for n_idx in nei_indexs if n_idx != clnt_idx]
            param_count = sum(p.numel() for p in w_local_mdl.values())
            comm_params = param_count * len(actual_neighbors)

            w_local_mdl, p_test_before_metric = client.train(
                copy.deepcopy(w_local_mdl), None, round_idx
            )

        elif self.args.algorithm == "Standalone":
            # ── Local Only (No Federation, No KD) ─────────────────────
            w_local_mdl, p_test_before_metric = client.train(
                copy.deepcopy(w_per_mdls_lstrd[clnt_idx]),
                None,
                round_idx,
            )

        else:
            # ── Decentralized Knowledge Distillation (TypicalKD) ──────────────────────
            n_0 = getattr(self.args, 'n0', 10.0)
            neighbor_info = []
            for n_idx in nei_indexs:
                if n_idx != clnt_idx:
                    if getattr(self.args, 'idea', 'crad') in ('baseline', 'uniform'):
                        neighbor_info.append((w_per_mdls_lstrd[n_idx], self.client_archs[n_idx], None, n_idx))
                    else:
                        if getattr(self.args, 'fix_eq6', True) and hasattr(self, 'client_val_class_counts') and self.client_val_class_counts is not None:
                            count_vec = torch.nan_to_num(self.client_val_class_counts[n_idx].float(), nan=0.0)
                        else:
                            count_vec = torch.nan_to_num(self.client_class_counts[n_idx].float(), nan=0.0)

                        acc_vec = torch.nan_to_num(self.classwise_accs[n_idx].float(), nan=0.0)
                        eps = getattr(self.args, 'eps_var', 1e-6)
                        tilde_a = (acc_vec * count_vec + 2.0) / (count_vec + 4.0)
                        scaled_rel = 1.0 / ((tilde_a * (1.0 - tilde_a)) / (count_vec + 4.0) + eps)
                        zero_mask = (count_vec == 0)
                        k_zero_strat = getattr(self.args, 'k_zero_strategy', 'drop')
                        if k_zero_strat == 'drop':
                            scaled_rel[zero_mask] = 0.0
                        else:
                            prior_weight = 1.0 / (0.0625 + eps)
                            scaled_rel[zero_mask] = prior_weight
                        scaled_rel = torch.nan_to_num(scaled_rel, nan=0.0, posinf=0.0, neginf=0.0)
                        train_counts = torch.nan_to_num(self.client_class_counts[n_idx].float(), nan=0.0)
                        neighbor_info.append((w_per_mdls_lstrd[n_idx], self.client_archs[n_idx], scaled_rel, train_counts, n_idx))
            # Comm overhead = receiving one model per neighbor
            total_neighbor_params = sum(
                sum(p.numel() for p in entry[0].values()) 
                for entry in neighbor_info
            )
            comm_params = total_neighbor_params

            w_local_mdl, p_test_before_metric = client.train(
                copy.deepcopy(w_per_mdls_lstrd[clnt_idx]),
                neighbor_info,
                round_idx,
                class_weights=class_weights,
            )

        # ── Side-by-Side Standalone Comparison ────────────────────────
        standalone_acc = 0.0
        if self.compare_local:
            w_stand_new, stand_metrics = client.train(
                copy.deepcopy(self.w_standalone_per_mdls[clnt_idx]),
                None,
                round_idx,
            )
            self.w_standalone_per_mdls[clnt_idx] = copy.deepcopy(w_stand_new)
            standalone_acc = (
                stand_metrics['test_correct'] / stand_metrics['test_total']
                if stand_metrics['test_total'] > 0 else 0.0
            )

        elapsed = time.perf_counter() - start_time
        if self.parallel:
            print(f"    ⏱ Client {clnt_idx} [{self.client_archs[clnt_idx]}] done in {elapsed:.1f}s")

        return {
            'clnt_idx': clnt_idx,
            'w_local_mdl': w_local_mdl,
            'p_test_before_metric': p_test_before_metric,
            'comm_params': comm_params,
            'standalone_acc': standalone_acc,
        }

    def train(self):
        # Track total training wall-clock time
        training_start = time.perf_counter()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        start_round = getattr(self, 'start_round', 0)
        self.best_val_acc = 0.0
        self.best_val_round = 0
        self.best_local_test_acc_at_val = 0.0
        self.best_global_test_acc_at_val = 0.0
        self.best_missing_acc_at_val = 0.0
        self.best_present_acc_at_val = 0.0
        self.best_local_acc = 0.0
        self.best_global_acc = 0.0
        if start_round > 0 and hasattr(self, 'checkpoint_data') and self.checkpoint_data is not None:
            self.best_val_acc = self.checkpoint_data.get('best_val_acc', self.checkpoint_data.get('best_local_acc', 0.0))
            self.best_val_round = self.checkpoint_data.get('best_val_round', self.checkpoint_data.get('round', 0))
            self.best_local_test_acc_at_val = self.checkpoint_data.get('best_local_test_acc_at_val', 0.0)
            self.best_global_test_acc_at_val = self.checkpoint_data.get('best_global_test_acc_at_val', self.checkpoint_data.get('best_global_acc', 0.0))
            self.best_local_acc = self.best_val_acc
            self.best_global_acc = self.best_global_test_acc_at_val
            # Fallbacks
            if self.best_val_acc == 0.0 and 'person_val_acc_before' in self.stat_info:
                p_accs = self.stat_info['person_val_acc_before']
                if p_accs:
                    self.best_val_acc = max(p_accs)
            elif self.best_val_acc == 0.0 and 'person_test_acc_before' in self.stat_info:
                p_accs = self.stat_info['person_test_acc_before']
                if p_accs:
                    self.best_val_acc = max(p_accs)
            if self.best_global_acc == 0.0 and 'last_overall_acc' in self.stat_info:
                self.best_global_acc = self.stat_info.get('last_overall_acc', 0.0)

        # Initialise per-client weight stores (each has its own arch)
        w_per_mdls = [
            self.client_list[i].model_trainer.get_model_params()
            for i in range(self.args.num_clients)
        ]

        num_classes = len(self.class_counts[0]) if self.class_counts else 10
        self.classwise_accs = [torch.ones(num_classes) / num_classes for _ in range(self.args.num_clients)]

        # Per-client validation-set counts & correct counts for Tasks 1 & 2
        if hasattr(self, 'client_val_class_counts') and isinstance(self.client_val_class_counts, torch.Tensor):
            val_counts_tensor = self.client_val_class_counts.clone()
            self.client_val_class_counts = [val_counts_tensor[i].clone() for i in range(self.args.num_clients)]
        else:
            self.client_val_class_counts = [torch.zeros(num_classes) for _ in range(self.args.num_clients)]
            if hasattr(self, 'val_cls_counts_dict') and self.val_cls_counts_dict:
                for ci in range(self.args.num_clients):
                    if ci in self.val_cls_counts_dict:
                        for c, cnt in self.val_cls_counts_dict[ci].items():
                            if c < num_classes:
                                self.client_val_class_counts[ci][c] = float(cnt)

        self.client_val_class_correct = [torch.zeros(num_classes) for _ in range(self.args.num_clients)]
        self.client_test_class_counts = [torch.zeros(num_classes) for _ in range(self.args.num_clients)]
        self.client_test_class_correct = [torch.zeros(num_classes) for _ in range(self.args.num_clients)]

        # Populate classwise_accs from checkpoint if resuming
        if hasattr(self, 'checkpoint_data') and self.checkpoint_data is not None:
            print("🔍 Evaluating loaded/checkpoint models to initialize class-wise validation statistics...")
            for ci in range(self.args.num_clients):
                client = self.client_list[ci]
                metrics = client.local_val(client.model_trainer.get_model_params())
                for c in range(num_classes):
                    tot = metrics['per_class_total'].get(c, 0)
                    corr = metrics['per_class_correct'].get(c, 0)
                    self.client_val_class_counts[ci][c] = float(tot)
                    self.client_val_class_correct[ci][c] = float(corr)
                    if tot > 0:
                        self.classwise_accs[ci][c] = corr / tot
                    else:
                        self.classwise_accs[ci][c] = 0.0
            print("✅ Class-wise validation statistics initialized from checkpoint.")

        # Apply Static Privacy Noise to shared validation counts if enabled (E5: Privacy Variant)
        privacy_noise = getattr(self.args, 'privacy_noise', 'none')
        privacy_eps = getattr(self.args, 'privacy_epsilon', 1.0)
        if privacy_noise == 'laplace':
            for ci in range(self.args.num_clients):
                noise = torch.from_numpy(np.random.laplace(0.0, 1.0 / max(privacy_eps, 1e-6), size=self.client_val_class_counts[ci].shape)).float()
                self.client_val_class_counts[ci] = torch.clamp(torch.round(self.client_val_class_counts[ci] + noise), min=0.0)
            self.logger.info(f"🔒 [E5 Privacy] Applied static Laplace noise (eps={privacy_eps}) to shared validation counts.")
            print(f"🔒 [E5 Privacy] Applied static Laplace noise (eps={privacy_eps}) to shared validation counts.")
        elif privacy_noise == 'bucket':
            for ci in range(self.args.num_clients):
                cnts = self.client_val_class_counts[ci]
                k_b = torch.zeros_like(cnts)
                k_b[(cnts >= 1) & (cnts <= 10)] = 5.5
                k_b[(cnts >= 11) & (cnts <= 50)] = 30.5
                k_b[(cnts >= 51) & (cnts <= 200)] = 125.5
                k_b[cnts > 200] = 250.0
                self.client_val_class_counts[ci] = k_b
            self.logger.info(f"🔒 [E5 Privacy] Applied static midpoint bucketing to shared validation counts.")
            print(f"🔒 [E5 Privacy] Applied static midpoint bucketing to shared validation counts.")

        # Auto-detect lowest sample count class for client 10 (Client 11)
        if getattr(self.args, 'distill_target_class', -1) == -1 and self.args.num_clients >= 11:
            counts = self.client_class_counts[10]
            target_cls = int(torch.argmin(counts).item())
            self.args.distill_target_class = target_cls
            print(f"\n💡 [AUTO-DETECT] Client 10's lowest sample class is Class {target_cls} "
                  f"(Count: {int(counts[target_cls].item())}). "
                  f"Setting --distill_target_class to {target_cls} for Client 11 (index 11).\n")
            self.logger.info(f"[AUTO-DETECT] distill_target_class set to {target_cls}")

        # Initialize curriculum state trackers
        self.client_target_class = [None] * self.args.num_clients

        # Setup persistent worker pool for multi-GPU/Ray training
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_ray = getattr(self.args, 'use_ray', False)
        use_persistent_workers = getattr(self.args, 'multigpu', False) and num_gpus >= 2 and not use_ray

        processes = []
        task_queues = []
        result_queues = []
        self.ray_actors = []

        if use_ray:
            import ray
            print("🔗 Connecting to existing Ray cluster...")
            ray.init(address='auto')
            total_ray_gpus = int(ray.cluster_resources().get('GPU', 0))
            
            # Dynamically detect how many GPUs SLURM allocated to this job
            slurm_gpus_env = os.environ.get('SLURM_GPUS')
            if slurm_gpus_env is not None:
                try:
                    target_gpus = int(slurm_gpus_env)
                except ValueError:
                    target_gpus = 1
            else:
                target_gpus = total_ray_gpus # Fallback if not running in SLURM
            
            self.logger.info(f"Target GPUs expected: {target_gpus}")
            
            # Wait up to 45 seconds for all worker nodes to register their GPUs in Ray
            wait_timeout = 45
            elapsed = 0
            while total_ray_gpus < target_gpus and elapsed < wait_timeout:
                print(f"⏳ Waiting for Ray workers to register GPUs... (Current: {total_ray_gpus}/{target_gpus}, Elapsed: {elapsed}s)")
                time.sleep(5)
                elapsed += 5
                total_ray_gpus = int(ray.cluster_resources().get('GPU', 0))
            
            self.logger.info(f"Registered cluster GPUs: {total_ray_gpus}")
            
            if total_ray_gpus > 0:
                num_workers = min(self.args.num_clients, total_ray_gpus)
                gpu_fraction = 1.0  # Force 1 whole dedicated GPU per worker
            else:
                num_workers = self.args.num_clients
                gpu_fraction = 0.0
                
            print(f"🚀 Initializing {num_workers} persistent Ray workers (1 per dedicated GPU)...")
            print(f"⚙️ Ray Actor Resource Allocation: {gpu_fraction:.2f} GPU per worker")

            for worker_id in range(num_workers):
                # The modulo operation here safely chunks your 10 clients across your 8 workers:
                # Worker 0 gets: Client 0, Client 8
                # Worker 1 gets: Client 1, Client 9
                # Workers 2-7 get: Clients 2-7 respectively
                worker_client_ids = [clnt_idx for clnt_idx in range(self.args.num_clients) if clnt_idx % num_workers == worker_id]
                if not worker_client_ids:
                    continue
                
                # Instantiate worker with a full dedicated GPU allocation
                actor = RayTrainerActor.options(num_gpus=gpu_fraction).remote(
                    worker_id,
                    self.args,
                    self.client_archs,
                    worker_client_ids
                )
                self.ray_actors.append(actor)
                
            print("⏳ Waiting for Ray workers to load data and partition...")
            ray.get([actor.health_check.remote() for actor in self.ray_actors])
            print("✅ All Ray persistent workers initialized across the cluster.")

        elif use_persistent_workers:
            import torch.multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
                
            from experiments.run_client_batch import persistent_worker
            
            print(f"🚀 Initializing persistent workers on {num_gpus} GPUs...")
            for gpu_id in range(num_gpus):
                tq = mp.Queue()
                rq = mp.Queue()
                task_queues.append(tq)
                result_queues.append(rq)
                
                p = mp.Process(target=persistent_worker, args=(gpu_id, tq, rq))
                p.start()
                processes.append(p)
                
            # Send init task
            for gpu_id in range(num_gpus):
                gpu_client_ids = [clnt_idx for clnt_idx in range(self.args.num_clients) if clnt_idx % num_gpus == gpu_id]
                task_queues[gpu_id].put({
                    'type': 'init',
                    'args': self.args,
                    'client_archs': self.client_archs,
                    'client_ids': gpu_client_ids
                })
                
            # Wait for confirmation with status monitoring
            print("⏳ Waiting for workers to load data...")
            for gpu_id in range(num_gpus):
                # Check if process is still alive while waiting
                while result_queues[gpu_id].empty():
                    if not processes[gpu_id].is_alive():
                        exit_code = processes[gpu_id].exitcode
                        err_msg = f"❌ Worker on GPU {gpu_id} died with exit code {exit_code} during initialization."
                        print(err_msg)
                        raise RuntimeError(err_msg)
                    time.sleep(1)
                
                res = result_queues[gpu_id].get()
                if res.get('type') == 'error':
                    err_msg = f"❌ Worker on GPU {gpu_id} initialization failed: {res.get('error')}"
                    print(err_msg)
                    raise RuntimeError(err_msg)
            print("✅ Persistent workers initialized.")
            import atexit
            def cleanup_workers():
                print("🧹 Cleaning up persistent workers...")
                for gpu_id in range(num_gpus):
                    try:
                        task_queues[gpu_id].put(None)
                    except Exception:
                        pass
                for p in processes:
                    try:
                        p.join(timeout=2)
                        if p.is_alive():
                            p.terminate()
                    except Exception:
                        pass
                print("✅ Cleanup complete.")
            atexit.register(cleanup_workers)

        for round_idx in range(start_round, self.args.num_rounds):
            self.resource_tracker.start_round(round_idx)
            round_start = time.perf_counter()
            print("Communication round : {}".format(round_idx))
            self.logger.info("#### Communication round : {}".format(round_idx))

            # Calculate class-wise weights if curriculum is active
            curr_mode = getattr(self.args, 'curriculum_target_class_mode', 'none')
            curr_start = getattr(self.args, 'curriculum_start_round', 100)
            threshold = getattr(self.args, 'curriculum_target_threshold', 0.95)
            w_keep_alive = getattr(self.args, 'curriculum_keep_alive_weight', 0.2)
            w_target = getattr(self.args, 'curriculum_target_weight', 2.0)

            boost_idx = getattr(self.args, 'boost_class_idx', -1)
            boost_weight = getattr(self.args, 'boost_class_weight', 1.0)

            client_class_weights = [None] * self.args.num_clients
            use_dyn_scaling = getattr(self.args, 'use_dynamic_scaling', False)
            use_drag_scaling = getattr(self.args, 'use_drag_scaling', False)
            scaling_max_boost = getattr(self.args, 'scaling_max_boost', 0.02)
            scaling_max_reduction = getattr(self.args, 'scaling_max_reduction', 0.005)
            scaling_decay_end = getattr(self.args, 'scaling_decay_end_round', 200)

            if curr_mode != "none" and round_idx >= curr_start:
                for c_idx in range(self.args.num_clients):
                    weights = torch.ones(num_classes)
                    if curr_mode == "soft_mask":
                        # Scale down KD weight for classes with acc >= threshold
                        for class_c in range(num_classes):
                            acc = self.classwise_accs[c_idx][class_c].item()
                            if acc >= threshold:
                                weights[class_c] = w_keep_alive
                        client_class_weights[c_idx] = weights
                    elif curr_mode == "worst_only":
                        curr_target = self.client_target_class[c_idx]
                        promote = False
                        if curr_target is None:
                            promote = True
                        else:
                            curr_acc = self.classwise_accs[c_idx][curr_target].item()
                            if curr_acc >= threshold:
                                promote = True
                                prev_target = curr_target
                                prev_acc = curr_acc
                        
                        if promote:
                            accs = self.classwise_accs[c_idx]
                            worst_class = None
                            worst_acc = 999.0
                            for class_c in range(num_classes):
                                acc = accs[class_c].item()
                                if acc < threshold and acc < worst_acc:
                                    worst_acc = acc
                                    worst_class = class_c
                            
                            if worst_class is not None:
                                self.client_target_class[c_idx] = worst_class
                                if curr_target is not None:
                                    msg = (f"🔄 [CURRICULUM] Client {c_idx}: Target Class {prev_target} reached "
                                           f"{prev_acc*100:.2f}% (>= {threshold*100:.1f}%). Promoting next worst class: "
                                           f"Class {worst_class} (Current Acc: {worst_acc*100:.2f}%)")
                                else:
                                    msg = f"🎯 [CURRICULUM] Client {c_idx} initial targeting: Class {worst_class} (Acc: {worst_acc*100:.2f}%)"
                                print(f"\n{msg}\n")
                                self.logger.info(msg)
                            else:
                                # All classes are above threshold
                                if curr_target is not None and curr_target != -1:
                                    msg = f"🏁 [CURRICULUM] Client {c_idx}: All classes reached threshold {threshold*100:.1f}%. Restoring uniform KD."
                                    print(f"\n{msg}\n")
                                    self.logger.info(msg)
                                self.client_target_class[c_idx] = -1
                        
                        active_target = self.client_target_class[c_idx]
                        if active_target is not None and active_target >= 0:
                            weights = torch.full((num_classes,), w_keep_alive)
                            weights[active_target] = w_target
                            client_class_weights[c_idx] = weights
                        else:
                            client_class_weights[c_idx] = torch.ones(num_classes)
                    elif curr_mode == "data_aware":
                        # Data-Aware: reduce KD only for classes with BOTH
                        #   (1) accuracy >= threshold  AND
                        #   (2) local sample count >= min_local_samples
                        # Classes with high acc but low local data depend on KD — keep full weight.
                        min_samples = getattr(self.args, 'curriculum_min_local_samples', 500)
                        for class_c in range(num_classes):
                            acc = self.classwise_accs[c_idx][class_c].item()
                            local_count = int(self.client_class_counts[c_idx][class_c].item())
                            if acc >= threshold and local_count >= min_samples:
                                weights[class_c] = w_keep_alive
                        client_class_weights[c_idx] = weights
                if boost_idx >= 0:
                    for c_idx in range(self.args.num_clients):
                        if client_class_weights[c_idx] is not None:
                            client_class_weights[c_idx][boost_idx] = boost_weight
            elif use_dyn_scaling:
                if round_idx < scaling_decay_end:
                    decay_factor = 1.0 - (round_idx / scaling_decay_end)
                else:
                    decay_factor = 0.0
                
                for c_idx in range(self.args.num_clients):
                    counts = self.client_class_counts[c_idx]
                    min_samples = torch.min(counts).item()
                    max_samples = torch.max(counts).item()
                    
                    weights = torch.ones(num_classes)
                    if max_samples > min_samples:
                        for class_c in range(num_classes):
                            n = counts[class_c].item()
                            ratio = (n - min_samples) / (max_samples - min_samples)
                            # Interpolate: boost at min, reduction at max
                            adjustment = scaling_max_boost * (1.0 - ratio) - scaling_max_reduction * ratio
                            weights[class_c] = 1.0 + adjustment * decay_factor
                    client_class_weights[c_idx] = weights
                if boost_idx >= 0:
                    for c_idx in range(self.args.num_clients):
                        if client_class_weights[c_idx] is not None:
                            client_class_weights[c_idx][boost_idx] = boost_weight
            elif use_drag_scaling:
                if round_idx < scaling_decay_end:
                    decay_factor = 1.0 - (round_idx / scaling_decay_end)
                else:
                    decay_factor = 0.0
                
                drag_mult = getattr(self.args, 'drag_boost_multiplier', 1.0)
                
                for c_idx in range(self.args.num_clients):
                    weights = torch.ones(num_classes)
                    for class_c in range(num_classes):
                        acc = self.classwise_accs[c_idx][class_c].item()
                        drag = (1.0 - acc) * 10.0  # Overall accuracy drag in % points
                        
                        if drag > 5.9:
                            boost = 0.02 * drag_mult
                        elif 3.5 <= drag <= 5.9:
                            boost = 0.01 * drag_mult
                        else:
                            boost = 0.0
                            
                        weights[class_c] = 1.0 + boost * decay_factor
                    client_class_weights[c_idx] = weights
                if boost_idx >= 0:
                    for c_idx in range(self.args.num_clients):
                        if client_class_weights[c_idx] is not None:
                            client_class_weights[c_idx][boost_idx] = boost_weight
            elif getattr(self.args, 'use_hardcoded_reduction', False):
                tiers = getattr(self.args, 'hardcoded_reduction_tiers', 1)
                mult = getattr(self.args, 'hardcoded_reduction_multiplier', 1.0)
                
                for c_idx in range(self.args.num_clients):
                    client_class_weights[c_idx] = torch.ones(num_classes)
                
                # Tier 1: stable client/class pairs (accuracy change < 0.5%) -> 1% * mult reduction
                stable_pairs = [
                    (0, 1), (0, 2), (0, 7),
                    (1, 1), (1, 2), (1, 6),
                    (3, 6), (4, 8), (6, 5),
                    (7, 1), (8, 7), (9, 1)
                ]
                for c, cl in stable_pairs:
                    if c < self.args.num_clients:
                        client_class_weights[c][cl] = 1.0 - 0.01 * mult
                
                # Tier 2: intermediate client/class pairs (0.5% <= change <= 1.0%) -> 0.5% * mult reduction
                if tiers >= 2:
                    mid_pairs = [
                        (2, 9), (5, 6), (6, 9),
                        (8, 1), (8, 8), (8, 9)
                    ]
                    for c, cl in mid_pairs:
                        if c < self.args.num_clients:
                            client_class_weights[c][cl] = 1.0 - 0.005 * mult
                
            elif getattr(self.args, 'use_class_sample_scaling', False):
                ramp_rounds = getattr(self.args, 'class_scaling_ramp_rounds', 100)
                low_threshold = getattr(self.args, 'class_scaling_low_count', 100)
                
                if round_idx < ramp_rounds:
                    ratio = round_idx / ramp_rounds
                    low_kd = 0.5 + ratio * (0.7 - 0.5)
                    high_kd = 0.3 + ratio * (0.5 - 0.3)
                else:
                    low_kd = 0.7
                    high_kd = 0.5
                
                for c_idx in range(self.args.num_clients):
                    counts = self.client_class_counts[c_idx]
                    weights = torch.ones(num_classes)
                    for class_c in range(num_classes):
                        local_count = counts[class_c].item()
                        if local_count < low_threshold:
                            weights[class_c] = low_kd
                        else:
                            weights[class_c] = high_kd
                    client_class_weights[c_idx] = weights
                
                if boost_idx >= 0:
                    for c_idx in range(self.args.num_clients):
                        if client_class_weights[c_idx] is not None:
                            client_class_weights[c_idx][boost_idx] = boost_weight
            elif boost_idx >= 0:
                for c_idx in range(self.args.num_clients):
                    weights = torch.ones(num_classes)
                    weights[boost_idx] = boost_weight
                    client_class_weights[c_idx] = weights



            p_test_metrics_before = {
                'num_samples':       [],
                'num_correct':       [],
                'losses':            [],
                'per_class_correct': [{} for _ in range(self.args.num_clients)],
                'per_class_total':   [{} for _ in range(self.args.num_clients)],
                'standalone_accs':   [],
            }

            # Snapshot of last-round weights (used by all clients this round)
            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)

            if getattr(self.args, 'use_shared_memory', False):
                for w in w_per_mdls_lstrd:
                    if w is not None:
                        for k, t in w.items():
                            t.share_memory_()
                if self.w_standalone_per_mdls is not None:
                    for w in self.w_standalone_per_mdls:
                        if w is not None:
                            for k, t in w.items():
                                t.share_memory_()

            if self.parallel:
                # ── PARALLEL MODE: Train all clients concurrently ──────────
                results = [None] * self.args.num_clients
                
                if use_ray:
                    futures = [
                        actor.train_clients.remote(
                            round_idx,
                            w_per_mdls_lstrd,
                            client_class_weights,
                            self.classwise_accs,
                            self.compare_local,
                            self.w_standalone_per_mdls,
                            client_val_class_counts=self.client_val_class_counts,
                            client_val_class_correct=self.client_val_class_correct,
                            client_test_class_counts=self.client_test_class_counts,
                            client_test_class_correct=self.client_test_class_correct,
                        )
                        for actor in self.ray_actors
                    ]
                    ray_results = ray.get(futures)
                    for actor_res in ray_results:
                        for clnt_idx, client_res in actor_res.items():
                            results[clnt_idx] = client_res
                            if self.compare_local and client_res.get('w_standalone_new') is not None:
                                self.w_standalone_per_mdls[clnt_idx] = copy.deepcopy(client_res['w_standalone_new'])

                elif use_persistent_workers:
                    # Push training task to each worker
                    for gpu_id in range(num_gpus):
                        gpu_client_ids = [clnt_idx for clnt_idx in range(self.args.num_clients) if clnt_idx % num_gpus == gpu_id]
                        task_queues[gpu_id].put({
                            'type': 'train',
                            'round_idx': round_idx,
                            'w_per_mdls_lstrd': w_per_mdls_lstrd,
                            'client_class_weights': client_class_weights,
                            'classwise_accs': self.classwise_accs,
                            'client_val_class_counts': self.client_val_class_counts,
                            'client_val_class_correct': self.client_val_class_correct,
                            'client_test_class_counts': self.client_test_class_counts,
                            'client_test_class_correct': self.client_test_class_correct,
                            'compare_local': self.compare_local,
                            'w_standalone_per_mdls': self.w_standalone_per_mdls,
                            'client_ids': gpu_client_ids
                        })
                    
                    # Gather training results
                    for gpu_id in range(num_gpus):
                        res = result_queues[gpu_id].get()
                        if res.get('type') == 'error':
                            err_msg = f"❌ Persistent worker on GPU {gpu_id} failed: {res.get('error')}"
                            print(err_msg)
                            self.logger.error(err_msg)
                            raise RuntimeError(err_msg)
                        for clnt_idx, client_res in res['results'].items():
                            results[clnt_idx] = client_res
                            if self.compare_local and client_res.get('w_standalone_new') is not None:
                                self.w_standalone_per_mdls[clnt_idx] = copy.deepcopy(client_res['w_standalone_new'])
                else:
                    # Fallback to ThreadPoolExecutor (for single-GPU or CPU parallel mode)
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    max_workers = self.args.num_clients
                    if getattr(self.args, "multigpu", False) and num_gpus > 0:
                        max_workers = num_gpus
                    
                    print(f"  ⚡ Training {self.args.num_clients} clients in parallel (max workers: {max_workers})...")
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                self._train_single_client,
                                clnt_idx, round_idx, w_per_mdls_lstrd, num_classes, client_class_weights[clnt_idx]
                            ): clnt_idx
                            for clnt_idx in range(self.args.num_clients)
                        }
                        for future in as_completed(futures):
                            result = future.result()
                            results[result['clnt_idx']] = result
                
                # Collect all results in order
                for result in results:
                    clnt_idx = result['clnt_idx']
                    p_test_before_metric = result['p_test_before_metric']
                    
                    w_per_mdls[clnt_idx] = copy.deepcopy(result['w_local_mdl'])
                    
                    self.stat_info["comm_overhead_params"] += result['comm_params']
                    self.stat_info["comm_overhead_mb"] += (result['comm_params'] * 4) / (1024**2)
                    self.resource_tracker.add_payload_bytes(result['comm_params'] * 4)
                    
                    p_test_metrics_before['num_samples'].append(p_test_before_metric['test_total'])
                    p_test_metrics_before['num_correct'].append(p_test_before_metric['test_correct'])
                    p_test_metrics_before['losses'].append(p_test_before_metric['test_loss'])
                    p_test_metrics_before['per_class_correct'][clnt_idx] = p_test_before_metric['per_class_correct']
                    p_test_metrics_before['per_class_total'][clnt_idx] = p_test_before_metric['per_class_total']
                    p_test_metrics_before['standalone_accs'].append(result['standalone_acc'])
                    
                    # Update classwise reliability for next round (computed strictly on validation partition)
                    for c in range(num_classes):
                        tot = p_test_before_metric['per_class_total'].get(c, 0)
                        corr = p_test_before_metric['per_class_correct'].get(c, 0)
                        if getattr(self.args, 'privacy_noise', 'none') == 'none':
                            self.client_val_class_counts[clnt_idx][c] = float(tot)
                        self.client_val_class_correct[clnt_idx][c] = float(corr)
                        self.client_test_class_counts[clnt_idx][c] = float(tot)
                        self.client_test_class_correct[clnt_idx][c] = float(corr)
                        if tot > 0:
                            self.classwise_accs[clnt_idx][c] = corr / tot
                        else:
                            self.classwise_accs[clnt_idx][c] = 0.0

            else:
                # ── SEQUENTIAL MODE: Train clients one-by-one with per-client timing ──
                for clnt_idx in range(self.args.num_clients):
                    client_start = time.perf_counter()
                    print('  Round: {} Client: {}/{} [{}] Training'.format(
                        round_idx, clnt_idx, self.args.num_clients,
                        self.client_archs[clnt_idx]
                    ))
                    self.logger.info(
                        f'@@@@ Training Client {clnt_idx} ({self.client_archs[clnt_idx]}) '
                        f'Round {round_idx}'
                    )

                    result = self._train_single_client(clnt_idx, round_idx, w_per_mdls_lstrd, num_classes, client_class_weights[clnt_idx])
                    p_test_before_metric = result['p_test_before_metric']

                    w_per_mdls[clnt_idx] = copy.deepcopy(result['w_local_mdl'])

                    self.stat_info["comm_overhead_params"] += result['comm_params']
                    self.stat_info["comm_overhead_mb"] += (result['comm_params'] * 4) / (1024**2)
                    self.resource_tracker.add_payload_bytes(result['comm_params'] * 4)

                    p_test_metrics_before['num_samples'].append(p_test_before_metric['test_total'])
                    p_test_metrics_before['num_correct'].append(p_test_before_metric['test_correct'])
                    p_test_metrics_before['losses'].append(p_test_before_metric['test_loss'])
                    p_test_metrics_before['per_class_correct'][clnt_idx] = p_test_before_metric['per_class_correct']
                    p_test_metrics_before['per_class_total'][clnt_idx] = p_test_before_metric['per_class_total']
                    p_test_metrics_before['standalone_accs'].append(result['standalone_acc'])

                    # Update classwise reliability for next round (computed strictly on validation partition)
                    for c in range(num_classes):
                        tot = p_test_before_metric['per_class_total'].get(c, 0)
                        corr = p_test_before_metric['per_class_correct'].get(c, 0)
                        if getattr(self.args, 'privacy_noise', 'none') == 'none':
                            self.client_val_class_counts[clnt_idx][c] = float(tot)
                        self.client_val_class_correct[clnt_idx][c] = float(corr)
                        self.client_test_class_counts[clnt_idx][c] = float(tot)
                        self.client_test_class_correct[clnt_idx][c] = float(corr)
                        if tot > 0:
                            self.classwise_accs[clnt_idx][c] = corr / tot
                        else:
                            self.classwise_accs[clnt_idx][c] = 0.0

                    client_elapsed = time.perf_counter() - client_start
                    print(f'    ⏱ Client {clnt_idx} [{self.client_archs[clnt_idx]}] done in {client_elapsed:.1f}s')

            # ── Round-level metrics ───────────────────────────────────────────
            _eval_skipped = round_idx < getattr(self.args, 'skip_eval_until_round', 0)
            if getattr(self.args, 'freeze_teachers', False):
                # Evaluate only the active student (the last client)
                p_val_acc_before = (
                    p_test_metrics_before['num_correct'][-1] /
                    max(1, p_test_metrics_before['num_samples'][-1])
                )
                p_val_loss = (
                    p_test_metrics_before['losses'][-1] /
                    max(1, p_test_metrics_before['num_samples'][-1])
                )
            else:
                if _eval_skipped:
                    p_val_acc_before = float('nan')
                    p_val_loss = float('nan')
                else:
                    p_val_acc_before = sum(
                        np.array(p_test_metrics_before['num_correct'][i]) /
                        np.array(p_test_metrics_before['num_samples'][i])
                        for i in range(self.args.num_clients)
                    ) / self.args.num_clients

                    p_val_loss = sum(
                        np.array(p_test_metrics_before['losses'][i]) /
                        np.array(p_test_metrics_before['num_samples'][i])
                        for i in range(self.args.num_clients)
                    ) / self.args.num_clients

            # Evaluate local test partition (held-out local test set)
            if _eval_skipped:
                p_local_test_acc = float('nan')
                p_local_test_loss = float('nan')
            else:
                p_local_test_acc, p_local_test_loss, _ = self._eval_local_test(w_per_mdls)

            # ── Missing vs present class accuracy (KD evidence metric) ────────
            # Global test is EXPENSIVE (infers all client models on 10 k images).
            # Run only on round 0, the final round, and every 5th round.
            # Between evaluations reuse the last cached result.
            global_per_class = None
            do_global_eval = (
                self.use_global_test
                and self.test_global is not None
                and round_idx >= getattr(self.args, 'skip_eval_until_round', 0)
                and (
                    round_idx == 0
                    or round_idx == self.args.num_rounds - 1
                    or round_idx % 5 == 4          # rounds 4, 9, 14, …
                )
            )
            if do_global_eval:
                global_per_class = self._eval_global_per_class(w_per_mdls)
                missing_acc, present_acc, overall_acc = self._compute_missing_present_acc(
                    global_per_class, source="global"
                )
                # Cache for reuse on off-rounds
                self.stat_info["last_missing_acc"] = missing_acc
                self.stat_info["last_present_acc"] = present_acc
                self.stat_info["last_overall_acc"] = overall_acc
            elif self.use_global_test:
                # Reuse last cached values (or 0.0 on very first rounds)
                missing_acc = self.stat_info.get("last_missing_acc", 0.0)
                present_acc = self.stat_info.get("last_present_acc", 0.0)
                overall_acc = self.stat_info.get("last_overall_acc", 0.0)
            else:
                # Original local path (preserved exactly for ablation)
                local_per_class_correct = p_test_metrics_before['per_class_correct']
                local_per_class_total   = p_test_metrics_before['per_class_total']
                missing_acc, present_acc = self._compute_missing_present_acc_local(
                    local_per_class_correct, local_per_class_total
                )
                overall_acc = p_val_acc_before

            self.stat_info["missing_class_acc"].append(missing_acc)
            self.stat_info["present_class_acc"].append(present_acc)
            self.stat_info["global_class_acc"].append(overall_acc)

            # Per-client class accuracy table (round 0, last, every 5th)
            # Suppress entirely while eval is skipped to avoid all-N/A noise
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
                self._print_class_table(p_test_metrics_before, round_idx, global_per_class=global_per_class)

                # Log data_aware weight decisions alongside the class table
                if curr_mode == "data_aware" and round_idx >= curr_start:
                    min_samples = getattr(self.args, 'curriculum_min_local_samples', 500)
                    # Resolve class names for logging
                    if self.args.dataset == 'pathmnist':
                        _da_names = ['adi', 'bg', 'deb', 'lym', 'muc', 'mus', 'norm', 'str', 'ca']
                    elif self.args.dataset == 'bloodmnist':
                        _da_names = ['baso', 'eosi', 'eryt', 'ig', 'lymp', 'mono', 'neut', 'plat']
                    elif self.args.dataset == 'retinamnist':
                        _da_names = ['norm', 'mild', 'mod', 'sev', 'prol']
                    elif self.args.dataset == 'cifar100':
                        _da_names = [str(i) for i in range(100)]
                    else:
                        _da_names = ['airplane', 'auto', 'bird', 'cat', 'deer',
                                     'dog', 'frog', 'horse', 'ship', 'truck']
                    print(f"\n📊 [DATA_AWARE] Weight decisions (threshold={threshold*100:.0f}%, "
                          f"min_samples={min_samples}, keep_alive={w_keep_alive}):")
                    for c_idx in range(self.args.num_clients):
                        if client_class_weights[c_idx] is not None:
                            reduced = []
                            kept = []
                            for cls_c in range(num_classes):
                                w = client_class_weights[c_idx][cls_c].item()
                                acc = self.classwise_accs[c_idx][cls_c].item()
                                cnt = int(self.client_class_counts[c_idx][cls_c].item())
                                if w < 1.0:
                                    reduced.append(f"{_da_names[cls_c]}({acc*100:.0f}%,{cnt}s)")
                                elif acc >= threshold:
                                    # High acc but kept because low local data
                                    kept.append(f"{_da_names[cls_c]}({acc*100:.0f}%,{cnt}s)")
                            parts = []
                            if reduced:
                                parts.append(f"REDUCED→{', '.join(reduced)}")
                            if kept:
                                parts.append(f"KEPT(KD-dep)→{', '.join(kept)}")
                            if parts:
                                print(f"  Client {c_idx}: {' | '.join(parts)}")
                    print()
                elif getattr(self.args, 'use_class_sample_scaling', False):
                    # Resolve class names for logging
                    if self.args.dataset == 'pathmnist':
                        _da_names = ['adi', 'bg', 'deb', 'lym', 'muc', 'mus', 'norm', 'str', 'ca']
                    elif self.args.dataset == 'bloodmnist':
                        _da_names = ['baso', 'eosi', 'eryt', 'ig', 'lymp', 'mono', 'neut', 'plat']
                    elif self.args.dataset == 'retinamnist':
                        _da_names = ['norm', 'mild', 'mod', 'sev', 'prol']
                    elif self.args.dataset == 'cifar100':
                        _da_names = [str(i) for i in range(100)]
                    else:
                        _da_names = ['airplane', 'auto', 'bird', 'cat', 'deer',
                                     'dog', 'frog', 'horse', 'ship', 'truck']
                    print(f"\n📊 [CLASS_SAMPLE_SCALING] Weight decisions for round {round_idx}:")
                    for c_idx in range(self.args.num_clients):
                        if client_class_weights[c_idx] is not None:
                            low_desc = []
                            high_desc = []
                            for cls_c in range(num_classes):
                                w = client_class_weights[c_idx][cls_c].item()
                                cnt = int(self.client_class_counts[c_idx][cls_c].item())
                                if cnt < getattr(self.args, 'class_scaling_low_count', 100):
                                    low_desc.append(f"{_da_names[cls_c]}({cnt}s,w={w:.2f})")
                                else:
                                    high_desc.append(f"{_da_names[cls_c]}({cnt}s,w={w:.2f})")
                            print(f"  Client {c_idx}:")
                            print(f"    Low-sample:  {', '.join(low_desc)}")
                            print(f"    High-sample: {', '.join(high_desc)}")
                    print()

            self.stat_info["person_val_acc_before"].append(p_val_acc_before)
            self.stat_info["person_test_acc_before"].append(p_local_test_acc)

            # Checkpoint Selection Rule (S1.1 & E1b):
            # Best checkpoint = highest average local validation accuracy across clients
            if not np.isnan(p_val_acc_before) and p_val_acc_before > self.best_val_acc:
                self.best_val_acc = p_val_acc_before
                self.best_val_round = round_idx
                self.best_local_test_acc_at_val = p_local_test_acc
                self.best_global_test_acc_at_val = overall_acc
                self.best_missing_acc_at_val = missing_acc
                self.best_present_acc_at_val = present_acc
                self.best_local_acc = p_val_acc_before
                self.best_global_acc = overall_acc

                if round_idx >= 200 or self.args.num_rounds <= 50 or (round_idx + 1) == self.args.num_rounds:
                    self._save_checkpoint(w_per_mdls, round_idx, suffix="best_val")
                    self._save_checkpoint(w_per_mdls, round_idx, suffix="best_local")

            # Convergence check
            if (self.stat_info["convergence_round"] == -1
                    and p_val_acc_before >= 0.5):
                self.stat_info["convergence_round"] = round_idx
                self.logger.info(
                    f">>> CONVERGENCE at round {round_idx} "
                    f"(acc={p_val_acc_before:.4f})"
                )

            eval_source = "global IID" if self.use_global_test else "local"
            global_eval_tag = " [cached]" if (self.use_global_test and not do_global_eval) else ""
            _eval_skipped = round_idx < getattr(self.args, 'skip_eval_until_round', 0)

            # Prepare comparison string if compare_local is enabled
            compare_str = ""
            if self.compare_local and not _eval_skipped:
                avg_stand = sum(p_test_metrics_before['standalone_accs']) / self.args.num_clients
                gain = p_val_acc_before - avg_stand
                compare_str = f" | Standalone: {avg_stand:.4f} | Gain: {gain*100:+.2f}%"

            name_prefix = f"[{self.args.name}] " if getattr(self.args, 'name', '') else ""

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

            if _eval_skipped:
                # Eval is suppressed — print a minimal training-only summary
                skip_until = getattr(self.args, 'skip_eval_until_round', 0)
                print(
                    "{}Round: {:3d} | [eval skipped until round {}] | Comm: {:.2f}MB | \n⏱ {}".format(
                        name_prefix,
                        round_idx,
                        skip_until,
                        self.stat_info["comm_overhead_mb"],
                        time_str,
                    )
                )
            elif self.use_global_test:
                print(
                    "{}Round: {:3d} | Val Local: {:.4f}{} | Global Acc (IID){}: {:.4f} | Local Test: {:.4f} "
                    "| Present: {:.2f}% | Missing: {:.2f}% | Loss: {:.4f} | Comm: {:.2f}MB | \n⏱ {}".format(
                        name_prefix,
                        round_idx,
                        p_val_acc_before,
                        compare_str,
                        global_eval_tag,
                        overall_acc,
                        p_local_test_acc,
                        present_acc * 100,
                        missing_acc * 100,
                        p_val_loss,
                        self.stat_info["comm_overhead_mb"],
                        time_str,
                    )
                )
            else:
                print(
                    "{}Round: {:3d} | Val Local: {:.4f}{} | Local Test: {:.4f} "
                    "| Present: {:.2f}% | Missing: {:.2f}% (local) "
                    "| Loss: {:.4f} | Comm: {:.2f}MB | \n⏱ {}".format(
                        name_prefix,
                        round_idx,
                        p_val_acc_before,
                        compare_str,
                        p_local_test_acc,
                        present_acc * 100,
                        missing_acc * 100,
                        p_val_loss,
                        self.stat_info["comm_overhead_mb"],
                        time_str,
                    )
                )
            self.logger.info({
                'round': round_idx,
                'val_acc': p_val_acc_before,
                'local_test_acc': p_local_test_acc,
                'global_acc': overall_acc,
                'p_test_acc': p_val_acc_before,
                'p_test_loss': p_val_loss,
                'present_acc': present_acc,
                'missing_acc': missing_acc,
                'missing_acc_source': eval_source,
                'standalone_acc': sum(p_test_metrics_before['standalone_accs']) / self.args.num_clients if self.compare_local else 0.0,
            })

            if (round_idx + 1) % 50 == 0:
                self.logger.info(
                    f'val_acc_so_far={self.stat_info["person_val_acc_before"]}'
                )
                self._save_checkpoint(w_per_mdls, round_idx)

            self.resource_tracker.end_round()
            del p_test_metrics_before, w_per_mdls_lstrd
            gc.collect()

        # ── Final summary ─────────────────────────────────────────────────────
        self.logger.info(
            f'all_val_acc={self.stat_info["person_val_acc_before"]}'
        )
        self.logger.info(
            f'all_test_acc={self.stat_info["person_test_acc_before"]}'
        )
        self.record_avg_inference_flops(w_per_mdls)

        eval_source = "global IID test" if self.use_global_test else "local partitioned test"

        self.logger.info({
            'best_val_round': self.best_val_round,
            'best_val_acc': self.best_val_acc * 100,
            'local_test_at_best_val': self.best_local_test_acc_at_val * 100,
            'global_test_at_best_val': self.best_global_test_acc_at_val * 100,
            'best_missing_acc': self.best_missing_acc_at_val * 100,
            'best_present_acc': self.best_present_acc_at_val * 100,
            'convergence_round': self.stat_info["convergence_round"],
            'total_comm_mb': self.stat_info["comm_overhead_mb"],
        })

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
        print(f"Convergence Round: {self.stat_info['convergence_round']}")
        print(f"Total Communication: {self.stat_info['comm_overhead_mb']:.2f} MB")
        total_elapsed = time.perf_counter() - training_start
        total_min = total_elapsed / 60
        if total_min >= 60:
            print(f"⏱ Total Training Time: {total_min / 60:.1f}h ({total_min:.0f}min)")
        else:
            print(f"⏱ Total Training Time: {total_min:.1f}min ({total_elapsed:.0f}s)")
        
        # ── Save Final Checkpoint ─────────────────────────────────────────────
        self._save_checkpoint(w_per_mdls, self.args.num_rounds - 1, is_final=True)
        
        # ── Resource Usage Summary & JSON Export ────────────────────────────
        self.resource_tracker.save_json()
        self.resource_tracker.print_summary_table()
        print("over")

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

    def _eval_local_test(self, w_per_mdls: list) -> tuple:
        """
        Evaluate every client model strictly on its local test partition (self.local_test_data).
        Returns:
            avg_test_acc: float
            avg_test_loss: float
            test_metrics_dict: dict
        """
        num_clients = self.args.num_clients
        correct_list = []
        total_list = []
        loss_list = []
        per_class_correct = {}
        per_class_total = {}

        for ci in range(num_clients):
            client = self.client_list[ci]
            metrics = client.local_test(w_per_mdls[ci], b_use_test_dataset=True)
            correct_list.append(metrics['test_correct'])
            total_list.append(metrics['test_total'])
            loss_list.append(metrics['test_loss'])
            per_class_correct[ci] = metrics.get('per_class_correct', {})
            per_class_total[ci] = metrics.get('per_class_total', {})

        avg_test_acc = sum(
            correct_list[i] / max(1, total_list[i]) for i in range(num_clients)
        ) / max(1, num_clients)
        avg_test_loss = sum(
            loss_list[i] / max(1, total_list[i]) for i in range(num_clients)
        ) / max(1, num_clients)

        return avg_test_acc, avg_test_loss, {
            'correct': correct_list,
            'total': total_list,
            'loss': loss_list,
            'per_class_correct': per_class_correct,
            'per_class_total': per_class_total,
        }

    # ── Global test evaluation helpers ────────────────────────────

    def _eval_global_per_class(self, w_per_mdls: list) -> dict:
        """
        Evaluate every client's model on the global IID test set.

        Returns:
            global_per_class : dict {client_idx: {'correct': {cls: int},
                                                   'total':   {cls: int}}}

        This runs in no-grad inference mode. Models are moved to device,
        evaluated, then moved back to CPU to avoid VRAM fragmentation.
        Each client uses its own architecture (heterogeneous-safe).
        """
        device = self.device
        num_classes = self.num_classes
        global_per_class = {}

        use_freeze_cache = getattr(self.args, 'freeze_teachers', False)
        if not hasattr(self, '_cached_global_per_class'):
            self._cached_global_per_class = {}

        # If Ray is active, perform evaluation in parallel on Ray workers
        if getattr(self.args, 'use_ray', False) and hasattr(self, 'ray_actors') and self.ray_actors:
            import ray
            
            # Map client indices to their matching Ray worker
            num_workers = len(self.ray_actors)
            worker_client_weights = {wi: {} for wi in range(num_workers)}
            
            for ci in range(self.args.num_clients):
                if use_freeze_cache and ci < self.args.num_clients - 1 and ci in self._cached_global_per_class:
                    global_per_class[ci] = copy.deepcopy(self._cached_global_per_class[ci])
                    continue
                
                worker_id = ci % num_workers
                worker_client_weights[worker_id][ci] = w_per_mdls[ci]
                
            # Submit evaluation tasks to Ray workers in parallel
            futures = []
            active_workers = []
            for wi, client_weights_dict in worker_client_weights.items():
                if client_weights_dict:
                    actor = self.ray_actors[wi]
                    futures.append(actor.evaluate_clients.remote(client_weights_dict))
                    active_workers.append(wi)
                    
            # Gather results
            ray_results = ray.get(futures)
            for res in ray_results:
                for ci, client_res in res.items():
                    global_per_class[ci] = client_res
                    if use_freeze_cache and ci < self.args.num_clients - 1:
                        self._cached_global_per_class[ci] = copy.deepcopy(client_res)
                        
            return global_per_class

        for ci in range(self.args.num_clients):
            if use_freeze_cache and ci < self.args.num_clients - 1 and ci in self._cached_global_per_class:
                global_per_class[ci] = copy.deepcopy(self._cached_global_per_class[ci])
                continue

            arch = self.client_archs[ci]
            if getattr(self.args, 'cache_eval_models', False):
                if not hasattr(self, '_eval_models_cache'):
                    self._eval_models_cache = {}
                if ci not in self._eval_models_cache:
                    model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
                    model.to(device)
                    self._eval_models_cache[ci] = model
                model = self._eval_models_cache[ci]
            else:
                model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
                model.to(device)
            model.load_state_dict(w_per_mdls[ci], strict=True)
            model.eval()

            per_class_correct = {}
            per_class_total   = {}

            use_cached = getattr(self.args, 'cache_global_test', False) and (self.test_global_batches is not None)

            if getattr(self.args, 'fast_eval', False):
                correct_by_class = torch.zeros(num_classes, dtype=torch.float32, device=device)
                total_by_class = torch.zeros(num_classes, dtype=torch.float32, device=device)

                with torch.no_grad():
                    if use_cached:
                        for x, target in self.test_global_batches:
                            pred = model(x)
                            _, predicted = torch.max(pred, -1)

                            target_flat = target.view(-1)
                            predicted_flat = predicted.view(-1)

                            total_by_class += torch.bincount(target_flat, minlength=num_classes).float()
                            correct_mask = predicted_flat.eq(target_flat)
                            if correct_mask.any():
                                correct_by_class += torch.bincount(target_flat[correct_mask], minlength=num_classes).float()
                    else:
                        for x, target in self.test_global:
                            x, target = x.to(device), target.to(device)
                            pred = model(x)
                            _, predicted = torch.max(pred, -1)

                            target_flat = target.view(-1)
                            predicted_flat = predicted.view(-1)

                            total_by_class += torch.bincount(target_flat, minlength=num_classes).float()
                            correct_mask = predicted_flat.eq(target_flat)
                            if correct_mask.any():
                                correct_by_class += torch.bincount(target_flat[correct_mask], minlength=num_classes).float()

                corr_cpu = correct_by_class.cpu()
                tot_cpu = total_by_class.cpu()
                per_class_correct = {cls: int(corr_cpu[cls].item()) for cls in range(num_classes)}
                per_class_total   = {cls: int(tot_cpu[cls].item()) for cls in range(num_classes)}
            else:
                with torch.no_grad():
                    if use_cached:
                        for x, target in self.test_global_batches:
                            pred = model(x)
                            _, predicted = torch.max(pred, -1)

                            for t, p in zip(target.view(-1), predicted.view(-1)):
                                t_val = int(t.item())
                                per_class_correct[t_val] = (
                                    per_class_correct.get(t_val, 0) + int((t == p).item())
                                )
                                per_class_total[t_val] = per_class_total.get(t_val, 0) + 1
                    else:
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

            if not getattr(self.args, 'cache_eval_models', False):
                model.cpu()
                del model
            global_per_class[ci] = {
                'correct': per_class_correct,
                'total':   per_class_total,
            }
            if use_freeze_cache and ci < self.args.num_clients - 1:
                self._cached_global_per_class[ci] = copy.deepcopy(global_per_class[ci])

        torch.cuda.empty_cache()
        return global_per_class

    def _compute_missing_present_acc(self, global_per_class: dict,
                                     source: str = "global"):
        """
        Compute missing/present class accuracy from global per-class stats.

        A class is "missing" for client ci if it is absent from ci's training
        data (not in client_present_classes[ci]). Accuracy is measured on the
        global test set, so zero missing-class accuracy means KD genuinely
        failed, not that the test set was empty.
        """
        total_missing_corr = total_missing_samp = 0
        total_present_corr = total_present_samp = 0

        clients_to_eval = [self.args.num_clients - 1] if getattr(self.args, 'freeze_teachers', False) else range(self.args.num_clients)
        for ci in clients_to_eval:
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

    def _compute_missing_present_acc_local(self, local_corr_list: list,
                                           local_tot_list: list):
        """
        Original local path (unchanged). Preserved for ablation.
        NOTE: will produce missing_acc ≈ 0.00% due to partitioned test data.
        """
        total_missing_corr = total_missing_samp = 0
        total_present_corr = total_present_samp = 0

        clients_to_eval = [self.args.num_clients - 1] if getattr(self.args, 'freeze_teachers', False) else range(self.args.num_clients)
        for ci in clients_to_eval:
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
        return missing_acc, present_acc

    # ── helpers ───────────────────────────────────────────────────────────────

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
        header = f"{'Client':<6} | {'Arch':<14} | {'Total':<6} | " + " | ".join(
            f"{n:<6}" for n in class_names
        )
        print(header)
        print("-" * len(header))
        for ci in range(self.args.num_clients):
            arch = self.client_archs[ci]
            if is_global:
                corr = global_per_class[ci]['correct']
                tot  = global_per_class[ci]['total']
            else:
                corr = p_test_metrics_before['per_class_correct'][ci]
                tot  = p_test_metrics_before['per_class_total'][ci]
            
            # Calculate total weighted accuracy for this client
            total_corr = sum(corr.values())
            total_samples = sum(tot.values())
            total_acc = total_corr / total_samples if total_samples > 0 else 0.0
            
            acc_str = f"{total_acc*100:>5.1f}%"
            if self.compare_local and not is_global:
                stand_acc = p_test_metrics_before['standalone_accs'][ci]
                gain = (total_acc - stand_acc) * 100
                acc_str = f"{total_acc*100:>4.1f}%({gain:>+4.1f})"

            row  = f"{ci:<6} | {arch:<14} | {acc_str:<8} | "
            for cls in range(num_classes):
                t = tot.get(cls, 0)
                row += (f"{(corr.get(cls, 0)/t*100):>5.1f}% | " if t > 0
                        else f"{'N/A':>6} | ")
            print(row)
        print("=" * 115 + "\n")

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total,
                        client_num_per_round, cs=False):
        """
        Topology and neighbor selection.
        Determines which peers a client communicates with in the current round.
        """
        if client_num_in_total == client_num_per_round:
            return list(range(client_num_in_total))
        if cs == "random":
            np.random.seed(round_idx + cur_clnt)
            idxs = np.random.choice(range(client_num_in_total),
                                    client_num_per_round, replace=False)
            while cur_clnt in idxs:
                idxs = np.random.choice(range(client_num_in_total),
                                        client_num_per_round, replace=False)
            return idxs
        elif cs == "ring":
            l = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            r = (cur_clnt + 1) % client_num_in_total
            return np.asarray([l, r])
        elif cs == "grid":
            l = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            r = (cur_clnt + 1) % client_num_in_total
            t = (cur_clnt - 9 + client_num_in_total) % client_num_in_total
            d = (cur_clnt + 9 + client_num_in_total) % client_num_in_total
            return np.asarray([l, r, t, d])
        elif cs == "exp":
            return np.asarray([
                (cur_clnt + 2**i) % client_num_in_total for i in range(7)
            ])
        elif cs == "full":
            idxs = np.arange(client_num_in_total)
            return np.delete(idxs, np.where(idxs == cur_clnt))
        return np.random.choice(range(client_num_in_total),
                                client_num_per_round, replace=False)

    def _aggregate_func(self, cur_clnt, client_num_in_total,
                        client_num_per_round, nei_indexs, w_per_mdls_lstrd):
        """
        Core Decentralized FedAvg aggregation.
        Averages the homogeneous model parameters of all specified neighbors.
        """
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_clnt])
        w = 1.0 / len(nei_indexs)
        for k in w_tmp:
            w_tmp[k] = sum(
                w_per_mdls_lstrd[ci][k] * w for ci in nei_indexs
            )
        return w_tmp

    def record_avg_inference_flops(self, w_per_mdls, mask_pers=None):
        try:
            inference_flops = []
            for client_idx in range(self.args.num_clients):
                w_this_clnt = w_per_mdls[client_idx]
                if mask_pers is None:
                    f_val = self.client_list[client_idx].model_trainer.count_inference_flops(w_this_clnt)
                else:
                    w_per = {
                        name: w_this_clnt[name] * mask_pers[client_idx][name]
                        for name in mask_pers[client_idx]
                    }
                    f_val = self.client_list[client_idx].model_trainer.count_inference_flops(w_per)
                if torch.is_tensor(f_val):
                    f_val = float(f_val.cpu().item())
                else:
                    f_val = float(f_val)
                inference_flops.append(f_val)
            self.stat_info["avg_inference_flops"] = (
                sum(inference_flops) / max(1, len(inference_flops))
            )
        except Exception as e:
            self.logger.warning(f"Could not calculate inference FLOPs: {e}")
            self.stat_info["avg_inference_flops"] = 0.0

    def init_stat_info(self):
        self.stat_info = {
            "person_val_acc_before":  [],
            "person_test_acc_before": [],
            "comm_overhead_params":   0,
            "comm_overhead_mb":       0,
            "convergence_round":      -1,
            "avg_inference_flops":    0,
            "missing_class_acc":      [],
            "present_class_acc":      [],
            "global_class_acc":       [],
            # Cache for global test (reused on off-evaluation rounds)
            "last_missing_acc":       0.0,
            "last_present_acc":       0.0,
            "last_overall_acc":       0.0,
        }
