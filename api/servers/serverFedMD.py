import copy
import gc
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from experiments.client import Client
from experiments.my_model_trainer import MyModelTrainer
from utils.hetero_models import create_hetero_model, assign_architectures
import torch.nn as nn
import torch.nn.functional as F

try:
    import ray
except ImportError:
    ray = None

if ray is not None:
    @ray.remote(num_gpus=1)
    class RayFedMDTrainerActor(object):
        def __init__(self, gpu_id, args, client_archs, client_ids):
            import os
            import sys
            import torch
            import random
            import numpy as np
            import logging
            
            try:
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
                
                from utils.dataprepare_util import load_data
                from experiments.my_model_trainer import MyModelTrainer
                from experiments.client import Client
                from utils.hetero_models import create_hetero_model
                
                self.gpu_id = gpu_id
                self.args = args
                self.client_archs = client_archs
                self.client_ids = client_ids
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(f"RayWorker_GPU_{gpu_id}")
                
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
                    
                print(f"[Ray Worker {gpu_id}] Loading partition data on worker...")
                dataset = load_data(args, args.dataset, logger)
                [train_data_num, test_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_counts] = dataset[:8]
                self.val_data_local_dict = dataset[8] if len(dataset) >= 9 else test_data_local_dict
                self.val_cls_counts_dict = dataset[9] if len(dataset) >= 10 else {}
                 
                self.class_counts = class_counts
                self.num_classes = len(class_counts[0]) if class_counts else 10
                self.clients = {}
                
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

            self.test_global_batches = []
            if getattr(args, 'use_global_test', False) and test_data_global is not None:
                for x, target in test_data_global:
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

        def train_fedmd_clients(self, round_idx, w_per_mdls_lstrd, public_x, all_consensus_logits):
            import copy
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
            
            results = {}
            alignment_dataset = TensorDataset(public_x, all_consensus_logits)
            alignment_loader = DataLoader(alignment_dataset, batch_size=self.args.batch_size, shuffle=True)
            
            for clnt_idx in self.client_ids:
                client = self.clients[clnt_idx]
                w_local = w_per_mdls_lstrd[clnt_idx]
                
                model = client.model_trainer.model
                model.load_state_dict(w_local)
                model.to(self.device)
                model.train()
                
                opt_name = getattr(self.args, 'client_optimizer', 'sgd').lower()
                if opt_name == 'adam':
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=self.args.lr * (self.args.lr_decay ** round_idx),
                        weight_decay=self.args.wd,
                    )
                else:
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=self.args.lr * (self.args.lr_decay ** round_idx),
                        momentum=self.args.momentum,
                        weight_decay=self.args.wd,
                    )
                
                use_amp = self.device.type == 'cuda'
                scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
                
                # 1. Alignment Phase
                for epoch in range(1):
                    for x, consensus_logits in alignment_loader:
                        x, consensus_logits = x.to(self.device), consensus_logits.to(self.device)
                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            student_logits = model(x)
                            loss_kd = F.l1_loss(student_logits, consensus_logits)
                        if not torch.isfinite(loss_kd):
                            continue
                        scaler.scale(loss_kd).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        scaler.step(optimizer)
                        scaler.update()
                
                # 2. Local Training Phase
                criterion = nn.CrossEntropyLoss().to(self.device)
                for epoch in range(self.args.local_epochs):
                    for x, labels in client.local_training_data:
                        x, labels = x.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            outputs = model(x)
                            loss_ce = criterion(outputs, labels.long())
                        if not torch.isfinite(loss_ce):
                            continue
                        scaler.scale(loss_ce).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        scaler.step(optimizer)
                        scaler.update()
                
                w_ret = copy.deepcopy(model.cpu().state_dict())
                val_metrics = client.local_val(w_ret)
                test_metrics = client.local_test(w_ret, b_use_test_dataset=True)
                results[clnt_idx] = {
                    'w_local_mdl': w_ret,
                    'p_val_before_metric': val_metrics,
                    'p_test_before_metric': test_metrics
                }
                
            return results
else:
    RayFedMDTrainerActor = None

class ServerFedMD(object):
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
        method_name = getattr(args, 'name', '') or getattr(args, 'exp_tag', '') or "FedMD"
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

        # Pre-load all batches directly to GPU VRAM if cache_global_test is enabled
        self.test_global_batches = None
        if getattr(args, 'cache_global_test', False) and self.test_global is not None:
            eval_loader = self.test_global
            if hasattr(self.test_global, 'dataset'):
                eval_loader = DataLoader(self.test_global.dataset, batch_size=256, shuffle=False, num_workers=0)
            self.test_global_batches = []
            print(f"🚀 [GlobalTest] Pre-loading all test batches directly to {self.device} memory...")
            for x, target in eval_loader:
                self.test_global_batches.append((x.to(self.device), target.to(self.device)))
            print(f"✅ [GlobalTest] Pre-loading complete. Cached {len(self.test_global_batches)} batches in VRAM.")

        # Architecture assignment
        self.client_archs = assign_architectures(args.num_clients)
        
        # Load Public Dataset for FedMD
        self._setup_public_dataset()

        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)
        self.init_stat_info()
        self._analyze_data_distribution()

    def _setup_public_dataset(self):
        data_dir = getattr(self.args, 'data_dir', './data')
        
        # Determine public dataset name
        pub_dataset_name = getattr(self.args, 'public_dataset', None)
        if pub_dataset_name is None:
            # Fall back to legacy hardcoded logic
            if self.args.dataset in ['bloodmnist', 'retinamnist', 'pathmnist']:
                if self.args.dataset == 'retinamnist':
                    pub_dataset_name = 'bloodmnist'
                elif self.args.dataset == 'bloodmnist':
                    pub_dataset_name = 'pathmnist'
                else: # pathmnist
                    pub_dataset_name = 'bloodmnist'
            elif self.args.dataset == 'cifar10':
                pub_dataset_name = 'cifar100'
            elif self.args.dataset == 'cifar100':
                pub_dataset_name = 'cifar10'
            else:
                pub_dataset_name = 'cifar10'  # default fallback

        # Determine transforms based on self.args.in_channels
        in_channels = getattr(self.args, 'in_channels', 3)
        if in_channels == 3:
            ts = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else: # in_channels == 1
            ts = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[0:1, :, :] if x.shape[0] == 3 else x),
                transforms.Normalize([0.5], [0.5]),
            ])

        # Load public dataset class dynamically
        if pub_dataset_name == 'tissuemnist':
            from api.data_preprocessing.tissuemnist.datasets import TissueMNIST_truncated
            pub_cls = TissueMNIST_truncated
            pub_path = os.path.join(data_dir, 'tissuemnist')
        elif pub_dataset_name == 'bloodmnist':
            from api.data_preprocessing.bloodmnist.datasets import BloodMNIST_truncated
            pub_cls = BloodMNIST_truncated
            pub_path = os.path.join(data_dir, 'bloodmnist')
        elif pub_dataset_name == 'pathmnist':
            from api.data_preprocessing.pathmnist.datasets import PathMNIST_truncated
            pub_cls = PathMNIST_truncated
            pub_path = os.path.join(data_dir, 'pathmnist')
        elif pub_dataset_name == 'retinamnist':
            from api.data_preprocessing.retinamnist.datasets import RetinaMNIST_truncated
            pub_cls = RetinaMNIST_truncated
            pub_path = os.path.join(data_dir, 'retinamnist')
        elif pub_dataset_name == 'dermamnist':
            from api.data_preprocessing.dermamnist.datasets import DermaMNIST_truncated
            pub_cls = DermaMNIST_truncated
            pub_path = os.path.join(data_dir, 'dermamnist')
        elif pub_dataset_name == 'organamnist':
            from api.data_preprocessing.organamnist.datasets import OrganAMNIST_truncated
            pub_cls = OrganAMNIST_truncated
            pub_path = os.path.join(data_dir, 'organamnist')
        elif pub_dataset_name == 'cifar10':
            pub_cls = torchvision.datasets.CIFAR10
            pub_path = os.path.join(data_dir, 'cifar10')
        elif pub_dataset_name == 'cifar100':
            pub_cls = torchvision.datasets.CIFAR100
            pub_path = os.path.join(data_dir, 'cifar100')
        elif pub_dataset_name == 'svhn':
            pub_cls = torchvision.datasets.SVHN
            pub_path = os.path.join(data_dir, 'svhn')
        elif pub_dataset_name == 'mnist':
            pub_cls = torchvision.datasets.MNIST
            pub_path = os.path.join(data_dir, 'mnist')
        elif pub_dataset_name == 'fashionmnist':
            pub_cls = torchvision.datasets.FashionMNIST
            pub_path = os.path.join(data_dir, 'fashionmnist')
        elif pub_dataset_name == 'emnist':
            pub_cls = torchvision.datasets.EMNIST
            pub_path = os.path.join(data_dir, 'emnist')
        else:
            raise ValueError(f"Unsupported public dataset: {pub_dataset_name}")

        raw_pub = pub_cls(root=pub_path, train=True, download=True, transform=ts)
        
        # FedMD uses a subset of public data (capped at dataset size)
        subset_size = min(5000, len(raw_pub))
        indices = np.random.choice(len(raw_pub), subset_size, replace=False)
        
        public_x = []
        for i in indices:
            img, _ = raw_pub[i]
            public_x.append(img)
            
        self.public_x = torch.stack(public_x)
        self.logger.info(f"Loaded {subset_size} public images ({pub_cls.__name__}) for FedMD consensus (Source: {pub_dataset_name}).")

    def _analyze_data_distribution(self):
        self.client_present_classes = {i: set() for i in range(self.args.num_clients)}
        for client_idx in range(self.args.num_clients):
            for x, labels in self.train_data_local_dict[client_idx]:
                for y in labels:
                    self.client_present_classes[client_idx].add(int(y.item()))

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        num_classes = len(self.class_counts[0]) if self.class_counts else 10
        for client_idx in range(self.args.num_clients):
            arch = self.client_archs[client_idx]
            model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
            trainer = MyModelTrainer(model, self.args, self.logger)
            trainer._arch_name = arch
            val_data = self.val_data_local_dict[client_idx] if hasattr(self, 'val_data_local_dict') and client_idx in self.val_data_local_dict else test_data_local_dict[client_idx]
            c = Client(
                client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx], self.args, self.device, trainer, self.logger, arch_name=arch,
                local_val_data=val_data,
            )
            self.client_list.append(c)

    def train(self):
        import time

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_ray = getattr(self.args, 'use_ray', False)

        self.ray_actors = []
        if use_ray and ray is not None:
            print("🔗 Connecting to existing Ray cluster...")
            ray.init(address='auto')
            total_ray_gpus = int(ray.cluster_resources().get('GPU', 0))

            slurm_gpus_env = os.environ.get('SLURM_GPUS')
            if slurm_gpus_env is not None:
                try:
                    target_gpus = int(slurm_gpus_env)
                except ValueError:
                    target_gpus = 1
            else:
                target_gpus = total_ray_gpus

            wait_timeout = 45
            elapsed = 0
            while total_ray_gpus < target_gpus and elapsed < wait_timeout:
                time.sleep(5)
                elapsed += 5
                total_ray_gpus = int(ray.cluster_resources().get('GPU', 0))

            if total_ray_gpus > 0:
                num_workers = min(self.args.num_clients, total_ray_gpus)
                gpu_fraction = 1.0
            else:
                num_workers = self.args.num_clients
                gpu_fraction = 0.0

            print(f"🚀 Initializing {num_workers} persistent Ray workers (1 per dedicated GPU) for FedMD...")
            print(f"⚙️ Ray Actor Resource Allocation: {gpu_fraction:.2f} GPU per worker")

            for worker_id in range(num_workers):
                worker_client_ids = [clnt_idx for clnt_idx in range(self.args.num_clients) if clnt_idx % num_workers == worker_id]
                if not worker_client_ids:
                    continue

                actor = RayFedMDTrainerActor.options(num_gpus=gpu_fraction).remote(
                    worker_id,
                    self.args,
                    self.client_archs,
                    worker_client_ids
                )
                self.ray_actors.append(actor)

            print("⏳ Waiting for Ray workers to load data and partition...")
            ray.get([actor.health_check.remote() for actor in self.ray_actors])
            print("✅ All Ray persistent workers initialized across the cluster.")

        w_per_mdls = [
            self.client_list[i].model_trainer.get_model_params()
            for i in range(self.args.num_clients)
        ]
        num_classes = len(self.class_counts[0]) if self.class_counts else 10
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

            all_consensus_logits = self._compute_public_consensus(w_per_mdls, num_classes)

            p_val_metrics_before = {
                'num_samples':       [],
                'num_correct':       [],
                'losses':            [],
                'per_class_correct': [{} for _ in range(self.args.num_clients)],
                'per_class_total':   [{} for _ in range(self.args.num_clients)],
            }
            p_test_metrics_before = {
                'num_samples':       [],
                'num_correct':       [],
                'losses':            [],
                'per_class_correct': [{} for _ in range(self.args.num_clients)],
                'per_class_total':   [{} for _ in range(self.args.num_clients)],
            }

            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls)

            if getattr(self.args, "parallel", False):
                results = [None] * self.args.num_clients

                if use_ray and self.ray_actors:
                    futures = [
                        actor.train_fedmd_clients.remote(
                            round_idx,
                            w_per_mdls_lstrd,
                            self.public_x,
                            all_consensus_logits
                        )
                        for actor in self.ray_actors
                    ]
                    ray_results = ray.get(futures)
                    for actor_res in ray_results:
                        for clnt_idx, client_res in actor_res.items():
                            results[clnt_idx] = client_res
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    max_workers = self.args.num_clients
                    if getattr(self.args, "multigpu", False) and num_gpus > 0:
                        max_workers = num_gpus

                    alignment_dataset = TensorDataset(self.public_x, all_consensus_logits)
                    alignment_loader = DataLoader(alignment_dataset, batch_size=self.args.batch_size, shuffle=True)

                    print(f"  ⚡ Training {self.args.num_clients} clients in parallel (max workers: {max_workers})...")

                    def _train_single(clnt_idx):
                        client = self.client_list[clnt_idx]
                        torch.set_num_threads(1)
                        w_local_mdl, p_val_metric, p_test_metric = self._train_fedmd_client(
                            client, w_per_mdls_lstrd[clnt_idx], alignment_loader, round_idx
                        )
                        return clnt_idx, w_local_mdl, p_val_metric, p_test_metric

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(_train_single, clnt_idx): clnt_idx
                            for clnt_idx in range(self.args.num_clients)
                        }
                        for future in as_completed(futures):
                            clnt_idx, w_local_mdl, p_val_metric, p_test_metric = future.result()
                            results[clnt_idx] = {
                                'w_local_mdl': w_local_mdl,
                                'p_val_before_metric': p_val_metric,
                                'p_test_before_metric': p_test_metric
                            }

                for clnt_idx, res in enumerate(results):
                    w_local_mdl = res['w_local_mdl']
                    p_val_metric = res.get('p_val_before_metric', res.get('p_test_before_metric'))
                    p_test_metric = res['p_test_before_metric']

                    w_per_mdls[clnt_idx] = w_local_mdl

                    p_val_metrics_before['num_samples'].append(p_val_metric['test_total'])
                    p_val_metrics_before['num_correct'].append(p_val_metric['test_correct'])
                    p_val_metrics_before['losses'].append(p_val_metric['test_loss'])
                    p_val_metrics_before['per_class_correct'][clnt_idx] = p_val_metric['per_class_correct']
                    p_val_metrics_before['per_class_total'][clnt_idx] = p_val_metric['per_class_total']

                    p_test_metrics_before['num_samples'].append(p_test_metric['test_total'])
                    p_test_metrics_before['num_correct'].append(p_test_metric['test_correct'])
                    p_test_metrics_before['losses'].append(p_test_metric['test_loss'])
                    p_test_metrics_before['per_class_correct'][clnt_idx] = p_test_metric['per_class_correct']
                    p_test_metrics_before['per_class_total'][clnt_idx] = p_test_metric['per_class_total']
            else:
                alignment_dataset = TensorDataset(self.public_x, all_consensus_logits)
                alignment_loader = DataLoader(alignment_dataset, batch_size=self.args.batch_size, shuffle=True)
                for clnt_idx in range(self.args.num_clients):
                    print(f'  Round: {round_idx} Client: {clnt_idx}/{self.args.num_clients} [{self.client_archs[clnt_idx]}] Training')
                    client = self.client_list[clnt_idx]
                    w_local_mdl, p_val_metric, p_test_metric = self._train_fedmd_client(
                        client, w_per_mdls[clnt_idx], alignment_loader, round_idx
                    )
                    w_per_mdls[clnt_idx] = w_local_mdl

                    p_val_metrics_before['num_samples'].append(p_val_metric['test_total'])
                    p_val_metrics_before['num_correct'].append(p_val_metric['test_correct'])
                    p_val_metrics_before['losses'].append(p_val_metric['test_loss'])
                    p_val_metrics_before['per_class_correct'][clnt_idx] = p_val_metric['per_class_correct']
                    p_val_metrics_before['per_class_total'][clnt_idx] = p_val_metric['per_class_total']

                    p_test_metrics_before['num_samples'].append(p_test_metric['test_total'])
                    p_test_metrics_before['num_correct'].append(p_test_metric['test_correct'])
                    p_test_metrics_before['losses'].append(p_test_metric['test_loss'])
                    p_test_metrics_before['per_class_correct'][clnt_idx] = p_test_metric['per_class_correct']
                    p_test_metrics_before['per_class_total'][clnt_idx] = p_test_metric['per_class_total']

            param_count = 2 * len(self.public_x) * num_classes * self.args.num_clients
            self.stat_info["comm_overhead_params"] += param_count
            self.stat_info["comm_overhead_mb"]     += (param_count * 4) / (1024**2)
            self.resource_tracker.add_payload_bytes(param_count * 4)

            # Metrics
            p_val_acc_before = sum(
                np.array(p_val_metrics_before['num_correct'][i]) /
                max(1, p_val_metrics_before['num_samples'][i])
                for i in range(self.args.num_clients)
            ) / self.args.num_clients

            p_local_test_acc = sum(
                np.array(p_test_metrics_before['num_correct'][i]) /
                max(1, p_test_metrics_before['num_samples'][i])
                for i in range(self.args.num_clients)
            ) / self.args.num_clients

            # ── Missing vs present class accuracy (KD evidence metric) ────────
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
                global_per_class = self._eval_global_per_class(w_per_mdls)
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
                local_per_class_correct = p_val_metrics_before['per_class_correct']
                local_per_class_total   = p_val_metrics_before['per_class_total']
                missing_acc, present_acc, overall_acc = self._compute_missing_present_acc_local(
                    local_per_class_correct, local_per_class_total
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
                    p_val_metrics_before,
                    round_idx,
                    global_per_class=global_per_class if (self.use_global_test and do_global_eval) else None
                )

            self.stat_info["missing_class_acc"].append(missing_acc)
            self.stat_info["present_class_acc"].append(present_acc)
            self.stat_info["global_class_acc"].append(overall_acc)
            self.stat_info["person_val_acc_before"].append(p_val_acc_before)
            self.stat_info["person_test_acc_before"].append(p_local_test_acc)
            
            p_val_loss = sum(
                np.array(p_val_metrics_before['losses'][i]) /
                max(1, p_val_metrics_before['num_samples'][i])
                for i in range(self.args.num_clients)
            ) / self.args.num_clients

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

            if self.use_global_test:
                global_eval_tag = " [cached]" if not do_global_eval else ""
                print(
                    "Round: {:3d} | Local Acc: {:.4f} | Global Acc (IID){}: {:.4f} | Local Test: {:.4f} "
                    "| Present: {:.2f}% | Missing: {:.2f}% | Loss: {:.4f} | Comm: {:.2f}MB".format(
                        round_idx,
                        p_val_acc_before,
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
                    "| Present: {:.2f}% | Missing: {:.2f}% (local) "
                    "| Loss: {:.4f} | Comm: {:.2f}MB".format(
                        round_idx,
                        p_val_acc_before,
                        p_local_test_acc,
                        present_acc * 100,
                        missing_acc * 100,
                        p_val_loss,
                        self.stat_info["comm_overhead_mb"],
                    )
                )

            if (round_idx + 1) % 50 == 0:
                self.logger.info(
                    f'val_acc_so_far={self.stat_info["person_val_acc_before"]}'
                )
                self._save_checkpoint(w_per_mdls, round_idx)

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

        self.logger.info("FedMD Training Finished.")
        self.logger.info(f'all_val_acc={self.stat_info["person_val_acc_before"]}')
        self.logger.info(f'all_test_acc={self.stat_info["person_test_acc_before"]}')
        
        self.record_avg_inference_flops(w_per_mdls)

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
            'convergence_round': self.stat_info["convergence_round"],
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
        print(f"Convergence Round: {self.stat_info['convergence_round']}")
        print(f"Total Communication: {self.stat_info['comm_overhead_mb']:.2f} MB")
        self.resource_tracker.save_json()
        self.resource_tracker.print_summary_table()
        print("over")

        if hasattr(self, '_eval_models_cache'):
            for m in self._eval_models_cache.values():
                del m
            self._eval_models_cache = {}

        if use_ray and hasattr(self, 'ray_actors') and self.ray_actors:
            for actor in self.ray_actors:
                del actor
            self.ray_actors = []
            import ray
            ray.shutdown()

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
        compare_local = getattr(self.args, 'compare_local', False)
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
            if compare_local and not is_global:
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

    def _compute_public_consensus(self, w_per_mdls, num_classes):
        # Build models for all clients
        client_models = []
        for ci in range(self.args.num_clients):
            arch = self.client_archs[ci]
            model = create_hetero_model(arch, num_classes=num_classes, tiny_experts=self.args.tiny_experts, in_channels=getattr(self.args, 'in_channels', 3))
            model.load_state_dict(w_per_mdls[ci], strict=True)
            model.to(self.device)
            model.eval()
            client_models.append(model)

        public_loader = DataLoader(TensorDataset(self.public_x), batch_size=256, shuffle=False)
        all_consensus_logits = []
        
        use_amp = self.device.type == 'cuda'
        with torch.no_grad():
            for (x,) in public_loader:
                x = x.to(self.device)
                batch_logits = 0
                for model in client_models:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        batch_logits += model(x)
                batch_logits /= len(client_models)
                # Clamp to a safe range so softmax never produces exact zeros/infs.
                # Without this, sharp post-training logits → one-hot softmax → log(0) = -inf → NaN KL loss.
                batch_logits = torch.clamp(batch_logits, min=-20.0, max=20.0)
                all_consensus_logits.append(batch_logits.cpu())
                
        for model in client_models:
            model.cpu()
            del model
        torch.cuda.empty_cache()
        
        return torch.cat(all_consensus_logits)

    def _train_fedmd_client(self, client, w_local, alignment_loader, round_idx):
        # Custom training loop for FedMD
        device = client.device
        model = client.model_trainer.model
        model.load_state_dict(w_local)
        model.to(device)
        model.train()
        
        opt_name = getattr(self.args, 'client_optimizer', 'sgd').lower()
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args.lr * (self.args.lr_decay ** round_idx),
                weight_decay=self.args.wd,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr * (self.args.lr_decay ** round_idx),
                momentum=self.args.momentum,
                weight_decay=self.args.wd,
            )
        
        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # 1. Alignment Phase (L1 loss on raw logits as in the original FedMD paper)
        for epoch in range(1): # usually 1 epoch for alignment
            for x, consensus_logits in alignment_loader:
                x, consensus_logits = x.to(device), consensus_logits.to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    student_logits = model(x)
                    
                    # Original FedMD formulation: L1 loss directly on raw logits (inherently stable)
                    loss_kd = F.l1_loss(student_logits, consensus_logits)
                
                if not torch.isfinite(loss_kd):
                    continue
                
                scaler.scale(loss_kd).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                
        # 2. Local Training Phase (CE on private dataset)
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(self.args.local_epochs):
            for x, labels in client.local_training_data:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(x)
                    loss_ce = criterion(outputs, labels.long())
                
                # Guard: skip backward if loss is NaN/Inf
                if not torch.isfinite(loss_ce):
                    continue
                
                scaler.scale(loss_ce).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                
        w_ret = copy.deepcopy(model.cpu().state_dict())
        
        # Eval on validation and local test sets
        val_metrics = client.local_val(w_ret)
        test_metrics = client.local_test(w_ret, b_use_test_dataset=True)
        return w_ret, val_metrics, test_metrics

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

    def init_stat_info(self):
        self.stat_info = {
            "person_val_acc_before":  [],
            "person_test_acc_before": [],
            "missing_class_acc":      [],
            "present_class_acc":      [],
            "global_class_acc":       [],
            "comm_overhead_params":   0,
            "comm_overhead_mb":       0,
            "convergence_round":      -1,
            "avg_inference_flops":    0,
            "last_missing_acc":       0.0,
            "last_present_acc":       0.0,
            "last_overall_acc":       0.0,
        }

    def _eval_global_per_class(self, w_per_mdls: list) -> dict:
        device = self.device
        num_classes = len(self.class_counts[0]) if self.class_counts else 10
        global_per_class = {}

        use_ray = getattr(self.args, 'use_ray', False)
        if use_ray and hasattr(self, 'ray_actors') and self.ray_actors:
            import ray
            num_workers = len(self.ray_actors)
            worker_client_weights = {wi: {} for wi in range(num_workers)}
            
            for ci in range(self.args.num_clients):
                worker_id = ci % num_workers
                worker_client_weights[worker_id][ci] = w_per_mdls[ci]
                
            futures = []
            active_workers = []
            for wi, client_weights_dict in worker_client_weights.items():
                if client_weights_dict:
                    actor = self.ray_actors[wi]
                    futures.append(actor.evaluate_clients.remote(client_weights_dict))
                    active_workers.append(wi)
                    
            ray_results = ray.get(futures)
            for res in ray_results:
                for ci, client_res in res.items():
                    global_per_class[ci] = client_res
                        
            return global_per_class

        use_cached = getattr(self.args, 'cache_global_test', False) and (self.test_global_batches is not None)
        
        for ci in range(self.args.num_clients):
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
                                per_class_correct[t_val] = per_class_correct.get(t_val, 0) + int((t == p).item())
                                per_class_total[t_val] = per_class_total.get(t_val, 0) + 1
                    else:
                        for x, target in self.test_global:
                            x, target = x.to(device), target.to(device)
                            pred = model(x)
                            _, predicted = torch.max(pred, -1)
                            for t, p in zip(target.view(-1), predicted.view(-1)):
                                t_val = int(t.item())
                                per_class_correct[t_val] = per_class_correct.get(t_val, 0) + int((t == p).item())
                                per_class_total[t_val] = per_class_total.get(t_val, 0) + 1

            if not getattr(self.args, 'cache_eval_models', False):
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
