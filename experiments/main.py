"""
CRAD: Class-wise Reliability-Aware Distillation for Decentralized Heterogeneous Federated Learning.
Entry point for benchmark experiments.

Example run commands:
1. Run CRAD benchmark on CIFAR-10:
   python run_crad_cifar10.py
   # or: python experiments/main.py --algorithm TypicalKD --idea 4.3_pre_v0 --adaptive_pre_filter --val_test_protocol option_a --k_zero_strategy drop --fix_eq6 --dataset cifar10 --num_clients 10 --num_rounds 300 --alpha_kd 0.7 --partition_alpha 0.3 --T 4.0 --cs full --lr 0.1 --seed 1024 --use_global_test --fast_eval --parallel --name "CRAD_cifar10_paper"

2. Run CRAD benchmark on CIFAR-100:
   python run_crad_cifar100.py

3. Run CRAD benchmark on PathMNIST:
   python run_crad_pathmnist.py
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import random
import time

import numpy as np
import torch
import torchvision.datasets.cifar as cifar_module

# Override default Toronto links with fast BrainChip mirror
cifar_module.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
cifar_module.CIFAR100.url = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"

# NumPy compatibility setup for PyTorch weights_only=True load
class scalar(float):
    def __new__(cls, *args, **kwargs):
        if not args: return float.__new__(cls, 0.0)
        val = args[0]
        if isinstance(val, (bytes, bytearray)):
            try:
                return float(np.frombuffer(val, dtype=args[1])[0])
            except Exception:
                return float.__new__(cls, 0.0)
        try:
            return float.__new__(cls, float(val))
        except Exception:
            return float.__new__(cls, 0.0)

safe_classes = [
    (scalar, 'numpy.core.multiarray.scalar'),
    (scalar, 'numpy._core.multiarray.scalar'),
    argparse.Namespace,
    np.dtype,
]

if hasattr(np, 'dtypes'):
    for attr in dir(np.dtypes):
        val = getattr(np.dtypes, attr)
        if isinstance(val, type):
            safe_classes.append(val)

# Also add concrete dtype classes for modern NumPy (like numpy.dtype[float64])
for dt_name in ['float64', 'float32', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']:
    try:
        dt_type = type(np.dtype(dt_name))
        if dt_type not in safe_classes:
            safe_classes.append(dt_type)
    except Exception:
        pass

torch.serialization.add_safe_globals(safe_classes)

from utils.dataprepare_util import load_data
from utils.prefilter_util import compute_competence_mask
from experiments.my_model_trainer import MyModelTrainer
from experiments.dispfd import dispfd
from api.servers.serverFedMD import ServerFedMD
from api.servers.serverDFML import ServerDFML

from utils.hetero_models import create_hetero_model


class Tee:
    """Mirror sys.stdout to a text file with no format changes.

    Every print() call goes to both the console and the file,
    exactly as-is. Flush is called after every write so the file
    is always up-to-date even if the run crashes mid-way.
    """
    def __init__(self, file_path):
        self._console = sys.stdout
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._file = open(file_path, 'w', encoding='utf-8', buffering=1)

    def write(self, data):
        self._console.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._console.flush()
        self._file.flush()

    def isatty(self):
        return False

    def close(self):
        sys.stdout = self._console
        self._file.close()


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    # handler = logging.FileHandler(log_path, mode='w', encoding='UTF-8')
    # handler.setLevel(level=logging.DEBUG)
    # formatter = logging.Formatter('%(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    return logger

def main(args):
    if args.algorithm == 'learn':
        args.num_clients = 1

    # Select architecture pool dynamically based on --model_pool choice
    if args.model_pool == 1:
        args.arch_pool = ["resnet18", "resnet18_half", "cnn6"]
    elif args.model_pool == 2:
        args.arch_pool = ["mobilenetv2", "shufflenetv2", "efficientnet_b0", "densenet_bc"]

    # Normalize dataset name
    if args.dataset == "organmnist":
        args.dataset = "organamnist"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Make log names unique for concurrent execution
    if args.name:
        exp_tag = args.name.replace('/', '_').replace(' ', '_')
    else:
        tag_parts = [args.algorithm]
        if args.algorithm == 'TypicalKD' or args.algorithm == 'FedKD':
            tag_parts.append(args.idea)
            tag_parts.append(f"kd{args.alpha_kd}")
        else:
            tag_parts.append(args.model)
        exp_tag = "_".join(tag_parts)
    
    args.exp_tag = exp_tag
        
    # log_name = f"{exp_tag}_{timestamp}.log"
    # log_path = os.path.join(project_root, 'log', args.dataset, log_name)
    # os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logger_config(log_path=None, logging_name=exp_tag)

    # ── Console Tee: mirror all print() output to a .txt file ────────────────
    txt_name = f"{exp_tag}_{timestamp}_console.txt"
    txt_path = os.path.join(project_root, 'log', args.dataset, txt_name)
    tee = Tee(txt_path)
    sys.stdout = tee
    
    is_colab = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ or os.path.exists('/content')
    if is_colab:
        colab_note = "Running in Google Colab environment. Datasets will be stored on local fast disk (/content/data)."
        logger.info(colab_note)
        print(colab_note)

    # ── Log the exact command used ───────────────────────────────────────────
    full_cmd = "python " + " ".join(sys.argv)
    logger.info(f"EXECUTION COMMAND: {full_cmd}")
    print(f"EXECUTION COMMAND: {full_cmd}")


    # ── Environment info ─────────────────────────────────────────────────────
    import platform
    conda_env  = os.environ.get("CONDA_DEFAULT_ENV", "(not set)")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    venv       = os.environ.get("VIRTUAL_ENV", "")
    print(f"[ENV] Python     : {sys.version}")
    print(f"[ENV] Executable : {sys.executable}")
    print(f"[ENV] Platform   : {platform.platform()}")
    print(f"[ENV] Conda env  : {conda_env}" + (f"  ({conda_prefix})" if conda_prefix else ""))
    if venv:
        print(f"[ENV] Virtual env: {venv}")
    print(f"[ENV] PyTorch    : {torch.__version__}")

    # ── Reproducibility: seed ALL four RNG sources from --seed ────────────────
    # Every run with the same --seed value will have identical:
    #   • data partitioning (numpy)          • model init (torch)
    #   • data augmentation (random/numpy)   • GPU ops (cuda)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # no-op on CPU/MPS, harmless
    torch.backends.cudnn.deterministic = True # ensures CUDA conv is deterministic
    torch.backends.cudnn.benchmark     = False
    logger.info(f"[SEED] All RNGs seeded with seed={seed} "
                f"(random / numpy / torch / cuda). "
                f"Pass --seed <int> to change.")
    print(f"[SEED] seed={seed}")

    # Auto-detect Google Colab to download/store data on local fast ephemeral disk (/content/data)
    is_colab = 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ or os.path.exists('/content')
    if is_colab and args.data_dir == "data":
        args.data_dir = "/content/data"
        logger.info("[COLAB DETECTED] Redirecting default data directory to fast local disk (/content/data)")
        print("[COLAB DETECTED] Redirecting default data directory to fast local disk (/content/data)")
    elif not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_root, args.data_dir)


    # ── Device & Precision Detection ──────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    if getattr(args, 'compile', False) and device.type != 'cuda':
        print(f"⚠️ [WARN] torch.compile() is not fully supported or is unstable on {device.type}. Disabling compilation.")
        args.compile = False
    if device.type == 'cuda':
        try:
            if hasattr(torch, 'get_autocast_dtype'):
                dtype = torch.get_autocast_dtype(device.type)
            else:
                dtype = torch.get_autocast_gpu_dtype()
        except Exception:
            dtype = torch.float16
        precision_str = f"AMP mixed precision ({str(dtype).split('.')[-1]})"
    else:
        # MPS: no fp16 tensor cores, AMP adds overhead — use fp32
        precision_str = str(torch.get_default_dtype()).split('.')[-1]


    # ── Loss weight constraint validation ─────────────────────────────────────
    alpha_ce = max(0.0, 1.0 - args.alpha_kd)
    if args.alpha_kd > 1.0 + 1e-6:
        print(
            f"[WARN] alpha_kd ({args.alpha_kd}) > 1.0. CE weight clamped to 0.0. "
            f"Consider: alpha_kd=0.7 for a CE floor of 0.3."
        )

    print("=" * 80)
    print("Summary of training process:")
    print(f"Algorithm       : {args.algorithm}")
    print(f"Precision       : {precision_str}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Num clients     : {args.num_clients}")
    print(f"Num rounds      : {args.num_rounds}")
    print(f"Local epochs    : {args.local_epochs}")
    print(f"Dataset         : {args.dataset}")

    if args.algorithm in ["TypicalKD", "DFML"]:
        print(f"KD temperature  : {args.T}")
        if args.asymmetric_alpha:
            print(f"alpha_kd        : ASYMMETRIC (architecture-dependent)")
            print(f"  alpha_kd_large  (resnet18)      : {args.alpha_kd_large}")
            print(f"  alpha_kd_mid    (resnet18_half)  : {args.alpha_kd_mid}")
            print(f"  alpha_kd_small  (cnn6)           : {args.alpha_kd_small}")
            print(f"\n🔀 [ASYMMETRIC ALPHA] Architecture-dependent KD weights:")
            print(f"   resnet18      → alpha_kd = {args.alpha_kd_large} (CE = {max(0.0, 1.0 - args.alpha_kd_large):.2f})")
            print(f"   resnet18_half → alpha_kd = {args.alpha_kd_mid} (CE = {max(0.0, 1.0 - args.alpha_kd_mid):.2f})")
            print(f"   cnn6          → alpha_kd = {args.alpha_kd_small} (CE = {max(0.0, 1.0 - args.alpha_kd_small):.2f})\n")
        else:
            print(f"alpha_kd        : {args.alpha_kd}  (neighbor KD loss weight)")
        print(f"alpha_ce        : {alpha_ce:.3f}  (CE loss weight, derived)")
        print(f"KD Idea         : {args.idea}")
        if "4." in args.idea or "pre" in args.idea:
            print(f"n0 (half-sat)   : {args.n0}")
            print(f"gamma (power)   : {args.gamma}")
            print(f"eps_var (inv)   : {args.eps_var}")
    elif args.algorithm == "FedMD":
        print(f"Public Dataset  : {args.public_dataset or 'Same as dataset (default)'}")
        print(f"Client Optimizer: {args.client_optimizer}")

    print(f"Heterogeneous   : True (resnet18 / resnet18_half / cnn6)")
    print(f"Global test     : {'ENABLED' if args.use_global_test else 'DISABLED'}  (missing-class KD metric)")
    print("=" * 80)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset   = load_data(args, args.dataset, logger)
    if getattr(args, 'no_normalization', False):
        from torchvision import transforms
        train_data_local_dict = dataset[5]
        test_data_local_dict = dataset[6]
        test_data_global = dataset[3]
        for key in train_data_local_dict:
            ds = train_data_local_dict[key].dataset
            if hasattr(ds, 'transform') and ds.transform is not None:
                ds.transform.transforms = [t for t in ds.transform.transforms if not isinstance(t, transforms.Normalize)]
        for key in test_data_local_dict:
            ds = test_data_local_dict[key].dataset
            if hasattr(ds, 'transform') and ds.transform is not None:
                ds.transform.transforms = [t for t in ds.transform.transforms if not isinstance(t, transforms.Normalize)]
        if test_data_global is not None:
            ds = test_data_global.dataset
            if hasattr(ds, 'transform') and ds.transform is not None:
                ds.transform.transforms = [t for t in ds.transform.transforms if not isinstance(t, transforms.Normalize)]
        print("⚠️ [WARN] Input normalization transforms stripped from all datasets to match unnormalized baseline.")

    num_class = len(dataset[7][0])   # class_counts[0] length
    
    # Detect in_channels (1 for grayscale MedMNIST/MNIST, 3 for RGB)
    if args.dataset in ["organamnist", "tissuemnist", "mnist", "fashionmnist", "emnist"]:
        args.in_channels = 1
    else:
        args.in_channels = 3

    # Print data distribution table
    if args.dataset == 'pathmnist':
        class_names = ['adi', 'bg', 'deb', 'lym', 'muc', 'mus', 'norm', 'str', 'ca']
    elif args.dataset == 'bloodmnist':
        class_names = ['baso', 'eosi', 'eryt', 'ig', 'lymp', 'mono', 'neut', 'plat']
    elif args.dataset == 'retinamnist':
        class_names = ['norm', 'mild', 'mod', 'sev', 'prol']
    elif args.dataset == 'dermamnist':
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    elif args.dataset == 'organamnist':
        class_names = ['blad', 'f-l', 'f-r', 'heart', 'k-l', 'k-r', 'liver', 'lu-l', 'lu-r', 'pan', 'spl']
    elif args.dataset == 'tissuemnist':
        class_names = [f"Cls {c}" for c in range(8)]
    elif args.dataset == 'cifar10':
        class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        class_names = [f"Cls {c}" for c in range(num_class)]

    print("\n" + "=" * 125)
    # If pre-filtering is selected (idea == '4.3_pre'), we add a note to the header and compute the competent mask + thresholds
    pre_filtering_active = (args.idea in ('4.3_pre', '4.3_pre_v0'))
    competent_mask = None
    thresholds = None
    thresh_label = "Thresh"
    if pre_filtering_active:
        print("DATA DISTRIBUTION (Training samples per class per client; * indicates class is filtered/removed from teaching):")
        counts_tensor = torch.tensor(dataset[7], dtype=torch.float32)
        val_counts_tensor = None
        if getattr(args, 'split_val_test', False) and len(dataset) >= 10 and dataset[9]:
            val_cls_counts_dict = dataset[9]
            has_val_samples = any(len(c_dict) > 0 for c_dict in val_cls_counts_dict.values())
            if has_val_samples:
                num_clients_local = len(dataset[7])
                num_classes_local = len(dataset[7][0])
                val_counts_tensor = torch.zeros((num_clients_local, num_classes_local), dtype=torch.float32)
                for ci, c_dict in val_cls_counts_dict.items():
                    for c_idx, c_cnt in c_dict.items():
                        if c_idx < num_classes_local:
                            val_counts_tensor[ci, c_idx] = float(c_cnt)
        competent_mask, thresholds, thresh_label = compute_competence_mask(counts_tensor, args, val_counts_tensor=val_counts_tensor)
    else:
        print("DATA DISTRIBUTION (Training samples per class per client):")

    header = f"{'Client':<6} | " + " | ".join(f"{name:<8}" for name in class_names) + " | Total"
    print(header)
    print("-" * len(header))

    # Print the threshold row if pre-filtering is active
    if pre_filtering_active and thresholds is not None:
        thresh_items = []
        for val in thresholds:
            if isinstance(val, float) and val.is_integer():
                val_str = str(int(val))
            else:
                val_str = f"{val:.1f}"
            thresh_items.append(f"{val_str:>8}")
        row = f"{thresh_label:<6} | " + " | ".join(thresh_items) + " | "
        print(row)
        print("-" * len(header))

    class_counts = dataset[7]
    for ci, counts in enumerate(class_counts):
        row_items = []
        for c_idx, count in enumerate(counts):
            if pre_filtering_active and competent_mask is not None and not competent_mask[ci, c_idx]:
                val_str = f"{count}*"
            else:
                val_str = str(count)
            row_items.append(f"{val_str:>8}")
        row = f"{ci:<6} | " + " | ".join(row_items) + f" | {sum(counts):>6}"
        print(row)
    print("=" * 125 + "\n")

    # Print validation data distribution table if validation partition exists
    val_cls_counts_dict = dataset[9] if len(dataset) >= 10 else {}
    has_val_samples = any(len(c_dict) > 0 for c_dict in val_cls_counts_dict.values()) if val_cls_counts_dict else False
    if getattr(args, 'split_val_test', False) and has_val_samples:
        print("=" * 125)
        print(f"VALIDATION DATA DISTRIBUTION (Validation samples per class per client; Protocol: {getattr(args, 'val_test_protocol', 'option_a').upper()}):")
        print(header)
        print("-" * len(header))
        for ci in range(len(class_counts)):
            row_items = []
            tot_val = 0
            for c_idx in range(num_class):
                cnt = val_cls_counts_dict.get(ci, {}).get(c_idx, 0)
                tot_val += cnt
                row_items.append(f"{cnt:>8}")
            row = f"{ci:<6} | " + " | ".join(row_items) + f" | {tot_val:>6}"
            print(row)
        print("=" * 125 + "\n")

    # Print local test data distribution table
    test_data_local_dict = dataset[6] if len(dataset) >= 7 else {}
    if test_data_local_dict:
        protocol_str = f"Protocol: {getattr(args, 'val_test_protocol', 'option_a').upper()}" if getattr(args, 'split_val_test', False) else "Legacy Mode"
        print("=" * 125)
        print(f"LOCAL TEST DATA DISTRIBUTION (Local test samples per class per client; {protocol_str}):")
        print(header)
        print("-" * len(header))
        for ci in range(len(class_counts)):
            row_items = []
            tot_test = 0
            # Extract per-class count from local test loader dataset
            test_counts_c = {c: 0 for c in range(num_class)}
            if ci in test_data_local_dict:
                ds_obj = test_data_local_dict[ci].dataset
                t_arr = None
                if hasattr(ds_obj, 'indices') and hasattr(ds_obj, 'dataset'):
                    parent_ds = ds_obj.dataset
                    idxs = ds_obj.indices
                    if hasattr(parent_ds, 'targets') and parent_ds.targets is not None:
                        t_arr = np.array(parent_ds.targets)[idxs]
                    elif hasattr(parent_ds, 'target') and parent_ds.target is not None:
                        t_arr = np.array(parent_ds.target)[idxs]
                    elif hasattr(parent_ds, 'labels') and parent_ds.labels is not None:
                        t_arr = np.array(parent_ds.labels)[idxs]
                    else:
                        t_arr = np.array([parent_ds[i][1] for i in idxs])
                elif hasattr(ds_obj, 'targets') and ds_obj.targets is not None:
                    t_arr = np.array(ds_obj.targets)
                elif hasattr(ds_obj, 'target') and ds_obj.target is not None:
                    t_arr = np.array(ds_obj.target)
                elif hasattr(ds_obj, 'labels') and ds_obj.labels is not None:
                    t_arr = np.array(ds_obj.labels)

                if t_arr is not None:
                    if t_arr.ndim > 1:
                        t_arr = t_arr.ravel()
                    unq, u_cnts = np.unique(t_arr, return_counts=True)
                    for u, cnt_u in zip(unq, u_cnts):
                        if int(u) < num_class:
                            test_counts_c[int(u)] = int(cnt_u)
            for c_idx in range(num_class):
                cnt = test_counts_c.get(c_idx, 0)
                tot_test += cnt
                row_items.append(f"{cnt:>8}")
            row = f"{ci:<6} | " + " | ".join(row_items) + f" | {tot_test:>6}"
            print(row)
        print("=" * 125 + "\n")



    # ── Reference model (for FLOPs accounting in dispfd) ─────────────────────
    # dispfd creates per-client trainers internally; this is only the baseline
    # reference used by record_avg_inference_flops.
    ref_model   = create_hetero_model('resnet18', num_classes=num_class, tiny_experts=args.tiny_experts, in_channels=args.in_channels)
    ref_trainer = MyModelTrainer(ref_model, args, logger)

    print(f"torch version {torch.__version__}")

    # ── Device ────────────────────────────────────────────────────────────────
    if device.type == 'cuda':
        print(f"[GPU INFO] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[GPU INFO] GPU count = {torch.cuda.device_count()}")

    if args.algorithm == "FedMD":
        ServerFedMD(dataset, device, args, ref_trainer, logger).train()
    elif args.algorithm == "DFML":
        ServerDFML(dataset, device, args, ref_trainer, logger).train()
    else:
        dispfd(dataset, device, args, ref_trainer, logger).train()

    # ── Finalize Console Log ──────────────────────────────────────────────────
    print(f"\n[FINISH] Experiment completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── Dataset / model ───────────────────────────────────────────────────────
    parser.add_argument("--dataset",      type=str,   default="cifar10")
    parser.add_argument("--public_dataset", type=str, default=None,
                        help="Optional override for the public dataset in FedMD (e.g. tissuemnist, bloodmnist, etc.)")
    parser.add_argument("--model",        type=str,   default="resnet18",
                        choices=["resnet18", "resnet18_half", "cnn6", "pyramidnet", "mobilenetv2", "shufflenetv2", "efficientnet_b0", "densenet_bc"],
                        help="Base architecture (overridden by hetero pool in KD mode)")
    parser.add_argument("--arch_pool",    nargs='+',  default=["resnet18", "resnet18_half", "cnn6"],
                        help="List of architectures to assign round-robin to clients")
    parser.add_argument("--model_pool",   type=int,   default=1, choices=[1, 2],
                        help="Select client architecture pool: 1 = [resnet18, resnet18_half, cnn6] (original), 2 = [mobilenetv2, shufflenetv2, efficientnet_b0, densenet_bc] (SOTA)")
    parser.add_argument("--algorithm",    type=str,   default="TypicalKD",
                        choices=["TypicalKD", "CRAD", "FedAvg", "FedMD", "DFML", "Standalone"],
                        help="Algorithm: TypicalKD / CRAD (decentralized KD) | FedAvg | FedMD | DFML | Standalone")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=0.1)
    parser.add_argument("--lr_decay",     type=float, default=0.99)
    parser.add_argument("--wd",           type=float, default=5e-4)
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--nesterov",     action="store_true", default=False,
                        help="Enable Nesterov accelerated gradient momentum (default: False).")
    parser.add_argument("--num_rounds",   type=int,   default=300)
    parser.add_argument("--local_epochs", type=int,   default=1)
    parser.add_argument("--num_clients",  type=int,   default=10)
    parser.add_argument("--frac",         type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=1024)

    # ── Reviewer Revisions (Tasks 1, 2, 3, and 4) ──────────────────────────────
    parser.add_argument("--val_test_protocol", type=str, default="none", choices=["none", "option_a", "option_b"],
                        help="Dataset partition protocol: 'none' (default / original paper legacy mode), 'option_a' (50/50 shard split), or 'option_b' (resampled from benchmark test set). Setting option_a or option_b automatically activates validation splitting.")
    parser.add_argument("--split_val_test", action="store_true", default=False,
                        help="Split original local test set into dedicated local val set and untouched test set (auto-enabled when --val_test_protocol is set).")
    parser.add_argument("--no_split_val_test", action="store_false", dest="split_val_test",
                        help="Disable local test dataset splitting (legacy mode).")
    parser.add_argument("--val_test_ratio", type=float, default=0.5,
                        help="Ratio of local test data allocated to local validation set (default: 0.5).")
    parser.add_argument("--fix_eq6", action="store_true", default=False,
                        help="Use local validation set sample counts for Equation 6 m_j^c support counts (auto-enabled for revision protocols).")
    parser.add_argument("--no_fix_eq6", action="store_false", dest="fix_eq6",
                        help="Use legacy training sample counts for Equation 6 m_j^c.")
    parser.add_argument("--k_zero_strategy", type=str, default="prior", choices=["drop", "prior"],
                        help="Strategy when sample count k=0 for a class: 'drop' (weight=0.0) or 'prior' (Agresti-Coull prior, default: prior).")
    parser.add_argument("--privacy_noise", type=str, default="none", choices=["none", "laplace", "bucket"],
                        help="Privacy noise applied to shared validation counts m_j^c: 'none', 'laplace', or 'bucket'.")
    parser.add_argument("--privacy_epsilon", type=float, default=1.0,
                        help="Epsilon parameter for Laplace privacy noise or count bucketing (default: 1.0).")
    parser.add_argument("--track_costs", action="store_true", default=True,
                        help="Track network payload, wall-clock time, and peak memory usage per round (default: True).")
    parser.add_argument("--no_track_costs", action="store_false", dest="track_costs",
                        help="Disable resource usage and cost profiling.")


    # ── KD hyperparameters ────────────────────────────────────────────────────
    parser.add_argument("--T",            type=float, default=4.0,
                        help="Distillation temperature for TypicalKD.")
    parser.add_argument("--alpha_kd",     type=float, default=0.7,
                        help="Weight assigned to the neighbor Knowledge Distillation loss. "
                             "Remaining weight (1 - alpha_kd) is assigned to local Cross-Entropy.")
    parser.add_argument("--idea",          type=str,   default="4.3_pre_v0",
                        choices=["4.3_pre_v0", "crad", "4.3_pre", "baseline", "idea3"],
                        help="KD aggregation method: '4.3_pre_v0' / 'crad' (CRAD: Class-wise Reliability-Aware Distillation), "
                             "'baseline' (Uniform KD), 'idea3' (Uncertainty KD)")
    parser.add_argument("--n0",            type=float, default=10.0,
                        help="Half-saturation point for reliability estimation.")
    parser.add_argument("--gamma",         type=float, default=0.5,
                        help="Exponent for power-law weighting.")
    parser.add_argument("--eps_var",       type=float, default=1e-6,
                        help="Variance smoothing parameter for inverse-variance weighting.")
    parser.add_argument("--pre_filter_threshold", type=int, default=10,
                        help="Minimum per-class sample count for a teacher to be considered competent (default: 10).")
    parser.add_argument("--adaptive_pre_filter", action="store_true", default=False,
                        help="Adaptive pre-filtering: keep teachers whose sample count is at least the per-class median support.")
    parser.add_argument("--pre_filter_soft_decay", type=float, default=0.0,
                        help="Soft/decaying median scale: reduces median threshold by a decay factor.")
    parser.add_argument("--pre_filter_percentage", type=float, default=None,
                        help="Keep top P%% of teachers with highest support per class.")
    parser.add_argument("--pre_filter_max_fraction", type=float, default=None,
                        help="Relative-to-max floor: Keep teachers whose sample count is at least a fraction of the maximum count.")
    parser.add_argument('--use_safe_harbor', action='store_true', default=False,
                        help='Activate Median-Anchored Safe Harbor competence gating.')
    parser.add_argument('--safe_harbor_gamma', type=float, default=0.3,
                        help='Multiplier for non-zero class median (default: 0.3).')
    parser.add_argument('--safe_harbor_cap', type=float, default=50.0,
                        help='Ceiling cap for filtering threshold (default: 50.0).')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon for local CrossEntropyLoss.')
    parser.add_argument('--oe_lambda', type=float, default=0.0,
                        help='Weight for Out-of-Distribution Entropy maximization in Self-OE.')


    # ── DFML hyperparameters ──────────────────────────────────────────────────
    parser.add_argument("--alpha_min",    type=float, default=0.0)
    parser.add_argument("--alpha_max",    type=float, default=1.0)
    parser.add_argument("--dfml_K",       type=int,   default=1,
                        help="Number of mutual learning epochs for DFML aggregator.")

    # ── Partitioning ──────────────────────────────────────────────────────────
    parser.add_argument("--partition_method", type=str,   default="dir")
    parser.add_argument("--partition_alpha",  type=float, default=0.3)

    # ── Topology ──────────────────────────────────────────────────────────────
    parser.add_argument("--cs",           type=str,   default="full",
                        help="Neighbor selection: random | ring | grid | exp | full")

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--use_global_test", action="store_true", default=True,
                        help="Evaluates metrics on the centralized IID test set (Oracle evaluation; default: True).")
    parser.add_argument("--no_global_test", action="store_false", dest="use_global_test",
                        help="Disable centralized IID global test evaluation.")
    parser.add_argument("--compare_local", action="store_true", default=False,
                        help="Trains a 'Local Only' (no KD, no federation) model for each client "
                             "alongside the federated model to measure the KD gain/loss.")
    parser.add_argument("--fast_eval", action="store_true", default=False,
                        help="Enable vectorized fast evaluation (highly recommended on Blackwell GPUs).")
    parser.add_argument("--cache_global_test", action="store_true", default=False,
                        help="Cache the entire global test dataset in VRAM during initialization "
                             "to eliminate CPU dataloader bottlenecks during evaluation rounds.")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Compile models using torch.compile() (PyTorch 2.0+ speed optimization).")
    parser.add_argument("--cache_teachers", action="store_true", default=False,
                        help="Cache neighbor teacher model objects in memory to avoid instantiation overhead every round.")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Train all clients in parallel within each round using ThreadPoolExecutor. "
                             "Provides ~Nx speedup on GPU where N=num_clients. Safe because client "
                             "training is independent within a round.")
    parser.add_argument("--multigpu", action="store_true", default=False,
                        help="Distribute client models round-robin across all available GPUs "
                             "when training in parallel (only active when --parallel is enabled).")
    parser.add_argument("--skip_eval_until_round", type=int, default=0,
                        help="Skip local and global test evaluations until this round index to speed up training.")
    parser.add_argument("--dynamic_loss_scale", action="store_true", default=False,
                        help="Dynamically scale CE loss back to 1.0 (instead of alpha_ce) when there are no active teachers/neighbors in TypicalKD.")
    parser.add_argument("--persistent_optimizer", action="store_true", default=False,
                        help="Persist client optimizers across rounds to preserve SGD momentum buffers.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--num_workers",  type=int,   default=0,
                        help="Number of workers for data loader (speed optimization).")
    parser.add_argument("--no_normalization", action="store_true", default=False,
                        help="Disable input normalization (match unnormalized baseline learn method).")
    parser.add_argument("--device",       type=str,   default="cuda",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--data_dir",     type=str,   default="data")
    parser.add_argument("--result_path",  type=str,   default="results")
    parser.add_argument("--train",        type=int,   default=1, choices=[0, 1])
    parser.add_argument("--times",        type=int,   default=1)

    # ── Legacy args (kept for compatibility) ──────────────────────────────────
    parser.add_argument("--gen_batch_size",         type=int,   default=32)
    parser.add_argument("--client_optimizer",       type=str,   default="sgd")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01)
    parser.add_argument("--ensemble_lr",            type=float, default=1e-4)
    parser.add_argument("--beta",                   type=float, default=1.0)
    parser.add_argument("--lamda",                  type=int,   default=1)
    parser.add_argument("--mix_lambda",             type=float, default=0.1)
    parser.add_argument("--embedding",              type=int,   default=0)
    parser.add_argument("--K",                      type=int,   default=1)
    parser.add_argument("--anneal_factor",          type=float, default=1.0)
    parser.add_argument("--comm_round",             type=int,   default=300)
    parser.add_argument('--tiny_experts', type=str, default='none', choices=['none', 'broken', 'ova', 'router'],
                        help='Test Extreme Specialist architectures. Replaces cnn6 with a 10-expert ensemble.')
    parser.add_argument("--name",         type=str,   default="",
                        help="Experiment name to append to the log/checkpoint files")
    parser.add_argument("--load_checkpoint", type=str, default="",
                        help="Path to a checkpoint .pt file to load initial client weights")
    parser.add_argument("--distill_target_class", type=int, default=-1,
                        help="Single class index to distill to. If -1, automatically use the class with the lowest sample count.")
    parser.add_argument("--class_control_client_idx", type=int, default=11,
                        help="Client index (0-indexed) that will apply class-controlled distillation (default: 11).")
    parser.add_argument("--freeze_teachers", action="store_true", default=False,
                        help="If specified, keep weights of clients 0-9 frozen and skip their training.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="If specified, resume the training loop, statistics, and best accuracies from the checkpoint.")
    parser.add_argument("--reset_active_client", action="store_true", default=False,
                        help="If specified, reset the weights of the last client (client 10) to random initialization even when loading a checkpoint.")
    parser.add_argument("--cache_eval_models", action="store_true", default=False,
                        help="If specified, cache global evaluation models directly in VRAM to speed up global tests.")
    parser.add_argument("--use_shared_memory", action="store_true", default=False,
                        help="If specified, place multiprocessing queue tensors in shared memory to speed up IPC.")
    parser.add_argument("--curriculum_start_round", type=int, default=100,
                        help="The round index at which curriculum targeting starts.")
    parser.add_argument("--curriculum_target_class_mode", type=str, default="none",
                        choices=["none", "worst_only", "soft_mask", "data_aware"],
                        help="Curriculum mode to use: none | worst_only | soft_mask | data_aware")
    parser.add_argument("--curriculum_target_threshold", type=float, default=0.95,
                        help="Validation accuracy threshold to promote a class or mask it (default: 0.95).")
    parser.add_argument("--curriculum_keep_alive_weight", type=float, default=0.2,
                        help="KD loss weight multiplier for non-targeted/well-learned classes (default: 0.2).")
    parser.add_argument("--curriculum_target_weight", type=float, default=2.0,
                        help="KD loss weight multiplier for targeted classes (default: 2.0).")
    parser.add_argument("--curriculum_min_local_samples", type=int, default=500,
                        help="Minimum local training samples a client must have for a class "
                             "before that class's KD weight can be reduced in data_aware mode. "
                             "Classes below this count are considered KD-dependent. (default: 500)")
    parser.add_argument("--boost_class_idx", type=int, default=-1,
                        help="Index of class to boost distillation weight for all clients (default: -1, no boost).")
    parser.add_argument("--boost_class_weight", type=float, default=1.0,
                        help="Weight multiplier for the boosted class (default: 1.0).")
    parser.add_argument("--use_dynamic_scaling", action="store_true", default=False,
                        help="Enable data-aware class weight scaling with dynamic decay.")
    parser.add_argument("--scaling_max_boost", type=float, default=0.02,
                        help="Maximum weight boost applied to classes with minimum sample counts (default: 0.02).")
    parser.add_argument("--scaling_max_reduction", type=float, default=0.005,
                        help="Maximum weight reduction applied to classes with maximum sample counts (default: 0.005).")
    parser.add_argument("--scaling_decay_end_round", type=int, default=200,
                        help="The round index at which the weight adjustments decay completely to 0 (default: 200).")
    parser.add_argument("--use_drag_scaling", action="store_true", default=False,
                        help="Enable dynamic client-specific class weight scaling based on accuracy drag thresholds.")
    parser.add_argument("--drag_boost_multiplier", type=float, default=1.0,
                        help="Multiplier for the accuracy drag threshold boosts (default: 1.0).")
    parser.add_argument("--asymmetric_alpha", action="store_true", default=False,
                        help="Enable architecture-dependent KD alpha. Large models get lower alpha_kd "
                             "(more self-reliance), small models get higher alpha_kd (more peer learning).")
    parser.add_argument("--alpha_kd_large", type=float, default=0.3,
                        help="KD weight for large models (resnet18) when --asymmetric_alpha is on (default: 0.3).")
    parser.add_argument("--alpha_kd_mid", type=float, default=0.5,
                        help="KD weight for mid models (resnet18_half) when --asymmetric_alpha is on (default: 0.5).")
    parser.add_argument("--alpha_kd_small", type=float, default=0.9,
                        help="KD weight for small models (cnn6) when --asymmetric_alpha is on (default: 0.9).")
    parser.add_argument("--use_hardcoded_reduction", action="store_true", default=False,
                        help="Enable hardcoded weight reduction for stable client/class pairs.")
    parser.add_argument("--hardcoded_reduction_tiers", type=int, default=1, choices=[1, 2],
                        help="Number of tiers for hardcoded reduction: 1 (<0.5%%), 2 (<0.5%% and 0.5%%-1.0%%)")
    parser.add_argument("--hardcoded_reduction_multiplier", type=float, default=1.0,
                        help="Multiplier for the hardcoded reductions (default: 1.0)")
    parser.add_argument("--use_class_sample_scaling", action="store_true", default=False,
                        help="Enable inverse class-sample weight scaling with linear ramp up.")
    parser.add_argument("--class_scaling_ramp_rounds", type=int, default=100,
                        help="Number of rounds over which class-sample weights ramp up (default: 100).")
    parser.add_argument("--class_scaling_low_count", type=int, default=100,
                        help="Sample count threshold below which a class is considered low-sample (default: 100).")
    parser.add_argument("--use_ray", action="store_true", default=False,
                        help="Enable multi-node parallel client training using Ray.")


    args = parser.parse_args()

    # Auto-enable validation splitting if validation protocol is specified
    if args.val_test_protocol in ["option_a", "option_b"]:
        args.split_val_test = True
        args.fix_eq6 = True

    main(args)
