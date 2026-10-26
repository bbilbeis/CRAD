import torch
import numpy as np

def compute_competence_mask(counts_tensor, args, val_counts_tensor=None):
    """
    Computes a competence mask and thresholds based on pre-filtering settings in args.
    Checks both training sample counts AND validation sample counts when val_counts_tensor is provided.

    Args:
        counts_tensor (torch.Tensor): [N, C] tensor of training sample counts per class per teacher.
        args: Argument namespace with pre-filtering settings.
        val_counts_tensor (torch.Tensor, optional): [N, C] tensor of validation sample counts per class per teacher.

    Returns:
        competent_mask (np.ndarray): [N, C] boolean mask.
        thresholds (list): [C] list of thresholds used.
        thresh_label (str): Label for the threshold type.
    """
    num_class = counts_tensor.shape[1]
    counts_float = torch.nan_to_num(counts_tensor.float(), nan=0.0)
    
    # 1. Safe Harbor Pre-Filtering (takes precedence)
    if getattr(args, 'use_safe_harbor', False):
        safe_harbor_gamma = getattr(args, 'safe_harbor_gamma', 0.3)
        safe_harbor_cap = getattr(args, 'safe_harbor_cap', 50.0)
        
        # Remove zeroes to compute median only over active classes
        counts_nan = counts_float.clone()
        counts_nan[counts_float == 0] = float('nan')
        
        # Compute class-wise median support ignoring NaNs
        try:
            class_median_support = torch.nanmedian(counts_nan, dim=0).values
        except AttributeError:
            medians = []
            for c in range(num_class):
                valid = counts_float[:, c][counts_float[:, c] > 0]
                if len(valid) > 0:
                    medians.append(valid.median())
                else:
                    medians.append(torch.tensor(0.0))
            class_median_support = torch.stack(medians)
            
        class_median_support = torch.nan_to_num(class_median_support, nan=0.0)
        
        calculated_thresholds = torch.clamp(safe_harbor_gamma * class_median_support, max=safe_harbor_cap)
        competent_mask = (counts_float >= calculated_thresholds.unsqueeze(0)).bool().cpu().numpy()
        thresholds = calculated_thresholds.cpu().tolist()
        thresh_label = f"SH({safe_harbor_gamma})"

    # 2. Max-Fraction Pre-Filtering (Option 2: Relative-to-max floor)
    elif getattr(args, 'pre_filter_max_fraction', None) is not None and getattr(args, 'pre_filter_max_fraction', 0.0) > 0.0:
        max_fraction = getattr(args, 'pre_filter_max_fraction')
        counts_float = counts_tensor.float()
        max_counts = counts_float.max(dim=0).values  # [C]
        calculated_thresholds = max_fraction * max_counts
        
        competent_mask = (counts_float >= calculated_thresholds.unsqueeze(0)).bool().cpu().numpy()
        thresholds = calculated_thresholds.cpu().tolist()
        thresh_label = f"Max({max_fraction})"

    # 3. Top P% Pre-Filtering
    elif getattr(args, 'pre_filter_percentage', None) is not None and getattr(args, 'pre_filter_percentage', 0.0) > 0.0:
        pre_filter_percentage = getattr(args, 'pre_filter_percentage')
        counts_float = counts_tensor.float()
        q = torch.quantile(counts_float, 1.0 - pre_filter_percentage, dim=0)
        competent_mask = (counts_float >= q.unsqueeze(0)).bool().cpu().numpy()
        thresholds = q.cpu().tolist()
        thresh_label = f"{int(pre_filter_percentage * 100)}%"

    # 4. Adaptive (Median) Pre-Filtering
    elif getattr(args, 'adaptive_pre_filter', False):
        counts_float = counts_tensor.float()
        class_median_support = counts_float.median(dim=0).values
        soft_decay = getattr(args, 'pre_filter_soft_decay', 0.0)
        threshold = getattr(args, 'pre_filter_threshold', 10)
        
        if soft_decay > 0.0:
            decayed_threshold = class_median_support * (1.0 - soft_decay)
            threshold_floor = torch.clamp(class_median_support, max=float(threshold))
            effective_threshold = torch.max(decayed_threshold, threshold_floor)
            competent_mask = (counts_float >= effective_threshold.unsqueeze(0)).bool().cpu().numpy()
            thresholds = effective_threshold.cpu().tolist()
            thresh_label = f"Adp({int(soft_decay*100)}%)"
        else:
            competent_mask = (counts_float >= class_median_support.unsqueeze(0)).bool().cpu().numpy()
            thresholds = class_median_support.cpu().tolist()
            thresh_label = "Median"

    # 5. Fixed Threshold Pre-Filtering (default / Option 1)
    else:
        threshold = getattr(args, 'pre_filter_threshold', 10)
        competent_mask = (counts_tensor >= threshold).bool().cpu().numpy()
        thresholds = [float(threshold)] * num_class
        thresh_label = "Thresh"

    # Dual-Count Check: Also require validation count to be valid/non-zero
    if val_counts_tensor is not None:
        val_counts_float = val_counts_tensor.float()
        # Drop teacher j for class c if validation count is zero or below threshold
        val_valid = (val_counts_float > 0).cpu().numpy()
        competent_mask = competent_mask & val_valid

    return competent_mask, thresholds, thresh_label


def compute_dual_count_competence_mask(train_counts_tensor, val_counts_tensor, args=None, threshold=1):
    """
    Stage-1 Dual-Count Competence Mask.
    Drops teacher j for class c when:
    - Training support is zero/near-zero: n_{train,j}^c < threshold (teacher does not know the class)
    - OR Validation support is zero/near-zero: k_{val,j}^c < threshold (cannot verify that it does)

    Args:
        train_counts_tensor (torch.Tensor): [N, C] tensor of training sample counts per class per teacher.
        val_counts_tensor (torch.Tensor): [N, C] tensor of validation sample counts per class per teacher.
        args (optional): Argument namespace with pre-filtering settings.
        threshold (int or float): Minimum sample count threshold for both training and validation (default: 1).

    Returns:
        competent_mask (np.ndarray): [N, C] boolean mask.
        thresholds (list): [C] list of thresholds used.
        thresh_label (str): Label for the threshold type.
    """
    if args is not None:
        thresh = getattr(args, 'pre_filter_threshold', threshold)
    else:
        thresh = threshold

    num_class = train_counts_tensor.shape[1]
    train_float = torch.nan_to_num(train_counts_tensor.float(), nan=0.0)
    val_float = torch.nan_to_num(val_counts_tensor.float(), nan=0.0)

    train_competent = (train_float >= thresh)
    val_competent = (val_float >= thresh)

    dual_competent = (train_competent & val_competent).bool().cpu().numpy()
    thresholds = [float(thresh)] * num_class
    thresh_label = f"DualThresh({thresh})"
    return dual_competent, thresholds, thresh_label
