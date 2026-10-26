"""
Core Training Engine & Knowledge Distillation Logic for CRAD.

This module implements the client-side training loops and knowledge distillation
mechanisms for Class-wise Reliability-Aware Distillation (CRAD) in decentralized
heterogeneous federated learning.
"""
import copy
import logging
import os

import numpy as np
import torch
import torch._dynamo
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from experiments.model_trainer import ModelTrainer
from utils.hetero_models import create_hetero_model
from utils.prefilter_util import compute_competence_mask


# ───────────────────────────────────────────────────────────────────────────────
# Indexed Dataset Wrapper for caching teacher logits
# ───────────────────────────────────────────────────────────────────────────────

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return index, data, target

    def __len__(self):
        return len(self.dataset)


# ───────────────────────────────────────────────────────────────────────────────
# Distillation Loss Functions
# ───────────────────────────────────────────────────────────────────────────────

def compute_kl_div(log_p_student, q_bar, T, target_class=-1, class_weights=None):
    """Computes scaled Kullback-Leibler divergence between student and target distribution."""
    if class_weights is not None:
        kl_none = F.kl_div(log_p_student, q_bar, reduction="none")  # [B, C]
        weighted_kl = kl_none * class_weights.to(log_p_student.device).view(1, -1)
        return weighted_kl.sum() / log_p_student.size(0) * (T ** 2)
    elif target_class >= 0:
        kl_none = F.kl_div(log_p_student, q_bar, reduction="none")  # [B, C]
        return kl_none[:, target_class].sum() / log_p_student.size(0) * (T ** 2)
    else:
        return F.kl_div(log_p_student, q_bar, reduction="batchmean") * (T ** 2)


def standard_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits_list: list,
    T: float,
    target_class: int = -1,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Uniform Knowledge Distillation Baseline.
    Averages neighbor teacher predictions with uniform weights:
        q_bar = (1 / N) * sum( softmax(z_j / T) )
        Loss = KL(log_p_student || q_bar) * T^2
    """
    if not teacher_logits_list:
        return torch.tensor(0.0, device=student_logits.device)

    stacked = torch.stack(teacher_logits_list) / T
    p_teachers = F.softmax(stacked, dim=-1)
    p_teacher = p_teachers.mean(0).detach()

    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    return compute_kl_div(log_p_student, p_teacher, T, target_class, class_weights)


def idea3_entropy_kd_loss(student_logits, teacher_logits_list, T, target_class=-1, class_weights=None):
    """
    Uncertainty-Weighted Knowledge Distillation Baseline.
    Weights each teacher inversely by its prediction entropy on each sample.
    """
    if not teacher_logits_list:
        return torch.tensor(0.0, device=student_logits.device)

    stacked = torch.stack(teacher_logits_list) / T  # [N, B, C]
    p_teachers = F.softmax(stacked, dim=-1)

    log_p = torch.log(p_teachers + 1e-10)
    entropy = - (p_teachers * log_p).sum(-1)  # [N, B]

    weights = 1.0 / (entropy + 1e-4)  # [N, B]
    weights = weights / (weights.sum(0, keepdim=True) + 1e-10)  # [N, B]

    q_bar = (p_teachers * weights.unsqueeze(-1)).sum(0).detach()  # [B, C]

    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    return compute_kl_div(log_p_student, q_bar, T, target_class, class_weights)


def crad_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits_list: list,
    teacher_reliabilities_list: list,
    teacher_sample_counts_list: list,
    T: float,
    threshold: float = 10,
    adaptive_pre_filter: bool = False,
    soft_decay: float = 0.0,
    pre_filter_percentage: float = None,
    target_class: int = -1,
    class_weights: torch.Tensor = None,
    use_safe_harbor: bool = False,
    safe_harbor_gamma: float = 0.3,
    safe_harbor_cap: float = 50.0,
) -> torch.Tensor:
    """
    Class-wise Reliability-Aware Distillation (CRAD).

    Two-stage peer teacher aggregation:
    1. Competence gating:
       Identifies competent teachers E_c based on per-class support thresholds:
       E_c = { j | m_j^c >= thresh^c }
    2. Weighted consensus from competent teachers:
       q_tilde^c = (sum_{j in E_c} w_j^c * q_j^c) / (sum_{j in E_c} w_j^c)
    3. Class-wise agreement filtering:
       Computes absolute deviation a_j^c = |q_j^c - q_tilde^c| and discards
       outliers exceeding the median deviation tau^c among competent teachers.
    4. Precision (inverse-variance) weighted aggregation:
       q_bar^c = (sum_{j in S_c} w_j^c * q_j^c) / (sum_{j in S_c} w_j^c)
    5. KL Divergence minimization toward q_bar.
    """
    if not teacher_logits_list:
        return torch.tensor(0.0, device=student_logits.device)

    N = len(teacher_logits_list)
    device = student_logits.device

    # 0. Individual teacher distributions q_j = softmax(z_j / T)
    stacked = torch.stack(teacher_logits_list) / T
    p_teachers = F.softmax(stacked, dim=-1)  # [N, B, C]
    C = p_teachers.shape[-1]

    # Inverse-variance weights: [N, C] -> [N, 1, C]
    w_teachers = torch.stack(teacher_reliabilities_list).unsqueeze(1).to(device)  # [N, 1, C]
    w_teachers = torch.nan_to_num(w_teachers, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-class sample counts: [N, C]
    counts = torch.stack(teacher_sample_counts_list).to(device)  # [N, C]
    counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Competence gate
    class PrefilterConfig:
        pass
    cfg = PrefilterConfig()
    cfg.use_safe_harbor = use_safe_harbor
    cfg.safe_harbor_gamma = safe_harbor_gamma
    cfg.safe_harbor_cap = safe_harbor_cap
    cfg.pre_filter_percentage = pre_filter_percentage
    cfg.adaptive_pre_filter = adaptive_pre_filter
    cfg.pre_filter_soft_decay = soft_decay
    cfg.pre_filter_threshold = threshold

    mask_np, _, _ = compute_competence_mask(counts, cfg)
    competent_mask = torch.tensor(mask_np, device=device).float()
    competent_mask_expanded = competent_mask.unsqueeze(1)  # [N, 1, C]

    # Fallback to all teachers if no teacher is competent for a class
    num_competent_per_class = competent_mask.sum(dim=0)  # [C]
    fallback_classes = (num_competent_per_class == 0)

    effective_mask = competent_mask.clone()
    effective_mask[:, fallback_classes] = 1.0
    effective_mask_expanded = effective_mask.unsqueeze(1)  # [N, 1, C]

    # 2. Weighted consensus from competent teachers only
    gated_w = effective_mask_expanded * w_teachers  # [N, 1, C]
    gated_w = torch.nan_to_num(gated_w, nan=0.0, posinf=0.0, neginf=0.0)
    q_bar_pre_unnorm = torch.sum(p_teachers * gated_w, dim=0)  # [B, C]
    sum_gated = torch.sum(gated_w, dim=0)  # [B, C]
    safe_sum_gated = torch.where(sum_gated > 1e-10, sum_gated, torch.ones_like(sum_gated))
    q_bar_pre = torch.where(sum_gated > 1e-10, q_bar_pre_unnorm / safe_sum_gated, p_teachers.mean(dim=0))

    sum_pre = torch.sum(q_bar_pre, dim=-1, keepdim=True)
    q_bar_pre = torch.where(sum_pre > 1e-8, q_bar_pre / sum_pre, torch.full_like(q_bar_pre, 1.0 / C))

    # 3. Class-wise agreement filtering on competent subset
    a_j = torch.abs(p_teachers - q_bar_pre.unsqueeze(0))  # [N, B, C]

    non_competent_expanded = (1.0 - effective_mask_expanded)
    a_j_padded = a_j + non_competent_expanded * 1e6  # [N, B, C]

    num_effective = effective_mask.sum(dim=0)  # [C]
    median_idx = (num_effective.long() - 1).clamp(min=0) // 2  # [C]
    median_idx_expanded = median_idx.unsqueeze(0).unsqueeze(0).expand(1, p_teachers.shape[1], C)

    sorted_a_j, _ = torch.sort(a_j_padded, dim=0)  # [N, B, C]
    tau_c = sorted_a_j.gather(0, median_idx_expanded)  # [1, B, C]

    agreement_mask = (a_j_padded <= tau_c).float() * effective_mask_expanded  # [N, B, C]

    # 4. Final weighted aggregation from agreement-filtered competent teachers
    final_w = agreement_mask * w_teachers  # [N, B, C]
    final_w = torch.nan_to_num(final_w, nan=0.0, posinf=0.0, neginf=0.0)
    numerator = torch.sum(final_w * p_teachers, dim=0)  # [B, C]
    denominator = torch.sum(final_w, dim=0)  # [B, C]

    safe_denom = torch.where(denominator > 1e-10, denominator, torch.ones_like(denominator))
    q_bar_unnorm = torch.where(denominator > 1e-10, numerator / safe_denom, p_teachers.mean(dim=0))
    sum_q = torch.sum(q_bar_unnorm, dim=-1, keepdim=True)
    q_bar = torch.where(sum_q > 1e-8, q_bar_unnorm / sum_q, torch.full_like(q_bar_unnorm, 1.0 / C))
    q_bar = torch.nan_to_num(q_bar, nan=1.0 / C).detach()

    # 5. Distillation loss
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    return compute_kl_div(log_p_student, q_bar, T, target_class, class_weights)


# Direct aliases for CRAD function
idea4_3_prefiltered_kd_loss_v0 = crad_kd_loss
idea4_3_prefiltered_kd_loss = crad_kd_loss
crad_distillation_loss = crad_kd_loss


# ─────────────────────────────────────────────────────────────────────────────
# Baseline DFML helper functions
# ─────────────────────────────────────────────────────────────────────────────

def wsm_loss_fn(logits: torch.Tensor,
                labels: torch.Tensor,
                class_weights: torch.Tensor) -> torch.Tensor:
    """Weighted Softmax (WSM) cross-entropy loss from DFML."""
    w = class_weights.to(device=logits.device, dtype=logits.dtype)
    w_denom = w.clamp(min=1e-8)
    exp_z = torch.exp(logits)
    scaled_exp = exp_z / w_denom.unsqueeze(0)
    denom = scaled_exp.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    p_wsm = scaled_exp / denom
    batch_idx = torch.arange(labels.size(0), device=labels.device)
    p_true = p_wsm[batch_idx, labels].clamp(min=1e-10)
    return -torch.log(p_true).mean()


def dfml_kl_weighted_loss(n_idx: int,
                           all_logits_detached: list,
                           log_p_student: torch.Tensor,
                           model_param_counts: list,
                           T: float) -> torch.Tensor:
    """DFML model-size-weighted distillation component."""
    N = len(all_logits_detached)
    if N <= 1:
        return torch.tensor(0.0, device=log_p_student.device)

    denom_phi = sum(model_param_counts[q] for q in range(N) if q != n_idx)
    if denom_phi == 0:
        return torch.tensor(0.0, device=log_p_student.device)

    kl_loss = torch.tensor(0.0, device=log_p_student.device)
    for q in range(N):
        if q == n_idx:
            continue
        weight = model_param_counts[q] / denom_phi
        p_teacher = F.softmax(all_logits_detached[q] / T, dim=-1)
        kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
        kl_loss = kl_loss + weight * kl

    return kl_loss * (T ** 2)


def dfml_local_train(trainer,
                     train_data,
                     test_data,
                     device,
                     args,
                     round_idx: int,
                     class_weights: torch.Tensor) -> dict:
    """DFML client-side local training."""
    model = trainer.model
    model.to(device).train()

    if round_idx == 0:
        effective_cw = torch.ones(class_weights.size(0), device=device, dtype=torch.float32)
        effective_cw /= effective_cw.sum()
    else:
        effective_cw = class_weights.to(device=device, dtype=torch.float32)

    lr = args.lr * (args.lr_decay ** round_idx)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=args.momentum, weight_decay=args.wd,
    )
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(args.local_epochs):
        epoch_loss = []
        for batch in train_data:
            if len(batch) == 3:
                _, x, labels = batch
            else:
                x, labels = batch
            x, labels = x.to(device), labels.to(device)
            model.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss = wsm_loss_fn(logits, labels, effective_cw)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss.append(loss.item())

    metrics = trainer.test(test_data, device, args)
    model.cpu()
    return metrics


def dfml_run_mutual_learning(w_list: list,
                              arch_list: list,
                              agg_train_data,
                              device,
                              args,
                              round_idx: int,
                              alpha_t: float,
                              class_weights: torch.Tensor,
                              K: int,
                              T: float,
                              num_classes: int) -> list:
    """DFML mutual learning phase on aggregator dataset."""
    N = len(w_list)
    if N == 0:
        return w_list

    models = []
    in_channels = getattr(args, 'in_channels', 3)
    for w, arch in zip(w_list, arch_list):
        m = create_hetero_model(arch, num_classes=num_classes, in_channels=in_channels)
        m.load_state_dict(w, strict=True)
        m.to(device).train()
        models.append(m)

    param_counts = [sum(p.numel() for p in m.parameters()) for m in models]
    lr = args.lr * (args.lr_decay ** round_idx)
    optimizers = [
        torch.optim.SGD(m.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.wd)
        for m in models
    ]
    cw = class_weights.to(device=device, dtype=torch.float32)

    for _k in range(K):
        for x, labels in agg_train_data:
            x, labels = x.to(device), labels.to(device)
            all_logits_detached = []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    all_logits_detached.append(m(x))

            for n_idx, (m, opt) in enumerate(zip(models, optimizers)):
                m.train()
                opt.zero_grad()
                logits_n = m(x)
                loss_wsm = wsm_loss_fn(logits_n, labels, cw)
                log_p_n = F.log_softmax(logits_n / T, dim=-1)
                loss_kl = dfml_kl_weighted_loss(n_idx, all_logits_detached, log_p_n, param_counts, T)
                loss = (1.0 - alpha_t) * loss_wsm + alpha_t * loss_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
                opt.step()

    updated_weights = [copy.deepcopy(m.cpu().state_dict()) for m in models]
    for m in models:
        del m
    torch.cuda.empty_cache()
    return updated_weights


# ─────────────────────────────────────────────────────────────────────────────
# Client Model Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MyModelTrainer(ModelTrainer):

    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        self._cached_test_metrics = None

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        target = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        target.load_state_dict(model_parameters, strict=True)

    def get_trainable_params(self):
        return {name: param for name, param in self.model.named_parameters()}

    def train(self, train_data, test_data, device, args, round,
              neighbor_info=None, class_weights=None):
        """
        Local client training phase.
        Executes local SGD with cross-entropy and peer knowledge distillation.
        """
        model = self.model
        model.to(device)
        use_amp = device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        model.train()

        # Distillation hyperparameters
        alpha_kd = getattr(args, 'alpha_kd', 0.7)
        T = getattr(args, 'T', 4.0)

        if getattr(args, 'asymmetric_alpha', False):
            arch_name = getattr(self, '_arch_name', 'resnet18')
            if 'cnn' in arch_name:
                alpha_kd = getattr(args, 'alpha_kd_small', 0.9)
            elif 'half' in arch_name:
                alpha_kd = getattr(args, 'alpha_kd_mid', 0.5)
            else:
                alpha_kd = getattr(args, 'alpha_kd_large', 0.3)

        if round == 0:
            alpha_ce = 1.0
            current_alpha_kd = 0.0
        else:
            min_ce = 0.2
            if alpha_kd > (1.0 - min_ce):
                current_alpha_kd = (1.0 - min_ce)
                alpha_ce = min_ce
            else:
                alpha_ce = max(0.0, 1.0 - alpha_kd)
                current_alpha_kd = alpha_kd

        # Instantiate peer teacher models
        teachers = []
        teacher_reliabilities = []
        teacher_sample_counts = []
        if neighbor_info and current_alpha_kd > 0.0:
            num_classes = self._num_classes()
            in_channels = getattr(args, 'in_channels', 3)
            for entry in neighbor_info:
                if isinstance(entry, tuple):
                    if len(entry) == 5:
                        w, arch, rel, sample_counts, n_idx = entry
                        teacher_reliabilities.append(torch.nan_to_num(rel.clone().detach().to(device).float(), nan=0.0, posinf=0.0, neginf=0.0))
                        teacher_sample_counts.append(torch.nan_to_num(sample_counts.clone().detach().to(device).float(), nan=0.0, posinf=0.0, neginf=0.0))
                    elif len(entry) == 4:
                        w, arch, rel, n_idx = entry
                        if rel is not None:
                            teacher_reliabilities.append(torch.nan_to_num(rel.clone().detach().to(device).float(), nan=0.0, posinf=0.0, neginf=0.0))
                    elif len(entry) == 3:
                        w, arch, rel = entry
                        if rel is not None:
                            teacher_reliabilities.append(torch.nan_to_num(rel.clone().detach().to(device).float(), nan=0.0, posinf=0.0, neginf=0.0))
                    else:
                        w, arch = entry[0], entry[1]
                else:
                    w, arch = entry, self._infer_arch(entry)

                t = create_hetero_model(arch, num_classes=num_classes, in_channels=in_channels)
                t.load_state_dict(w, strict=True)
                t.to(device)
                t.eval()
                teachers.append(t)

        lr = args.lr * (args.lr_decay ** round)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=args.momentum, weight_decay=args.wd,
            nesterov=getattr(args, 'nesterov', False)
        )
        criterion = nn.CrossEntropyLoss().to(device)

        idea = getattr(args, 'idea', 'crad')

        for epoch in range(args.local_epochs):
            epoch_loss = []
            for batch in train_data:
                if len(batch) == 3:
                    _, x, labels = batch
                else:
                    x, labels = batch

                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()

                teacher_logits_list = []
                if teachers:
                    with torch.no_grad():
                        for t in teachers:
                            teacher_logits_list.append(t(x))

                with torch.amp.autocast('cuda', enabled=use_amp):
                    log_probs = model(x)
                    loss_ce = criterion(log_probs, labels.long())

                    if getattr(args, 'dynamic_loss_scale', False) and (not teacher_logits_list or current_alpha_kd == 0.0):
                        loss = loss_ce
                    else:
                        loss = alpha_ce * loss_ce

                    # Peer Knowledge Distillation Loss
                    if teacher_logits_list and current_alpha_kd > 0.0:
                        target_class = getattr(args, 'distill_target_class', -1)

                        if idea in ('baseline', 'uniform'):
                            loss_kd = standard_kd_loss(
                                log_probs, teacher_logits_list, T,
                                target_class=target_class, class_weights=class_weights
                            )
                        elif idea in ('idea3', 'uncertainty'):
                            loss_kd = idea3_entropy_kd_loss(
                                log_probs, teacher_logits_list, T,
                                target_class=target_class, class_weights=class_weights
                            )
                        else:
                            # CRAD aggregation (default)
                            pre_threshold = getattr(args, 'pre_filter_threshold', 10)
                            adaptive_pre_filter = getattr(args, 'adaptive_pre_filter', False)
                            soft_decay = getattr(args, 'pre_filter_soft_decay', 0.0)
                            pre_filter_percentage = getattr(args, 'pre_filter_percentage', None)
                            use_safe_harbor = getattr(args, 'use_safe_harbor', False)
                            safe_harbor_gamma = getattr(args, 'safe_harbor_gamma', 0.3)
                            safe_harbor_cap = getattr(args, 'safe_harbor_cap', 50.0)

                            loss_kd = crad_kd_loss(
                                log_probs, teacher_logits_list, teacher_reliabilities,
                                teacher_sample_counts, T,
                                threshold=pre_threshold,
                                adaptive_pre_filter=adaptive_pre_filter,
                                soft_decay=soft_decay,
                                pre_filter_percentage=pre_filter_percentage,
                                target_class=target_class, class_weights=class_weights,
                                use_safe_harbor=use_safe_harbor,
                                safe_harbor_gamma=safe_harbor_gamma,
                                safe_harbor_cap=safe_harbor_cap
                            )

                        loss = loss + current_alpha_kd * loss_kd

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss.append(loss.item())

            avg = sum(epoch_loss) / max(len(epoch_loss), 1)
            print(f"    [Client {self.id}] Epoch {epoch} Loss: {avg:.4f}")
            if self.logger:
                self.logger.info(f'Client {self.id} Epoch {epoch} Loss: {avg:.6f}')

        if round >= getattr(args, 'skip_eval_until_round', 0):
            test_metrics = self.test(test_data, device, args)
        else:
            num_classes = self._num_classes()
            test_metrics = {
                'test_correct': 0,
                'test_loss': 0.0,
                'test_total': 0,
                'per_class_correct': {c: 0 for c in range(num_classes)},
                'per_class_total': {c: 0 for c in range(num_classes)},
            }

        del teachers
        if not getattr(args, 'fast_eval', False):
            torch.cuda.empty_cache()

        return test_metrics

    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'per_class_correct': {},
            'per_class_total': {},
        }
        criterion = nn.CrossEntropyLoss().to(device)
        num_classes = self._num_classes()

        correct_by_class = torch.zeros(num_classes, dtype=torch.float32, device=device)
        total_by_class = torch.zeros(num_classes, dtype=torch.float32, device=device)

        with torch.no_grad():
            for x, target in test_data:
                x, target = x.to(device), target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

                target_flat = target.view(-1)
                predicted_flat = predicted.view(-1)

                total_by_class += torch.bincount(target_flat, minlength=num_classes).float()
                correct_mask = predicted_flat.eq(target_flat)
                if correct_mask.any():
                    correct_by_class += torch.bincount(target_flat[correct_mask], minlength=num_classes).float()

        corr_cpu = correct_by_class.cpu()
        tot_cpu = total_by_class.cpu()
        for cls in range(num_classes):
            metrics['per_class_correct'][cls] = int(corr_cpu[cls].item())
            metrics['per_class_total'][cls] = int(tot_cpu[cls].item())

        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def _num_classes(self) -> int:
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes
        if hasattr(self.model, 'linear') and isinstance(self.model.linear, nn.Linear):
            return self.model.linear.out_features
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Linear):
            return self.model.classifier.out_features
        args = getattr(self, 'args', None)
        if args is not None:
            dataset = getattr(args, 'dataset', '')
            if dataset == 'pathmnist':
                return 9
            if dataset == 'cifar100':
                return 100
            if dataset == 'cifar10':
                return 10
            if hasattr(args, 'num_classes'):
                return args.num_classes
        return 10

    def _infer_arch(self, state_dict: dict) -> str:
        return getattr(self, '_arch_name', 'resnet18')
