"""
Heterogeneous Model Registry & Architectures.

Defines the pool of neural network architectures (e.g., ResNet, CNN) 
and provides factory methods for model creation and round-robin 
architecture assignment across clients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# ResNet building blocks
# ──────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetHetero(nn.Module):
    """Parameterized ResNet — width_mult controls channel scaling."""

    def __init__(self, block, num_blocks, num_classes=10, width_mult=1.0, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        base = [64, 128, 256, 512]
        ch = [max(1, int(c * width_mult)) for c in base]
        self.in_planes = ch[0]

        self.conv1  = nn.Conv2d(in_channels, ch[0], 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(ch[0])
        self.layer1 = self._make_layer(block, ch[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ch[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ch[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ch[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(ch[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def forward_features(self, x):
        """Return (spatial_features, pooled_vector) for intermediate feature extraction."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        spatial = out                              # [B, C, H, W]
        pooled = F.adaptive_avg_pool2d(out, 1)
        pooled = pooled.view(pooled.size(0), -1)   # [B, C]
        return spatial, pooled


# ──────────────────────────────────────────────────────────────────────────────
# Simple CNN (architecture-heterogeneous baseline)
# ──────────────────────────────────────────────────────────────────────────────

class CNN6(nn.Module):
    """6-layer CNN — distinct architecture family from ResNet."""

    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.features(x).view(x.size(0), -1)
        return self.classifier(out)

    def forward_features(self, x):
        """Return (spatial_features, pooled_vector) for intermediate feature extraction."""
        # Run all layers except the final AdaptiveAvgPool2d (index -1)
        spatial = self.features[:-1](x)              # [B, 256, H, W]
        pooled = self.features[-1](spatial)           # AdaptiveAvgPool2d → [B, 256, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)     # [B, 256]
        return spatial, pooled


# ──────────────────────────────────────────────────────────────────────────────
# Extreme Specialist (MoE) Baselines
# ──────────────────────────────────────────────────────────────────────────────

class CNN6Micro(nn.Module):
    """A tiny version of CNN6 (~20k parameters)."""
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, 3, padding=1), nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 20, 3, padding=1), nn.BatchNorm2d(20), nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16
            nn.Conv2d(20, 40, 3, padding=1), nn.BatchNorm2d(40), nn.ReLU(),
            nn.Conv2d(40, 40, 3, padding=1), nn.BatchNorm2d(40), nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8
            nn.Conv2d(40, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(80, num_classes)

    def forward(self, x):
        out = self.features(x).view(x.size(0), -1)
        return self.classifier(out)


class MoETinyExperts_Broken(nn.Module):
    """
    Ensemble of 10 isolated experts. 
    Expert 'i' only sees images of class 'i' during training (Positive-Only Paradox).
    """
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.experts = nn.ModuleList([CNN6Micro(num_classes, in_channels=in_channels) for _ in range(num_classes)])

    def forward(self, x, labels=None):
        B = x.size(0)
        if labels is None or not self.training:
            combined_logits = torch.zeros(B, self.num_classes, device=x.device)
            for i in range(self.num_classes):
                expert_out = self.experts[i](x)
                combined_logits[:, i] = expert_out[:, i]
            return combined_logits

        out_logits = torch.zeros(B, self.num_classes, device=x.device)
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                expert_out = self.experts[i](x[mask])
                out_logits[mask] = expert_out
        return out_logits


class MoETinyExperts_OVA(nn.Module):
    """
    Ensemble of 10 experts without the isolation flaw.
    All experts see all images, learning both positive and negative features.
    """
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.experts = nn.ModuleList([CNN6Micro(2, in_channels=in_channels) for _ in range(num_classes)])

    def forward(self, x, labels=None):
        B = x.size(0)
        combined_logits = torch.zeros(B, self.num_classes, device=x.device)
        for i in range(self.num_classes):
            # Expert i predicts [Not_Class_i, Is_Class_i]. We take the positive logit.
            expert_out = self.experts[i](x)
            combined_logits[:, i] = expert_out[:, 1]
        return combined_logits


class LightweightRouter(nn.Module):
    """A very fast, 3-layer CNN designed strictly to route images, not classify them."""
    def __init__(self, num_experts=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_experts)

    def forward(self, x):
        out = self.features(x).view(x.size(0), -1)
        return self.classifier(out)


class MoETinyExperts_Router(nn.Module):
    """
    True Mixture of Experts (MoE) with a Learned Router.
    The Router (Gating Network) dynamically assigns trust to each expert based on the input image.
    """
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        # The gating network (Router) is lightweight (~14k params)
        self.router = LightweightRouter(num_classes, in_channels=in_channels)
        # The 10 specialized experts (6-layer, 113k params each)
        self.experts = nn.ModuleList([CNN6Micro(num_classes, in_channels=in_channels) for _ in range(num_classes)])

    def forward(self, x, labels=None):
        # 1. Router computes weights (probabilities) for each expert
        # Shape: [B, num_classes]
        routing_logits = self.router(x)
        routing_weights = torch.nn.functional.softmax(routing_logits, dim=1)

        # 2. Experts compute their predictions
        # Shape: [B, num_experts, num_classes]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # 3. Combine expert outputs based on router weights
        # routing_weights: [B, num_experts, 1]
        # output Shape: [B, num_classes]
        out_logits = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=1)
        
        return out_logits


# ──────────────────────────────────────────────────────────────────────────────
# Architecture Registry & Assignment
#
# Defines the pool of models used for architectural heterogeneity.
# Unlike FedAvg, TypicalKD supports knowledge transfer across varying architectures.
# ──────────────────────────────────────────────────────────────────────────────
# Global pool of architectures available for assignment
ARCH_POOL = ["resnet18", "resnet18_half", "cnn6", "mobilenetv2", "shufflenetv2", "efficientnet_b0", "densenet_bc"]


def create_hetero_model(arch_name: str, num_classes: int = 10, tiny_experts: str = 'none', in_channels: int = 3) -> nn.Module:
    """
    Factory function — create a model by name.

    Args:
        arch_name   : one of ARCH_POOL
        num_classes : output dimension (10 for CIFAR-10)
        tiny_experts: MoE variants
        in_channels : number of input channels (3 for RGB, 1 for Grayscale)
    Returns:
        nn.Module with logit output [B, num_classes]
    """
    if arch_name == "resnet18":
        return ResNetHetero(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_mult=1.0, in_channels=in_channels)
    elif arch_name == "resnet18_half":
        # Same depth, half the channels → width heterogeneity
        return ResNetHetero(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_mult=0.5, in_channels=in_channels)
    elif arch_name == "cnn6":
        if tiny_experts == 'broken':
            return MoETinyExperts_Broken(num_classes=num_classes, in_channels=in_channels)
        elif tiny_experts == 'ova':
            return MoETinyExperts_OVA(num_classes=num_classes, in_channels=in_channels)
        elif tiny_experts == 'router':
            return MoETinyExperts_Router(num_classes=num_classes, in_channels=in_channels)
        return CNN6(num_classes=num_classes, in_channels=in_channels)
    elif arch_name == "pyramidnet":
        from utils.pyramidnet import PyramidNet
        return PyramidNet(depth=272, alpha=200, num_classes=num_classes, bottleneck=True)
    elif arch_name == "mobilenetv2":
        from utils.lightweight_models import get_cifar_mobilenetv2
        return get_cifar_mobilenetv2(num_classes=num_classes, in_channels=in_channels)
    elif arch_name == "shufflenetv2":
        from utils.lightweight_models import get_cifar_shufflenetv2
        return get_cifar_shufflenetv2(num_classes=num_classes, in_channels=in_channels)
    elif arch_name == "efficientnet_b0":
        from utils.lightweight_models import get_cifar_efficientnet_b0
        return get_cifar_efficientnet_b0(num_classes=num_classes, in_channels=in_channels)
    elif arch_name == "densenet_bc":
        from utils.lightweight_models import get_cifar_densenet_bc
        return get_cifar_densenet_bc(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Choose from {ARCH_POOL}")


def assign_architectures(num_clients: int, pool: list = None) -> list:
    """
    Round-robin architecture assignment across clients.
    Ensures architectural diversity even with small client counts.
    """
    target_pool = pool if pool is not None else ARCH_POOL
    return [target_pool[i % len(target_pool)] for i in range(num_clients)]
