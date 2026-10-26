import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# ── 1. MobileNetV2 for CIFAR-10 (32x32) ────────────────────────────────────────
def get_cifar_mobilenetv2(num_classes=10, in_channels=3):
    model = models.mobilenet_v2(weights=None)
    # Adjust first layer: kernel 3x3, stride 1, padding 1 (no downsampling)
    model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
    # Adjust classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ── 2. ShuffleNetV2 for CIFAR-10 (32x32) ────────────────────────────────────────
def get_cifar_shufflenetv2(num_classes=10, in_channels=3):
    model = models.shufflenet_v2_x1_0(weights=None)
    # Adjust first conv layer: kernel 3x3, stride 1, padding 1
    model.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False)
    # Disable initial maxpooling to prevent excessive feature map reduction
    model.maxpool = nn.Identity()
    # Adjust classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ── 3. EfficientNet-B0 for CIFAR-10 (32x32) ───────────────────────────────────
def get_cifar_efficientnet_b0(num_classes=10, in_channels=3):
    model = models.efficientnet_b0(weights=None)
    # Adjust first layer: kernel 3x3, stride 1, padding 1
    model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
    # Adjust classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ── 4. DenseNet-BC (L=40, k=12) for CIFAR-10 (32x32) ──────────────────────────
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # Concatenate input with the output
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNetBC(nn.Module):
    def __init__(self, nblocks, growth_rate=12, reduction=0.5, num_classes=10, in_channels=3):
        super(DenseNetBC, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(in_channels, num_planes, kernel_size=3, padding=1, bias=False)

        # Dense Block 1 and Transition 1
        self.dense1 = self._make_denseblock(BottleneckBlock, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = TransitionBlock(num_planes, out_planes)
        num_planes = out_planes

        # Dense Block 2 and Transition 2
        self.dense2 = self._make_denseblock(BottleneckBlock, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = TransitionBlock(num_planes, out_planes)
        num_planes = out_planes

        # Dense Block 3
        self.dense3 = self._make_denseblock(BottleneckBlock, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_denseblock(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_cifar_densenet_bc(num_classes=10, in_channels=3):
    # L=40 -> 6 blocks per group
    return DenseNetBC([6, 6, 6], growth_rate=12, reduction=0.5, num_classes=num_classes, in_channels=in_channels)
