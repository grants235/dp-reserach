"""
Model architectures for Channeled DP-SGD experiments.

All BatchNorm layers are replaced with GroupNorm (num_groups=16) for DP-SGD
compatibility. Verified via Opacus ModuleValidator.

Architectures:
- WideResNet-28-2  (~1.5M params)  primary model
- ResNet-20        (~0.27M params) secondary model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_GROUPS = 16  # GroupNorm groups, per spec


# ---------------------------------------------------------------------------
# WideResNet
# ---------------------------------------------------------------------------

class WideBasicBlock(nn.Module):
    """Wide residual block with GroupNorm and optional dropout."""

    def __init__(self, in_planes: int, out_planes: int, stride: int,
                 dropout_rate: float = 0.0, n_groups: int = N_GROUPS):
        super().__init__()
        self.bn1 = nn.GroupNorm(n_groups, in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.GroupNorm(n_groups, out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=stride,
                               padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """
    WideResNet-d-k for CIFAR (32x32 input).

    depth d = 28, widen_factor k = 2 gives ~1.5M params.
    Number of blocks per group = (d - 4) / 6 = 4 for d=28.
    Channel widths: [16, 16*k, 32*k, 64*k].
    """

    def __init__(self, depth: int = 28, widen_factor: int = 2,
                 dropout_rate: float = 0.0, num_classes: int = 10,
                 n_groups: int = N_GROUPS):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must satisfy (d-4) % 6 == 0"
        n_blocks = (depth - 4) // 6
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, nChannels[0], 3, padding=1, bias=False)

        self.layer1 = self._make_layer(WideBasicBlock, n_blocks,
                                       nChannels[0], nChannels[1], stride=1,
                                       dropout_rate=dropout_rate, n_groups=n_groups)
        self.layer2 = self._make_layer(WideBasicBlock, n_blocks,
                                       nChannels[1], nChannels[2], stride=2,
                                       dropout_rate=dropout_rate, n_groups=n_groups)
        self.layer3 = self._make_layer(WideBasicBlock, n_blocks,
                                       nChannels[2], nChannels[3], stride=2,
                                       dropout_rate=dropout_rate, n_groups=n_groups)

        self.bn = nn.GroupNorm(n_groups, nChannels[3])
        self.fc = nn.Linear(nChannels[3], num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(block, n_blocks, in_planes, out_planes, stride,
                    dropout_rate, n_groups):
        layers = [block(in_planes, out_planes, stride, dropout_rate, n_groups)]
        for _ in range(1, n_blocks):
            layers.append(block(out_planes, out_planes, 1, dropout_rate, n_groups))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def features(self, x):
        """Return penultimate-layer features (before fc)."""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


# ---------------------------------------------------------------------------
# ResNet-20 (He et al., 2016 CIFAR variant)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Standard residual block with GroupNorm."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1,
                 n_groups: int = N_GROUPS):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.GroupNorm(n_groups, out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(n_groups, out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.GroupNorm(n_groups, out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR-10/100 with GroupNorm (~0.27M params)."""

    def __init__(self, num_classes: int = 10, n_groups: int = N_GROUPS):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(n_groups, 16)

        self.layer1 = self._make_layer(16, 16, 3, stride=1, n_groups=n_groups)
        self.layer2 = self._make_layer(16, 32, 3, stride=2, n_groups=n_groups)
        self.layer3 = self._make_layer(32, 64, 3, stride=2, n_groups=n_groups)

        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(in_planes, out_planes, n_blocks, stride, n_groups):
        layers = [BasicBlock(in_planes, out_planes, stride, n_groups)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_planes, out_planes, 1, n_groups))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_model(arch: str, num_classes: int, n_groups: int = N_GROUPS) -> nn.Module:
    """
    arch: 'wrn28-2' | 'resnet20'
    Returns model with GroupNorm instead of BatchNorm.
    """
    if arch == "wrn28-2":
        return WideResNet(depth=28, widen_factor=2, dropout_rate=0.0,
                          num_classes=num_classes, n_groups=n_groups)
    elif arch == "resnet20":
        return ResNet20(num_classes=num_classes, n_groups=n_groups)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def validate_model_for_dp(model: nn.Module) -> bool:
    """
    Use Opacus ModuleValidator to verify there are no incompatible layers
    (e.g. BatchNorm). Returns True if model is DP-compatible.
    """
    try:
        from opacus.validators import ModuleValidator
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print("DP validation errors:")
            for e in errors:
                print(f"  {e}")
            return False
        return True
    except ImportError:
        # Opacus not installed: manual check for BatchNorm
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                print(f"WARNING: BatchNorm layer found at {name}")
                return False
        return True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
