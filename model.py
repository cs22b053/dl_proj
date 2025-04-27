import torch
import torch.nn as nn
import torch.nn.functional as F

class DWR(nn.Module):
    """Dilation-wise Residual module (from YOLOv8-UC)"""
    def __init__(self, channels, dilation_rates=(1,2,3)):
        super(DWR, self).__init__()
        # Step 1: basic 3x3 conv
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        # Split channels into groups for dilated convolutions
        self.num_groups = len(dilation_rates)
        if channels % self.num_groups != 0:
            raise ValueError("Channels must be divisible by number of dilation rates")
        group_ch = channels // self.num_groups
        # Step 2: parallel dilated depthwise convs (one per group)
        self.dilated_convs = nn.ModuleList()
        for r in dilation_rates:
            self.dilated_convs.append(
                nn.Conv2d(group_ch, group_ch, kernel_size=3, padding=r, dilation=r, 
                          groups=group_ch, bias=False)
            )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv_pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        splits = torch.split(out, out.size(1) // self.num_groups, dim=1)
        outs = []
        for split, conv in zip(splits, self.dilated_convs):
            outs.append(conv(split))
        out = torch.cat(outs, dim=1)
        out = self.relu(self.bn2(out))
        out = self.conv_pointwise(out)
        # Residual connection
        return out + x

class C2fDWR(nn.Module):
    """C2f block with DWR modules instead of Bottleneck (C2f-DWR)"""
    def __init__(self, in_channels, out_channels, num_dwr=1, expansion=0.5):
        super(C2fDWR, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        # Stack of DWR modules
        self.dwrs = nn.Sequential(*[DWR(hidden_channels) for _ in range(num_dwr)])
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dwrs(out)
        out = self.bn2(self.conv2(out))
        return out

class LSKA(nn.Module):
    """Large Separable Kernel Attention (LSKA) module"""
    def __init__(self, channels, kernel_size=13):
        super(LSKA, self).__init__()
        # Depthwise convs with large separable kernels
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                                  padding=(kernel_size//2, 0), groups=channels, bias=False)
        self.dw_conv2 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                                  padding=(0, kernel_size//2), groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.dw_conv1(x)
        out = self.dw_conv2(out)
        out = self.pw_conv(out)
        scale = self.sigmoid(out)
        return x * scale  # spatial attention

class SPPFLSKA(nn.Module):
    """SPPF with embedded LSKA attention (SPPF-LSKA)"""
    def __init__(self, in_channels, out_channels, pool_kernel=5):
        super(SPPFLSKA, self).__init__()
        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=1, padding=pool_kernel//2)
        self.conv2 = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lska = LSKA(out_channels)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.pool(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.lska(out)  # apply LSKA attention
        return out

class RepConv(nn.Module):
    """RepConv: Re-parameterizable conv with 3x3 and 1x1 branches"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RepConv, self).__init__()
        pad = kernel_size // 2
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv3x3(x) + self.conv1x1(x)
        out = self.bn(out)
        return out

class RepHead(nn.Module):
    """RepHead detection head with shared RepConv for classification and regression"""
    def __init__(self, in_channels, num_classes):
        super(RepHead, self).__init__()
        # Shared RepConv layers
        self.repconv1 = RepConv(in_channels, in_channels)
        self.repconv2 = RepConv(in_channels, in_channels)
        # Classification branch
        self.cls_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # Regression branch: 4 coords + 1 objectness
        self.reg_conv = nn.Conv2d(in_channels, 5, kernel_size=1)
    def forward(self, x):
        x = F.relu(self.repconv1(x))
        x = F.relu(self.repconv2(x))
        cls_out = self.cls_conv(x)
        reg_out = self.reg_conv(x)
        return cls_out, reg_out

class YOLOv8UC(nn.Module):
    """YOLOv8-UC model assembly (simplified)"""
    def __init__(self, num_classes=5):
        super(YOLOv8UC, self).__init__()
        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Replace last two C2f modules with C2f-DWR
        self.c2f1 = C2fDWR(32, 64, num_dwr=1)
        self.c2f2 = C2fDWR(64, 128, num_dwr=1)
        # SPPF-LSKA neck
        self.sppf = SPPFLSKA(128, 128)
        # Detection head
        self.head = RepHead(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.c2f1(x)
        x = self.c2f2(x)
        x = self.sppf(x)
        cls_out, reg_out = self.head(x)
        # Outputs: (B, num_classes, H, W) and (B, 5, H, W)
        return cls_out, reg_out
