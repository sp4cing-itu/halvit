import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# HalvitResnet50Bottleneck
# =====================================================================================
class HalvitResnet50Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, num_blocks, stride):
        super().__init__()
        
        self.convp = nn.Parameter(torch.empty((mid_channels, out_channels, 1, 1)))
        nn.init.xavier_normal_(self.convp)

        self.processing_blocks = nn.ModuleList()
        self.final_bns = nn.ModuleList()

        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            self.processing_blocks.append(nn.Sequential(
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, mid_channels, 3, block_stride, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU()
            ))
            self.final_bns.append(nn.BatchNorm2d(out_channels))

        self.downsample_path = None
        self.main_path_upsampler = None
        
        if in_channels != out_channels:
            num_new_channels = out_channels - in_channels
            self.main_path_upsampler = nn.Sequential(
                nn.Conv2d(in_channels, num_new_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(num_new_channels),
                nn.GELU()
            )
        
        if stride != 1:
            self.downsample_path = nn.Sequential(
                 nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
                 nn.BatchNorm2d(out_channels),
                 nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        if self.main_path_upsampler:
            x = torch.cat([self.main_path_upsampler(x), x], dim=1)
        if self.downsample_path:
             identity = self.downsample_path(x)
        elif self.main_path_upsampler:
             identity = x
        
        x_main = x
        for i in range(len(self.processing_blocks)):
            identity_of_block = x_main
            
            out = F.conv2d(x_main, self.convp)
            out = self.processing_blocks[i](out)
            out = F.conv2d(out, self.convp.permute(1, 0, 2, 3))
            out = self.final_bns[i](out)
            
            if i == 0:
                x_main = out + identity
            else:
                x_main = out + identity_of_block
            
            x_main = F.relu(x_main)
            
        return x_main

# =====================================================================================
# StandardBottleneck
# =====================================================================================
class StandardBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels); self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels); self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; return self.relu(out)

# =====================================================================================
# ResNet50 _make_layer
# =====================================================================================
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 3, 1, use_halvit_structure=False)
        self.layer2 = self._make_layer(128, 4, 2, use_halvit_structure=False)
        self.layer3 = self._make_layer(256, 6, 2, use_halvit_structure=True)
        self.layer4 = self._make_layer(512, 3, 2, use_halvit_structure=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, mid_channels, blocks, stride, use_halvit_structure=False):
        expansion = 4
        
        if use_halvit_structure:
            out_channels = mid_channels * expansion
            module = HalvitResnet50Bottleneck(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                num_blocks=blocks,
                stride=stride
            )
            self.in_channels = out_channels
            return module
        else:
            out_channels_std = mid_channels * expansion
            downsample = None
            if stride != 1 or self.in_channels != out_channels_std:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels_std, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels_std)
                )
            layers = [StandardBottleneck(self.in_channels, mid_channels, stride, downsample)]
            self.in_channels = out_channels_std
            for _ in range(1, blocks):
                layers.append(StandardBottleneck(self.in_channels, mid_channels))
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.global_pool(x); x = torch.flatten(x, 1)
        return x

# --- Test ---
if __name__ == "__main__":
    model = ResNet50(num_classes=1000)
    total_params = sum(p.numel() for n, p in model.named_parameters() if 'fc' not in n)
    print(f"halvit resnet50 parameter num: {total_params:,}")
