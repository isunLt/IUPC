import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsparse
import torchsparse.nn as spnn


class ProjectHead2d(nn.Module):
    def __init__(self, in_dim, inner_dim=512, out_dim=256):
        super(ProjectHead2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=True)


class BasicBlock2d(nn.Module):
    """
    两层3*3卷积
    Input: [B, inplanes, H, W]
    Output: [B, outplanes, (H-1)/2+1, (W-1)/2+1]
    """

    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes, 2 * stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out


class ConvBlock2d(nn.Module):

    def __init__(self, inplanes, outplanes, ks=3, stride=1, dilation=1):
        super(ConvBlock2d, self).__init__()
        self.net = nn.Sequential(
            conv3x3(in_planes=inplanes, out_planes=outplanes, stride=stride),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DeConvBlock2d(nn.Module):

    def __init__(self, inplanes, outplanes, ks=3, stride=1, dilation=1):
        super(DeConvBlock2d, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=inplanes, out_channels=outplanes, kernel_size=ks, stride=stride),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock2d(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1):
        super(ResBlock2d, self).__init__()
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=outplanes, stride=stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=outplanes, out_planes=outplanes)
        self.downsample = nn.Sequential(
            conv1x1(in_planes=inplanes, out_planes=outplanes, stride=stride),
            nn.BatchNorm2d(outplanes)
        ) if not (inplanes == outplanes and stride == 1) else None
        self.bn2 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc)
        )

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
