import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


# feature linear enhance module
class FeatureLinearEnhancement(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureLinearEnhancement, self).__init__()

        self.norm_layer = nn.Sequential(nn.Conv2d(in_channel // 4 * 3, out_channel, 3, 1, 1, bias=True),
                                        nn.BatchNorm2d(out_channel, affine=False),
                                        nn.ReLU(inplace=True))

        self.align_layer = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)

        self.conv_shared = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True))
        self.conv_gamma = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.normal_(0, 0.01)
        self.conv_beta.weight.data.normal_(0, 0.01)
        self.conv_gamma.bias.data.fill_(0)
        self.conv_beta.bias.data.fill_(0)

    def forward(self, shared_feature, guide_feature):
        # 8 * 64 * 256 * 256
        aligned_feature = self.align_layer(shared_feature)
        feature_concat = self.conv_shared(aligned_feature)
        # 8 * 32 * 256 * 256
        guide_normed = self.norm_layer(guide_feature)
        # 8 * 32 * 256 * 256
        gamma = self.conv_gamma(feature_concat)
        beta = self.conv_beta(feature_concat)
        # 8 * 32 * 256 * 256
        # shared_feature = self.align_layer(shared_feature)
        b, c, h, w = aligned_feature.size()
        # 8 * 32 * 65536
        aligned_feature = aligned_feature.view(b, c, h * w)
        # 8 * 32 * 1 * 1
        source_mean = torch.mean(aligned_feature, dim=-1, keepdim=True).unsqueeze(3)
        source_std = torch.std(aligned_feature, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + source_std
        beta = beta + source_mean

        return guide_normed * gamma + beta


# dual residual fusion
class DualResidualFusion(nn.Module):
    def __init__(self, in_channel, out_factor):
        super(DualResidualFusion, self).__init__()
        self.align_layer = nn.Conv2d(in_channel // 2, in_channel, 3, 1, 1, bias=True)
        self.conv_down_a = nn.Conv2d(in_channel, in_channel, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(in_channel, in_channel, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(in_channel, in_channel, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(in_channel, in_channel, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(in_channel * 2, in_channel * out_factor, 3, 1, 1, bias=True)
        self.active = nn.ReLU(inplace=True)

    def forward(self, source_feature, enhanced_feature):
        # 8 * 32 * 128 * 128
        source_feature = self.conv_down_a(self.align_layer(source_feature))
        # 8 * 32 * 128 * 128
        res_a = self.active(self.conv_down_a(enhanced_feature)) - source_feature
        # 8 * 32 * 256 * 256
        out_a = self.active(self.conv_up_a(res_a)) + enhanced_feature

        # 8 * 32 * 128 * 128
        res_b = source_feature - self.active(self.conv_down_b(enhanced_feature))
        # 8 * 32 * 256 * 256
        out_b = self.active(self.conv_up_b(res_b + source_feature))
        # 8 * 64 * 256 * 256
        out = self.active(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out


class PrivateBlock(nn.Module):
    def __init__(self, out_channel):
        super(PrivateBlock, self).__init__()

        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.shut_cut = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, in_feature):
        out = F.relu(self.bn1(self.conv1(in_feature)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        return F.relu(out + in_feature, inplace=True)


class PrivateFeatureExtractor(nn.Module):
    def __init__(self, num_layer, in_channel, out_channel):
        super(PrivateFeatureExtractor, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, kernel_size=3, padding=1),
            nn.Conv2d(out_channel // 4, out_channel // 2, kernel_size=3, padding=1),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        self.feature_extractor = nn.ModuleList([PrivateBlock(out_channel=out_channel) for _ in range(num_layer)])

    def forward(self, in_feature):
        out_feature = self.init_conv(in_feature)

        for layer in self.feature_extractor:
            out_feature = layer(out_feature)

        return out_feature


class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, guide_layers=5, source_layers=3, shared_backbone='resnet18'):
        super(MultiModalFeatureExtractor, self).__init__()

        self.guide_extractor = PrivateFeatureExtractor(num_layer=guide_layers, in_channel=3, out_channel=48)
        self.source_extractor = PrivateFeatureExtractor(num_layer=source_layers, in_channel=1, out_channel=16)

        self.shared_extractor = smp.UnetPlusPlus(shared_backbone, classes=64, in_channels=4, encoder_weights='imagenet')

        self.lfem = FeatureLinearEnhancement(in_channel=64, out_channel=32)
        self.drfm = DualResidualFusion(in_channel=32, out_factor=2)

    def forward(self, guide, source):
        # 8 * 48 * 256 * 256
        guide_feature = self.guide_extractor(guide)
        # 8 * 16 * 256 * 256
        source_feature = self.source_extractor(source)
        # 8 * 64 * 256 * 256
        common_feature = self.shared_extractor(torch.cat([source, guide], dim=1))

        # 8 * 32 * 256 * 256
        enhanced_feature = self.lfem(common_feature, guide_feature)
        # 8 * 64 * 256 * 256
        out = self.drfm(source_feature, enhanced_feature)

        return out


class VarEstimateModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(VarEstimateModule, self).__init__()

        self.var_conv = nn.Sequential(*[nn.Conv2d(in_channel, in_channel, kernel_size, padding=(kernel_size // 2)),
                                        nn.ELU(inplace=True),
                                        nn.Conv2d(in_channel, in_channel, kernel_size, padding=(kernel_size // 2)),
                                        nn.ELU(inplace=True),
                                        nn.Conv2d(in_channel, out_channel, kernel_size, padding=(kernel_size // 2)),
                                        nn.ELU(inplace=True)
                                        ])

    def forward(self, x):
        return self.var_conv(x)
