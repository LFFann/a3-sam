
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=2, out_conv_channels=1):
        super(Discriminator, self).__init__()
        ambiguous_channels = 16
        entmap_channels = 16
        logits_channels = 16
        embedding_channels = 64

        self.out_conv_channels = out_conv_channels

        self.ambiguous_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ambiguous_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ambiguous_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ambiguous_channels, out_channels=ambiguous_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ambiguous_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.entmap_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=entmap_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=entmap_channels, out_channels=entmap_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(entmap_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.logits_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=logits_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(logits_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=logits_channels, out_channels=logits_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(logits_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_channels, out_channels=embedding_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=embedding_channels, out_channels=out_conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def get_entropy_map(self, p):
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    def forward(self, pred_UNet, pred_YNet, pred_UNet_soft,  pred_VNet_soft, entmap1, entmap2):

        num_classes = pred_UNet_soft.shape[1]
        pred_UNet_label = torch.argmax(pred_UNet_soft, dim=1)
        pred_YNet_label = torch.argmax(pred_VNet_soft, dim=1)
        pred_UNet_one_hot = F.one_hot(pred_UNet_label, num_classes=num_classes).permute(0, 3, 1, 2)
        pred_YNet_one_hot = F.one_hot(pred_YNet_label, num_classes=num_classes).permute(0, 3, 1, 2)

        ambiguous_area = torch.abs(pred_UNet_one_hot - pred_YNet_one_hot).to(dtype=torch.float32)
        uncertainty_area = torch.cat((entmap1, entmap2), dim=1)
        pred_logits = torch.cat((pred_UNet, pred_YNet), dim=1)

        ambiguous_info = self.ambiguous_conv(1 - ambiguous_area)
        uncertainty_info = self.entmap_conv(1 - uncertainty_area)
        pred_info = self.logits_conv(pred_logits)

        x = torch.cat((ambiguous_info, uncertainty_info, pred_info), dim=1)

        x = self.final_conv(x)

        return x
