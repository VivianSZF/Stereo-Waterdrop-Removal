import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torchvision import models
from models.rattnet.row_attention_softindex_levelpos_recepdilated import row_attention_maxindex

class rattnet_index(nn.Module):
    def __init__(self, backbone='resnet'):
        super(rattnet_index, self).__init__()
        self.feature_extractor = feature_extractor_res()
        self.row_attention = row_attention_maxindex()
        self.refinement = refinement_new(n_blocks=16)
        self.dis_refinement = dis_refinement()
        

    def forward(self, left, right): 
        left_features = self.feature_extractor(left)
        right_features = self.feature_extractor(right)
        new_features_l, new_features_r, index_l, index_r = self.row_attention(left_features, right_features)
        
        final_left = self.refinement(left_features, new_features_l)
        final_right = self.refinement(right_features, new_features_r)

        final_index_l = self.dis_refinement(index_l)
        final_index_r = self.dis_refinement(index_r, minus=True)
        return final_left, final_right, final_index_l, final_index_r


class refinement_new(nn.Module):
    def __init__(self, in_channels=1024, n_blocks=16):
        super(refinement_new, self).__init__()
        self.head_1 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_1 = nn.Sequential(
            *[ResBlock(in_channels//2) for _ in range(n_blocks)],
        )
        self.tail_1 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LeakyReLU(0.1, True),
        )
        self.head_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_2 = nn.Sequential(
            *[ResBlock(in_channels//4) for _ in range(n_blocks)],
        )
        self.tail_2 = nn.Sequential(
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LeakyReLU(0.1, True),
        )
        self.head_3 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_3 = nn.Sequential(
            *[ResBlock(in_channels//8) for _ in range(n_blocks)],
        )
        self.tail_3 = nn.Sequential(
            nn.Conv2d(in_channels//8, in_channels//8, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LeakyReLU(0.1, True),
        )
        self.head_4 = nn.Sequential(
            nn.Conv2d(in_channels//8, in_channels//16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_4 = nn.Sequential(
            *[ResBlock(in_channels//16) for _ in range(n_blocks)],
        )
        self.tail_4 = nn.Sequential(
            nn.Conv2d(in_channels//16, in_channels//16, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LeakyReLU(0.1, True),
        )
        self.final = nn.Conv2d(in_channels//16, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, ori_features, new_features):
        t = torch.cat([ori_features[-1], new_features[0]], 1)
        t = self.head_1(t)
        t = self.body_1(t)
        x = self.tail_1(t)

        t = torch.cat([x, new_features[1]], 1)
        t = self.head_2(t)
        t = self.body_2(t)
        x = self.tail_2(t)

        t = torch.cat([x, new_features[2]], 1)
        t = self.head_3(t)
        t = self.body_3(t)
        x = self.tail_3(t)

        t = x

        t = self.head_4(t)
        t = self.body_4(t)
        x = self.tail_4(t)

        x = self.final(x)

        return x


class feature_extractor_res(nn.Module):
    def __init__(self):
        super(feature_extractor_res, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu 
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 
        self.res3 = resnet.layer2 
        self.res4 = resnet.layer3 

    def forward(self, x):
        feature_maps = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.res2(x)
        feature_maps.append(x)
        x = self.res3(x)
        feature_maps.append(x)
        x = self.res4(x)
        feature_maps.append(x)

        return feature_maps


class dis_refinement(nn.Module):
    def __init__(self):
        super(dis_refinement, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, padding_mode='reflect')

    def forward(self, index_l, minus=False):
        if minus:
            index_l[0] = -index_l[0]
            index_l[1] = -index_l[1]
            index_l[2] = -index_l[2]

        _, c, h, w = index_l[0].size()
        x = F.interpolate(index_l[0], scale_factor=2, mode='nearest')
        x = x*2
        x = self.conv1(torch.cat([x, index_l[1]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x*2
        x = self.conv2(torch.cat([x, index_l[2]], dim=1))
        x = F.interpolate(x, scale_factor=4, mode='nearest')

        return x*4


class ResBlock(nn.Module):
    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x