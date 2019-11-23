# encoding: utf-8


import random
import torch
from torch import nn
import torchvision.models as models
from .backbones.resnet import ResNet
from .layers.SE_Resnet import SEResnet
from .layers.SE_module import SELayer

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, data_set, width_ratio=1.0, height_ratio=0.33):
        super(Baseline, self).__init__()

        self.mix_conv = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False))
        self.preact = SEResnet('resnet101')
        self.pose_se = SELayer(1024)
        self.global_se = SELayer(1024)
        self.layer_lis = [[3, 4, 6, 3], [3, 4, 23, 3]]
        self.base = ResNet(self.layer_lis[1], last_stride)
        self.data = data_set

        res = models.resnet101(pretrained=True)
        model_dict = self.base.state_dict()
        res_pretrained_dict = res.state_dict()
        res_pretrained_dict = {k: v for k, v in res_pretrained_dict.items() if k in model_dict}
        model_dict.update(res_pretrained_dict)
        self.base.load_state_dict(model_dict)
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),


        )
        self.layer4.load_state_dict(res.layer4.state_dict())

        self.bfe_layer4 = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.bfe_layer4.load_state_dict(res.layer4.state_dict())

        self.batch_erase = BatchDrop(height_ratio, width_ratio)

        self.bfe_gap = nn.AdaptiveAvgPool2d(1)
        self.pose_gap = nn.AdaptiveMaxPool2d(1)
        self.lead_gap = nn.AdaptiveMaxPool2d(1)
        self.lead_avg_gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.lead_bottleneck = nn.BatchNorm1d(2048)
        self.lead_bottleneck.bias.requires_grad_(False)
        self.lead_classifier = nn.Linear(2048, self.num_classes, bias=False)

        self.part1_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.part1_bottleneck.bias.requires_grad_(False)
        self.part1_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.lead_bottleneck.apply(weights_init_kaiming)
        self.lead_classifier.apply(weights_init_classifier)
        self.mix_conv.apply(weights_init_kaiming)
        self.part1_bottleneck.apply(weights_init_kaiming)
        self.part1_classifier.apply(weights_init_classifier)


    def forward(self, x):
        pose_out = self.preact(x)
        pose_out = self.pose_se(pose_out)
        a, lead = self.base(x)

        pose_part = torch.cat([a, pose_out], dim=1)
        pose_part = self.mix_conv(pose_part)
        pose_part = self.global_se(pose_part)
        global_pose_part = self.layer4(pose_part)
        pose_part_feat = self.pose_gap(global_pose_part)
        pose_part_feat = pose_part_feat.view(pose_part_feat.shape[0], -1)
        pose_part_feat = nn.functional.normalize(pose_part_feat)\

        x = self.bfe_layer4(pose_part)
        if self.training and random.randint(0, 1):
            x = self.batch_erase(x)
        global_feat = self.bfe_gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = nn.functional.normalize(global_feat)

        lead_feat = self.lead_gap(lead) + self.lead_avg_gap(lead)
        lead_feat = lead_feat.view(lead_feat.shape[0], -1)
        lead_feat = nn.functional.normalize(lead_feat)

        if(self.data == 'cuhk03'):
            feat1 = global_feat
            feat2 = pose_part_feat
            feat3 = lead_feat
        else:
            feat1 = self.bottleneck(global_feat)
            feat2 = self.part1_bottleneck(pose_part_feat)
            feat3 = self.lead_bottleneck(lead_feat)

        if self.training:
            cls_score = self.classifier(feat1)
            cls_score_pose = self.part1_classifier(feat2)
            cls_score_lead = self.lead_classifier(feat3)
            return [cls_score, cls_score_pose, cls_score_lead], [global_feat, pose_part_feat, lead_feat]
        else:
            return torch.cat([feat1 + feat2, feat3], dim=1)


