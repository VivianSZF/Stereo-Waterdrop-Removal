from torch.nn import init
import torch
from torchvision import models
from collections import namedtuple


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std



class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = Vgg16(requires_grad=False).cuda()

    def forward(self, output, gt, batch_size, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        gt = normalize_batch(gt)
        output = normalize_batch(output)
        features_in = self.vgg(gt)
        features_out = self.vgg(output)
        l1_loss = torch.nn.L1Loss().cuda()
        content_loss = 0
        if 'relu1_2' in layers:
            content_loss += l1_loss(features_in.relu1_2, features_out.relu1_2)
        if 'relu2_2' in layers:
            content_loss += l1_loss(features_in.relu2_2, features_out.relu2_2)*0.5
        if 'relu3_3' in layers:
            content_loss += l1_loss(features_in.relu3_3, features_out.relu3_3)*0.4
        if 'relu4_3' in layers:
            content_loss += l1_loss(features_in.relu4_3, features_out.relu4_3)
        return content_loss



class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out




