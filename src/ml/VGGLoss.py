import torch.nn as nn
from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self, layers=8):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = self.vgg[:layers + 1]
        self.vgg.eval()
        self.vgg.requires_grad_(False)
    
    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        return nn.MSELoss()(input_vgg, target_vgg)