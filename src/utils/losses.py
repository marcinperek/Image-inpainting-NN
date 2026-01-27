import torchvision.models as models
import torch

class VGGLoss(torch.nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:9]
        self.vgg = vgg.eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.criterion = torch.nn.MSELoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = self.criterion(input_features, target_features)
        return loss
        