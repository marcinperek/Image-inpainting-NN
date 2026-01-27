import torch
import torch.nn as nn
from torchsummary import summary


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GatedConv, self).__init__()
        self.conv_A= nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_B= nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        A = self.conv_A(x)
        B = self.conv_B(x)
        B = self.sigmoid(B)
        return A * B
    

class GatedDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale_factor=2):
        super(GatedDeConv, self).__init__()
        self.conv = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scale_factor = scale_factor
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv(x)
        return x


class ContextualAttention(nn.Module):
    def __init__(self, in_dim):
        super(ContextualAttention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out
    

class CoarseNet(nn.Module):
    def __init__(self, cnum=48):
        super(CoarseNet, self).__init__()

        self.model = nn.Sequential(
            GatedConv(4, cnum, 5, 1, 2),
            nn.ELU(),
            GatedConv(cnum, 2*cnum, 3, 2, 1),
            nn.ELU(),
            GatedConv(2*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(2*cnum, 4*cnum, 3, 2, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            
            GatedConv(4*cnum, 4*cnum, 3, 1, 2, dilation=2),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 4, dilation=4),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 8, dilation=8),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 16, dilation=16),
            nn.ELU(),
            GatedDeConv(4*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(2*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedDeConv(2*cnum, cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(cnum, cnum//2, 3, 1, 1),
            nn.ELU(),
            GatedConv(cnum//2, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)
    

class RefinementNet(nn.Module):
    def __init__(self, cnum=48):
        super(RefinementNet, self).__init__()

        self.conv_branch = nn.Sequential(
            GatedConv(3, cnum, 5, 1, 2),
            nn.ELU(),
            GatedConv(cnum, 2*cnum, 3, 2, 1),
            nn.ELU(),
            GatedConv(2*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(2*cnum, 4*cnum, 3, 2, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 2, 2),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 4, 4),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 8, 8),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 16, 16),
            nn.ELU()
        )

        self.attention_branch = nn.Sequential(
            GatedConv(3, cnum, 5, 1, 2),
            nn.ELU(),
            GatedConv(cnum, 2*cnum, 3, 2, 1),
            nn.ELU(),
            GatedConv(2*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(2*cnum, 4*cnum, 3, 2, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ReLU(),
            ContextualAttention(4*cnum),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU()
        )

        self.decoder = nn.Sequential(
            GatedConv(8*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(4*cnum, 4*cnum, 3, 1, 1),
            nn.ELU(),
            GatedDeConv(4*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(2*cnum, 2*cnum, 3, 1, 1),
            nn.ELU(),
            GatedDeConv(2*cnum, cnum, 3, 1, 1),
            nn.ELU(),
            GatedConv(cnum, cnum//2, 3, 1, 1),
            nn.ELU(),
            GatedConv(cnum//2, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x_conv = self.conv_branch(x)
        x_attn = self.attention_branch(x)
        x = torch.cat([x_conv, x_attn], dim=1)
        x = self.decoder(x)
        return x
    

class DeepFill(nn.Module):
    def __init__(self, device, cnum=48):
        super(DeepFill, self).__init__()
        self.device = device
        self.coarse_net = CoarseNet(cnum)
        self.refinement_net = RefinementNet(cnum)
    
    def summary(self, input_size=(4, 256, 256)):
        summary(self.to(self.device), input_size=input_size)

    def forward(self, x):
        mask = x[:, 3, ...].unsqueeze(1)
        img = x[:, :3, ...]
        coarse_output = self.coarse_net(x)
        refined_input = coarse_output * mask + img * (1 - mask)
        refined_output = self.refinement_net(refined_input)
        return coarse_output, refined_output