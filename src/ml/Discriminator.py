from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, LeakyReLU, Sigmoid
from torchsummary import summary

class Discriminator(Module):
    def __init__(self, device, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.device = device

        self.main = Sequential(
            # input size 3 x 256 x 256
            Conv2d(3, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),

            # hidden_dim x 128 x 128
            Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(hidden_dim * 2),
            LeakyReLU(0.2, inplace=True),

            # (hidden_dim*2) x 64 x 64
            Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(hidden_dim * 4),
            LeakyReLU(0.2, inplace=True),

            # (hidden_dim*4) x 32 x 32
            Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(hidden_dim * 8),
            LeakyReLU(0.2, inplace=True),

            # (hidden_dim*8) x 16 x 16
            Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(hidden_dim * 8),
            LeakyReLU(0.2, inplace=True),

            # (hidden_dim*8) x 8 x 8
            Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(hidden_dim * 8),
            LeakyReLU(0.2, inplace=True),

            # (hidden_dim*8) x 4 x 4
            Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)
            # 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input)
    
    def summary(self):
        return summary(self.to(self.device), (3, 256, 256))