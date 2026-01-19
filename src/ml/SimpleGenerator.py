from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, Tanh

class SimpleGenerator(Module):
    def __init__(self, device):
        super(SimpleGenerator, self).__init__()
        num_channels = 3
        ngf = 64
        self.main = Sequential(
            # state size 3x256x256
            Conv2d(num_channels, ngf, kernel_size=5, padding="same", bias=False),
            BatchNorm2d(ngf),
            ReLU(True),
            # state size ngf x 256 x 256
            Conv2d(ngf, ngf, kernel_size=3, padding="same", bias=False),
            BatchNorm2d(ngf),
            ReLU(True),
            # state size ngf x 256 x 256
            Conv2d(ngf, ngf, kernel_size=3, padding="same", bias=False),
            BatchNorm2d(ngf),
            ReLU(True),
            # state size ngf x 256 x 256
            Conv2d(ngf, ngf, kernel_size=3, padding="same", bias=False),
            BatchNorm2d(ngf),
            ReLU(True),
            # state size ngf x 256 x 256
            Conv2d(ngf, num_channels, kernel_size=3, padding="same", bias=False),
            Tanh()
            # state size 3 x 256 x 256
        )
    
    def forward(self, input):
        return self.main(input)