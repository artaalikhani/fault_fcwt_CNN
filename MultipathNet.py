import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)
    
class MPBlock(nn.Module):
    def __init__(self, Channel_h = 16):
        super().__init__()
        self.b2 = ConvBlock(Channel_h, Channel_h)
        self.b3 = ConvBlock(Channel_h, Channel_h)
        self.b4 = ConvBlock(Channel_h, Channel_h)
    def forward (self, x):
        f2 = self.b2(x)
        
        f3 = self.b3(f2)
        
        f4 = self.b4(f3)

        f5 = torch.cat((f2,f3,f4),1)
        
        return f5

class Net(nn.Module):
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.name = 'Multi-path Net'
        self.device = torch.device("cpu")
        self.bn = nn.BatchNorm2d(in_channels)
        self.b1 = ConvBlock(in_channels, 8, kernel_size=(5,17), stride=(4,16))
        
        self.mp = MPBlock(8)
        
        self.b6 = ConvBlock(24, 8, kernel_size=(9,9))
        self.fc= nn.LazyLinear(n_class)
        self.flatten = nn.Flatten()
        
        

    def forward(self, x):
        #x = x.to(self.device).type(torch.DoubleTensor)
        f0 = self.bn(x)
        f1 = self.b1(x)

        f5 = self.mp(f1)
        
        f6 = self.b6(f5)

        f7 = self.flatten(f6)
        out=self.fc(f7)
        return out
