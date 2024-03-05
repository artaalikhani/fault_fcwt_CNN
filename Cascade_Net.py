import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.functional import Tensor
import typing



class ConvLSTMCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=1):
        super(ConvLSTMCell, self).__init__()
        self.intermediate_channels = intermediate_channels
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels *  4,
            kernel_size=kernel_size, padding= (kernel_size // 2)*2,dilation=2, bias=True,
        )
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> typing.Tuple:

        c, h = state
        h = h.to(device=x.device)
        c = c.to(device=x.device)
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        g = torch.sigmoid(g)
        d = torch.tanh(d)
        c =  a * c +  g * d
        h = b * torch.tanh(c)
        return c, h



class RegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return F.relu(out)
    
class CascadeBlock(nn.Module):
    def __init__(self, Channel_h = 16):
        super(CascadeBlock, self).__init__()
        self.b2 = RegBlock(Channel_h, Channel_h)
        self.b3 = RegBlock(Channel_h, Channel_h)
        self.b4 = RegBlock(Channel_h, Channel_h)
    def forward (self, x):
        f2 = self.b2(x)
        
        f3 = self.b3(f2)
        
        f4 = self.b4(f3)

        c4 = torch.cat((x,f2,f3,f4),1)
        
        return c4

class Net(nn.Module):
    def __init__(self, in_channels, n_class):
        super().__init__()
        self.name = 'Cascade_Net'
        self.device = torch.device("cpu")
        self.b0 = nn.BatchNorm2d(in_channels)
        self.b1 = RegBlock(in_channels, 4, kernel_size=(5,17), stride=(4,16))
        
        self.c1 = CascadeBlock(4)
        
        
        self.b6 = RegBlock(16, 16, kernel_size=(9,9))
        self.fc= nn.LazyLinear(n_class)
        self.flatten = nn.Flatten()
        
        

    def forward(self, x):
        #x = x.to(self.device).type(torch.DoubleTensor)
        f0 = self.b0(x)
        f1 = self.b1(f0)
        #f1 = F.relu(f1)
        #f2 = self.pool(f1)
        #f3 = self.b2(f2)
        c4 = self.c1(f1)
        
        f6 = self.b6(c4)
        f6 = F.relu(f6)
        f8 = self.flatten(f6)
        out=self.fc(f8)
        return out
