from torch import nn
from torch.nn import functional as F
import torch
# 96--256--8*32--4*32--8*32--256--5
class SpatialGatingUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        #空间交互，输入输出4个通道
        self.spatial_proj = nn.Conv1d(4, 4, kernel_size=5,padding=2)#
        nn.init.constant_(self.spatial_proj.bias, 1.0)
    def forward(self, x):
        u, v = x.chunk(2, dim=-2)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out
class gMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        #对通道进行
        self.channel_proj1 = nn.Linear(8,8)  # List of linear layers
        self.channel_proj2 = nn.Linear(4,8)
        self.sgu = SpatialGatingUnit()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        #交换维度，对8个通道进行全连接做channel_proj
        x = F.gelu(self.channel_proj1(x))
        #交换回来
        x = x.permute(0, 2, 1)
        x = self.sgu(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.channel_proj2(x))
        x = x.permute(0, 2, 1)
        out = x + residual
        return out
class gMLP(nn.Module):
    def __init__(self, in_features, out_features, mlp_features,num_blocks):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock() for _ in range(num_blocks)]
        )
        self.numchannel = 8#8 channels
        self.channel_mapping = nn.Linear(in_features, mlp_features)  # Mapping from 96 to d_model dimensions
        self.outfc=nn.Linear(mlp_features,out_features)
    def forward(self, x):
        # Split 96-dimensional input into 8 channels with 32 dimensions each
        x = self.channel_mapping(x)
        x = x.view(x.size(0), self.numchannel , 32)
        x=self.model(x)
        #全连接输出
        x=x.reshape((x.size(0),x.size(1)*x.size(2)))
        x=self.outfc(x);
        return x
if __name__ == "__main__":

    x = torch.rand(1, 5)
    modelfthe = gMLP(in_features=5, out_features=96, mlp_features=256, num_blocks=3)
    # modelfthe = SpatialGatingUnit()
    y = modelfthe(x)
    print("Output shape: ", y)