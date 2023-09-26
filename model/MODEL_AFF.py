
import torch
import torch.nn as nn

class MLP12(nn.Module):
    def __init__(self, device):
        super(MLP12, self).__init__()
        
        self.W_g_attention = nn.Parameter(torch.randn(5, 10).to(device), requires_grad=True)
        self.b_g_attention = nn.Parameter(torch.randn(5, 1).to(device), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.t()
        g_attention = self.sigmoid(torch.matmul(self.W_g_attention, x) + self.b_g_attention)
        ones_vector = torch.ones((5, 1), device=x.device)
        xi = x[0:5]
        xi2 = x[5:10]
        x0 = torch.mul(g_attention, xi) + torch.mul((ones_vector - g_attention), xi2)
        x = x0.t()
        return x
# class MLP12(nn.Module):
#     def __init__(self, device):
#         super(MLP12, self).__init__()
        
#         self.W_g_attention = nn.Parameter(torch.randn(5, 10).to(device), requires_grad=True)
#         self.b_g_attention = nn.Parameter(torch.randn(5, 1).to(device), requires_grad=True)

#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         x = x.t()
#         g_attention = self.sigmoid(torch.matmul(self.W_g_attention, x) + self.b_g_attention)
#         ones_vector = torch.ones((5, 1), device=x.device)
#         xi = x[0:5]
#         xi2 = x[5:10]
#         x0 = torch.mul(g_attention, xi) + torch.mul((ones_vector - g_attention), xi2)
#         x = x0.t()
#         return x