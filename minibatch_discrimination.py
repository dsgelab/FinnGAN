import torch
import torch.nn as nn

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features # A
        self.out_features = out_features # B
        self.kernel_dims = kernel_dims # C
        self.mean = mean
        self.T = nn.utils.spectral_norm(nn.Linear(in_features, out_features * kernel_dims))

    def forward(self, x):
        # x is NxDxE (D*E = A)
        # T is AxBxC
        matrices = self.T(x.contiguous().view(x.shape[0], -1))#.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        o_b = o_b.unsqueeze(1).repeat(1, x.shape[1], 1) # NxDxB
            
        x = torch.cat([x, o_b], -1) # NxDx(E+B)
        return x