import torch
import torch.nn as nn
import torch.nn.functional as F

from config_all.config_lgg import HP


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)
    assert sum(sizes_list) == dim
    if sizes_list[-1]<0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j-1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list

def get_chunks(x,sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1,begin,s)
        out.append(y)
        begin += s
    return out

class AbstractFusion(nn.Module):

    def __init__(self):
        super(AbstractFusion, self).__init__()

    def forward(self, input_v, input_q):
        raise NotImplementedError


class MtFusion(AbstractFusion):

    def __init__(self):
        super(MtFusion, self).__init__()
        self.list_linear_h1_2d = nn.ModuleList([
            nn.Linear(HP.Lh1_2d_inputdim, HP.Lh1_2d_outputdim)
            for i in range(HP.R)])
        self.list_linear_h2 = nn.ModuleList([
            nn.Linear(HP.Lh2_inputdim, HP.Lh2_outputdim)
            for i in range(HP.R)])

    def forward(self, input_1, input_2):
        if input_1.dim() != input_2.dim() and input_1.dim() != 2:
            raise ValueError

        x_1 = input_1
        x_2 = input_2

        x_mm = []
        for i in range(HP.R):
            x_h1 = self.list_linear_h1_2d[i](x_1)

            x_h2 = self.list_linear_h2[i](x_2)
            x_mm.append(torch.mul(x_h2, x_h1))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1)
        return x_mm


class MtFusion2(MtFusion):

    def __init__(self):
        super(MtFusion2, self).__init__()

    def forward(self, input_1, input_2):
        if input_1.dim() != input_2.dim() and input_1.dim() != 3:
            raise ValueError
        batch_size = input_1.size(0)
        weight_height = input_1.size(1)
        if not input_1.is_contiguous():
            input_1 = input_1.contiguous()
        if not input_2.is_contiguous():
            input_2 = input_2.contiguous()
        x_1 = input_1.view(batch_size * weight_height, HP.LA_outputdim)
        x_2 = input_2.view(batch_size * weight_height, HP.LA_outputdim)
        x_mm = super().forward(x_1, x_2)
        x_mm = x_mm.view(batch_size, weight_height, HP.Lh2_outputdim)
        return x_mm
