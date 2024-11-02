from torch.nn import Parameter
from fusion import *
from torch import nn
from torch.nn import functional as F
from config_all.config_lgg import HP


class LinearNet(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(LinearNet, self).__init__()
        self.Encoder1 = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(),
                                      nn.Linear(hidden_size, hidden_size), nn.Tanh())

    def forward(self, x):
        out = self.Encoder1(x)
        return out


def BFusion(input1, input2):
    fusion = torch.mul(input1, input2)
    fusion_sum = fusion.view(fusion.size(0), 100, 1)
    fusion_sum = fusion_sum.sum(2)
    fusion_sum = torch.sqrt(F.relu(fusion_sum)) - torch.sqrt(F.relu(-fusion_sum))
    fusion_sum = F.normalize(fusion_sum, p=2)
    return fusion_sum



class BaseAttLayer(nn.Module):

    def __init__(self):
        super(BaseAttLayer, self).__init__()

        self.gene_encoder = LinearNet(HP.LN_inputdim, HP.LN_outputdim)
        self.cnv_encoder = LinearNet(HP.LN_inputdim, HP.LN_outputdim)

        self.linear_gc_att = nn.Sequential(nn.Linear(HP.LA_inputdim, HP.LA_outputdim))
        self.linear_att_muatn = nn.Linear(HP.Lam_inputdim, HP.Lam_outputdim)
        # self.linear_att_mlb = nn.Linear(HP.Laml_inputdim, HP.Laml_outputdim)

        self.linear_sam = nn.Linear(HP.Lm_inputdim, HP.Lm_outputdim)

        self.list_linear_1_fusion = None
        self.linear_2_fusion = None
        self.linear_classif = None
        self.norm = None

        self.self_attention = nn.Sequential(nn.Linear(HP.SelfAtt_inputdim, HP.SelfAtt_outputdim),
                                            nn.Sigmoid()
                                            )

        self.output_range = Parameter(torch.FloatTensor([4]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-2]), requires_grad=False)


    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def _attention(self, input1, input2):
        batch_size = input1.size(0)


        x1 = self.linear_gc_att(input1)
        x1 = x1.view(batch_size, 1, HP.LA_outputdim)

        x2 = self.linear_gc_att(input2)
        x2 = x2.view(batch_size, 1, HP.LA_outputdim)

        x_att_12 = self._fusion_att(x1, x2)

        x_att_12 = self.linear_att_muatn(x_att_12)

        x_att_12 = x_att_12.view(batch_size, HP.Lam_outputdim, 1)

        list_att_12_split = torch.split(x_att_12, 1, dim=1)

        list_att_12 = []
        for x_att in list_att_12_split:
            x_att = x_att.contiguous()

            x_att = x_att.view(batch_size, 1)
            x_att = F.softmax(x_att, dim=1)
            list_att_12.append(x_att)

        x_1 = input1.view(batch_size, 1, HP.LN_outputdim)

        list_1_att = []
        for i, x_att_12 in enumerate(list_att_12):

            x_att_12 = x_att_12.view(batch_size, 1, 1)

            x_att_12 = x_att_12.expand(batch_size, 1, HP.LN_outputdim)

            x_1_att = torch.mul(x_att_12, x_1)

            x_1_att = x_1_att.sum(1)
            list_1_att.append(x_1_att)

        return list_1_att

    def _fusion_glimpses(self, list_1_att, x2):

        list_1 = []
        for glimpse_id, x_1_att in enumerate(list_1_att):

            x_1_1 = self.list_linear_1_fusion[glimpse_id](x_1_att)
            list_1.append(x_1_1)

        x_1 = torch.cat(list_1, 1)


        x_1 = self.linear_sam(x_1)
        x_2 = self.linear_2_fusion(x2)


        x_12 = self._fusion_classif(x_1, x_2)
        return x_12

    def _classif(self, x):
        x1 = self.linear_classif(x)
        out = torch.sigmoid(x1)
        out = out * self.output_range + self.output_shift
        return out

    def forward(self, input_gene, input_cnv):
        if input_gene.dim() != 2 and input_cnv.dim() != 2:
            raise ValueError

        x_gene = self.gene_encoder(input_gene)

        x_cnv = self.cnv_encoder(input_cnv)

        list_gene_att = self._attention(x_gene, x_cnv)
        list_cnv_att = self._attention(x_cnv, x_gene)
        bimodal_gene_cnv = self._fusion_glimpses(list_gene_att, x_cnv)
        bimodal_cnv_gene = self._fusion_glimpses(list_cnv_att, x_gene)

        gene_self = BFusion(x_gene, x_gene)
        cnv_self = BFusion(x_cnv, x_cnv)
        inter_gene = torch.cat((x_gene, gene_self), 1)
        inter_cnv = torch.cat((x_cnv, cnv_self), 1)
        gene_weight = self.self_attention(inter_gene)
        cnv_weight = self.self_attention(inter_cnv)

        gene_weight_ex = (gene_weight.expand(x_gene.size(0), HP.SelfAtt_inputdim))
        cnv_weight_ex = (cnv_weight.expand(x_cnv.size(0), HP.SelfAtt_inputdim))
        inter_gene_w = gene_weight_ex * inter_gene
        inter_cnv_w = cnv_weight_ex * inter_cnv
        self_gene_cnv = inter_gene_w + inter_cnv_w
        x = torch.cat((bimodal_gene_cnv, bimodal_cnv_gene, self_gene_cnv), dim=1)

        x = self._classif(x)
        return x

class MtdfAtt(BaseAttLayer):

    def __init__(self):
        super(MtdfAtt, self).__init__()

        self.fusion_att = MtFusion2()
        self.list_linear_1_fusion = nn.ModuleList(
            [nn.Linear(HP.LL1f_inputdim, HP.LL1f_outputdim)
             for i in range(HP.R)]
        )
        self.linear_2_fusion = nn.Linear(HP.L2f_inputdim, HP.L2f_outputdim)
        self.linear_classif = nn.Sequential(nn.Linear(HP.Lcls_inputdim, HP.Lcls_hiddendim))
        self.fusion_classif = MtFusion()
        self.norm = nn.BatchNorm1d(HP.Lcls_inputdim)

    def _fusion_att(self, x_v, x_q):
        return self.fusion_att(x_v, x_q)

    def _fusion_classif(self, x_v, x_q):
        return self.fusion_classif(x_v, x_q)



if __name__ == '__main__':
    import torch

    x1 = torch.randn(10, 80).to('cuda')
    x2 = torch.randn(10, 80).to('cuda')
    model = MtdfAtt().to('cuda')
    print(model)
    y = model(x1, x2)
    print(y.shape)
