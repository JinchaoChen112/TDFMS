import torch
from torch.utils.data.dataset import Dataset


class Dataset_loader(Dataset):

    def __init__(self, data, split):
        self.x_gene = data[split]['gene']
        self.x_cnv = data[split]['cnv']
        self.censor = data[split]['censor']
        self.survtime = data[split]['survtime']
        self.id_ = data[split]['id']

    def __getitem__(self, index):
        censor = torch.tensor(self.censor[index]).type(torch.FloatTensor)
        survtime = torch.tensor(self.survtime[index]).type(torch.FloatTensor)
        gene = torch.tensor(self.x_gene[index]).type(torch.FloatTensor)
        cnv = torch.tensor(self.x_cnv[index]).type(torch.FloatTensor)
        id_ = self.id_[index]

        return gene, cnv, censor, survtime, id_

    def __len__(self):
        return len(self.censor)
