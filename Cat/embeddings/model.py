from torch import nn
import torch as th
import losses

def main():
    pass



class CatEmbeddings(nn.Module):
    def __init__(self):
        super(CatEmbeddings, self).__init__()
        self.neg_loss = losses.neg_loss
        self.proj_loss = losses.proj_loss
        self.inj_loss = losses.inj_loss

    def forward(self, input):
        neg_ax, proj_ax, inj_ax = input

        loss1 = self.neg_loss(neg_ax))
        loss2 = self.proj_loss(proj_ax)
        loss3 = self.inj_loss(inj_ax)

        return loss1 + loss2 + loss3


