import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from torch.nn.parameter import Parameter



class SoftAtt(nn.Module):
    def __init__(self):
        super(SoftAtt, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        self.pred = nn.Conv2d(64*4, 2,kernel_size=3, padding=1)
        self.Soft = Soft()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self,Out_six_rater, fea, out):

        fea = self.conv_block(fea)

        pred_cup_list = []
        pred_disc_list = []

        for i in range(len(Out_six_rater)):
            pred_disc_list.append(torch.sigmoid(Out_six_rater[i][:,0,:,:]).unsqueeze(1))
            pred_cup_list.append(torch.sigmoid(Out_six_rater[i][:,1,:, :]).unsqueeze(1))    #[b, 1, 256, 256]

        # calculate uncertainty
        pred_cup = torch.cat(pred_cup_list,dim=1)
        pred_disc = torch.cat(pred_disc_list,dim=1)

        uncertianty_cup = torch.std(pred_cup, dim=1).unsqueeze(1)
        uncertianty_disc = torch.std(pred_disc,dim=1).unsqueeze(1)      # [b,1,256,256]

        Att_disc = torch.sigmoid(out[:, 0, :, :].unsqueeze(1))
        Att_cup = torch.sigmoid(out[:, 1, :, :].unsqueeze(1))

        # Soft
        soft_u_cup_fea = self.Soft(uncertianty_cup, fea) + fea
        soft_u_disc_fea = self.Soft(uncertianty_disc,fea)+ fea
        soft_cup_fea = self.Soft(Att_cup, fea) + fea
        soft_disc_fea = self.Soft(Att_disc, fea) + fea

        fea_enhanced = torch.cat([soft_cup_fea,soft_disc_fea,soft_u_cup_fea,soft_u_disc_fea],dim=1)
        out = self.pred(fea_enhanced)

        return out, [Att_disc,Att_cup,uncertianty_cup,uncertianty_disc]


# Soft Attention
def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class Soft(nn.Module):
    # holistic attention module
    def __init__(self):
        super(Soft, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x