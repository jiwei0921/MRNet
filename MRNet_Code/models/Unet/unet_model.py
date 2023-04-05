""" Full assembly of the parts to form the complete network """

from .unet_parts import *

from torch import nn
import torch
from .res_net import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from torch.autograd import Variable


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UNet(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        # super(ResUnet, self).__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101()
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers


        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)


        '''~~~ self definition ~~~'''
        self.up_c1 = nn.ConvTranspose2d(6, 6, 4, stride=2)
        self.up_c2 = nn.ConvTranspose2d(6, 512, 2, stride=2)
        self.conv = nn.Conv2d(6,512,3,padding=1)

        self.CONVLSTMcell = ConvLSTMCell(512, 512)


    def forward(self, x, condition):
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]


        '''~~~ 1: Condition + feature ==> ConvLSTM ~~~'''
        condition = condition.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,8,8)
        condition = self.conv(condition)
        # condition = self.up_c2(self.up_c1(condition.unsqueeze(-1).unsqueeze(-1)))       # x = [b_size, 512, 8, 8]
        for t in range(0, 1):                   # step =1
            state = self.CONVLSTMcell(x, [condition, condition])
        x = state[0]
        '''~~~ 1: ENDs ~~~'''


        '''~~~ 0: Decoder ~~~'''
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)
        '''~~~ 0: ENDs ~~~'''



        '''
        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]
        '''
        return output, fea

    def close(self):
        for sf in self.sfs: sf.remove()




class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

