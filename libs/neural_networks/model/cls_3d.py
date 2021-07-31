
'''
based on https://github.com/MrGiovanni/ModelsGenesis/blob/master/pytorch/unet3d.py
The following changes were made.
1.remove a major bug:
replaced ContBatchNorm3d by nn.BatchNorm3d() because ContBatchNorm3d performed wrongly during reference.

2. add a class upsample_size

3. remove U-NET shortcart.

4. add drop out support

'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class _LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(_LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = _LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = _LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)
    else:
        layer1 = _LUConv(in_channel, 32 * (2 ** depth), act)
        layer2 = _LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act)

    return nn.Sequential(layer1, layer2)


class _DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(_DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out


class _OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(_OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out



class Cls_3d(nn.Module):

    def __init__(self, n_class=1, act='relu', dropout_prob=0):
        super(Cls_3d, self).__init__()

        self.down_tr64 = _DownTransition(1, 0, act)
        self.down_tr128 = _DownTransition(64, 1, act)
        self.down_tr256 = _DownTransition(128, 2, act)
        self.down_tr512 = _DownTransition(256, 3, act)

        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        self.out64 = self.down_tr64(x)
        self.out128 = self.down_tr128(self.out64)
        self.out256 = self.down_tr256(self.out128)
        self.out512 = self.down_tr512(self.out256)

        self.out_glb_avg_pool = F.avg_pool3d(self.out512, kernel_size=self.out512.size()[2:]).view(self.out512.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        if self.dropout_prob > 0:
            self.linear_out = self.dropout(self.linear_out)
        final_out = self.dense_2(F.relu(self.linear_out))
        return final_out

