# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torch
import torch.nn as nn
from torch.nn import functional as F

class TernaryLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        args = kwargs["args"]
        kwargs.pop("args", None)

        super(TernaryLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = ternarize(self.weight.org, "twn")

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class TernaryConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        args = kwargs["args"]
        kwargs.pop("args", None)

        super(TernaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = ternarize(self.weight.org, "twn")

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class TernaryConv2d_pad(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        args = kwargs["args"]
        kwargs.pop("args", None)

        super(TernaryConv2d_pad, self).__init__(*kargs, **kwargs)

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = ternarize(self.weight.org, "twn")

        # calculate and apply padding
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd], mode='replicate')

        input = F.pad(input, [padding_rows//2, padding_rows//2, padding_cols//2, padding_cols//2], mode='replicate')

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
def ternarize(tensor, thresh_type):
    ternary = tensor.clone()
    thresh = ternary.data.abs().sum() / ternary.data.numel()

    if thresh_type == 'twn':
        thresh *= 0.7
    elif thresh_type == 'tbn':
        thresh *= 0.4
    else:
        print("warning: unknown threshold "+thresh_type)

    ternary[ternary>thresh] = 1
    ternary[ternary<-thresh] = -1
    ternary[ternary.data.abs()<=thresh] = 0

    return ternary
