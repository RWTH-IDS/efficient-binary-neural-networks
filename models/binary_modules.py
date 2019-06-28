# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torch
import torch.nn as nn
from torch.nn import functional as F

class BinaryLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        args = kwargs["args"]
        kwargs.pop("args", None)

        # In stochastic binarization, w is binarized to +1 with probability p=f(w)
        # and to -1 with probability 1-p. Two prob distributions are implemented:
        # - sigmoid (f(w) = 1/(1+e^(-w)))
        # - hard sigmoid (f(w)=max(0, min(1, (w+1)/2))) = clamp((w+1)/2, 0, 1).
        # To achieve the binarization, the corresponding term ((w+1)/2 or 1/(1+e^(-w))
        # is added to a random number drawn from a uniform dist in [-0.5, 0.5),
        # rounded to the nearest integer (only 0 or 1 due to the choice of f(w)),
        # and scaled to be -1 or +1.
        
        super(BinaryLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = binarize(self.weight.org)

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        args = kwargs["args"]
        kwargs.pop("args", None)

        super(BinaryConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org = self.weight.data.clone()

        self.weight.data = binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class BinaryConv2d_pad(nn.Conv2d):
    """ Binary convolution with SAME-padding behaviour like in tensorflow

    Copied from: https://github.com/mlperf/inference/blob/master/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40
    """

    def __init__(self, *kargs, **kwargs):
        args = kwargs["args"]
        kwargs.pop("args", None)

        super(BinaryConv2d_pad, self).__init__(*kargs, **kwargs)

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

        self.weight.data = binarize(self.weight.org)

        # calculate and apply padding
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd], mode='replicate')

        input = F.pad(input, [padding_rows//2, padding_rows//2, padding_cols//2, padding_cols//2], mode='replicate')
        # actual convolution
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def binarize(tensor):
    tensor_o = tensor.clone() # implicitly unhooks tensor from gradient computation
    tensor_o = tensor_o.add_(1).div_(2).clamp_(0,1) # [-1,1] -> [0,1]

    tensor_o.round_()

    tensor_o.mul_(2).add_(-1)
    return tensor_o
