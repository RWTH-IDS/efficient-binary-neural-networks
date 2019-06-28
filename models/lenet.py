# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torch.nn as nn
from torch.nn.functional import relu, max_pool2d, tanh, avg_pool2d, hardtanh, pad

from models.binary_modules import *
from models.ternary_modules import *

class LeNet5_binary(nn.Module):
    def __init__(self, args):
        super(LeNet5_binary, self).__init__()

        self.width_infl = int(args.width_infl)

        self.apply_relu = args.apply_xnor_net
        self.last_fc_input_fullprecision = args.apply_xnor_net
        self.apply_last_bn = not args.apply_xnor_net
        self.apply_xnor_layer = args.apply_xnor_net

        self.tern_acts = self._to_bits(args.tern_acts)
        tern_weights = self._to_bits(args.tern_weights)

        padding = 0 if args.binarize_input else 2
        if args.first_weights_fullprecision:
            self.conv1 = nn.Conv2d(1, self.width_infl*32, (5,5), padding=padding, bias=not args.no_biases)
        else:
            self.conv1 = BinaryConv2d(1, self.width_infl*32, (5,5), padding=padding, bias=not args.no_biases, args=args) if(not tern_weights[0]) else TernaryConv2d(1, self.width_infl*32, (5,5), padding=padding, bias=not args.no_biases, args=args)

        self.conv2 = BinaryConv2d(self.width_infl*32, self.width_infl*64, (5,5), bias=not args.no_biases, args=args) if(not tern_weights[1]) else TernaryConv2d(self.width_infl*32, self.width_infl*64, (5,5), bias=not args.no_biases, args=args)
        self.fc1   = BinaryLinear(self.width_infl*64*5*5, 512, bias=not args.no_biases, args=args) if(not tern_weights[2]) else TernaryLinear(self.width_infl*64*5*5, 512, bias=not args.no_biases, args=args)

        if args.last_weights_fullprecision:
            self.fc2 = nn.Linear(512, 10, bias=not args.no_biases)
        else:
            self.fc2   = BinaryLinear(512, 10, bias=not args.no_biases, args=args) if(not tern_weights[3]) else TernaryLinear(512, 10, bias=not args.no_biases, args=args)

        self.batch_norm = not args.no_batchnorm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(self.width_infl*32)
            self.bn2 = nn.BatchNorm2d(self.width_infl*64)
            self.bn3 = nn.BatchNorm1d(512)
            if self.apply_last_bn: self.bn4 = nn.BatchNorm1d(10, affine=False)
            if self.apply_xnor_layer: self.bn5 = nn.BatchNorm2d(self.width_infl*32)

        self.ste = args.gradient_estimator
        if self.ste == "vanilla":
            self.ste_forward = lambda x: x # anonymous function that returns unchanged input (aka "do_nothing()")
        elif self.ste == "bnn":
            self.ste_forward = hardtanh
        elif self.ste == "bireal":
            self.ste_forward = self._bireal_ste
        elif self.ste == "swish":
            self.ste_forward = self._swish_ste
            self.swish_beta = args.swish_beta
        else:
            raise AttributeError("Straight-through estimator " + self.ste + " unknown or not implemented yet. Use vanilla, bnn, bireal or swish.")

        self.layer_order = args.layer_order
        if self.layer_order != "CBAP" and self.layer_order != "CPBA":
            raise AttributeError("Layer-order " + self.layer_order + " unknown or not implemented yet. Use CBAP or CPBA.")

        self.binarize_input = args.binarize_input
        self.binarize_input_threshold = args.binarize_input_threshold

        self.cuda = args.cuda

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        if self.binarize_input:
            x.data = binarize(x.data, thresh=self.binarize_input_threshold)
            x.data = pad(x.data, (2,2,2,2), value=-1)

        if self.layer_order == "CBAP":
            x = self.conv1(x)
            if self.apply_relu: x = relu(x)
            if self.batch_norm: x = self.bn1(x)
            x = self.ste_forward(x)
            x.data = binarize(x.data) if not self.tern_acts[0] else ternarize(x.data, "tbn")
            x = max_pool2d(x, (2,2))

            x = self.conv2(x)
            if self.apply_relu: x = relu(x)
            if self.batch_norm: x = self.bn2(x)
            x = self.ste_forward(x)
            x.data = binarize(x.data) if not self.tern_acts[1] else ternarize(x.data, "tbn")
            x = max_pool2d(x, (2,2))
        else:
            x = self.conv1(x)
            if self.apply_xnor_layer:
                x = self.bn5(x)
                x = relu(x)

            x = max_pool2d(x, (2,2))
            if self.batch_norm: x = self.bn1(x)
            x = self.ste_forward(x)
            x.data = binarize(x.data) if not self.tern_acts[0] else ternarize(x.data, "tbn")

            x = self.conv2(x)
            if self.apply_relu: x = relu(x)
            x = max_pool2d(x, (2,2))
            if self.batch_norm: x = self.bn2(x)
            x = self.ste_forward(x)
            x.data = binarize(x.data) if not self.tern_acts[1] else ternarize(x.data, "tbn")

        x = x.view(-1, num_flat_features(x))

        x = self.fc1(x)
        if self.apply_relu and not self.last_fc_input_fullprecision: x = relu(x)
        if self.batch_norm: x = self.bn3(x)

        if self.last_fc_input_fullprecision:
            x = relu(x)
        else:
            x = self.ste_forward(x)
            x.data = binarize(x.data) if not self.tern_acts[2] else ternarize(x.data, "tbn")

        x = self.fc2(x)
        if self.apply_last_bn: x = self.bn4(x)

        return x

    def _bireal_ste(self, x):
        # Implementation of piecewise polynomial STE from Liu et al, "Bi-real Net"
        x.clamp_(-1,1)

        a_r = x[x.data.abs()!=1]

        sign_m = (2*(a_r.data<0).type(torch.FloatTensor if not self.cuda else torch.cuda.FloatTensor)-1) # contains -1 where x_i>0 and +1 where_ x_i<0

        x[x.data.abs()!=1] = 2*a_r + sign_m*(a_r**2)
        return x

    def _swish_ste(self, x):
        # Implementation of swish-like STE following the equations from
        # Darabi et al, "BNN+"
        temp = x.data

        b = self.swish_beta
        x = 2*torch.sigmoid(b * x)*(1 + b*x*(1-torch.sigmoid(b * x))) - 1

        # Trick: inputs close to zero (~1e-9) get mapped to zero by the swish
        #        function above due to rounding error. Zeros get binarized to +1.
        #        This might introduce bigger error later on. Avoid that by
        #        setting zero-values back to their pre-swish values
        x.data[x.data==0] = temp[x.data==0]
        return x

    def _to_bits(self, string):
        return [s=='1' for s in string]


class LeNet5(nn.Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()
        # conv and fc layer definition
        self.conv1 = nn.Conv2d(1, 32, (5,5), padding=2)
        self.conv2 = nn.Conv2d(32, 64, (5,5))
        self.fc1   = nn.Linear(64*5*5, 512)
        self.fc2   = nn.Linear(512, 10)

        # batch-norm layer definition
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(10, affine=False)

        # randomly initialize weights
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)
        x = max_pool2d(x, (2,2))
        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)
        x = max_pool2d(x, (2,2))

        x = x.view(-1, num_flat_features(x)) # reshape last conv layer outputs for fc layers

        x = self.fc1(x)
        x = self.bn3(x)
        x = relu(x)
        x = self.fc2(x)
        x = self.bn4(x)
        return x


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s

    return num_features

