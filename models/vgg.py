# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torch.nn as nn
import torch.nn.functional as F

from models.fp_modules import *
from models.binary_modules import *
from models.ternary_modules import *

from torch.nn.functional import relu, max_pool2d, tanh, avg_pool2d, hardtanh, pad
class VGGNet7_binary(nn.Module):
    def __init__(self, args):
        super(VGGNet7_binary, self).__init__()

        # define channel scale factor depending on dataset to be trained on
        if args.dataset == 'cifar10':
            a = 1 * args.width_scale
        elif args.dataset == 'svhn':
            a = 0.5 * args.width_scale
        else:
            raise AttributeError("VGGNet type " + args.type + " unknown or not implemented yet. Use cifar or svhn.")

        self.tern_acts = self._to_bits(args.tern_acts)
        tern_weights = self._to_bits(args.tern_weights)

        if not args.l1_fp:
            self.conv1 = BinaryConv2d(3, int(a*128), (3,3), bias=not args.no_biases, args=args) if(not tern_weights[0]) else TernaryConv2d(3, int(a*128), (3,3), bias=not args.no_biases, args=args)
        else:
            self.conv1 = nn.Conv2d(3, int(a*128), (3,3))
        self.conv2 = BinaryConv2d_pad(int(a*128), int(a*128), (3,3), bias=not args.no_biases, args=args) if(not tern_weights[1]) else TernaryConv2d_pad(int(a*128), int(a*128), (3,3), bias=not args.no_biases, args=args)
        self.conv3 = BinaryConv2d_pad(int(a*128), int(a*256), (3,3), bias=not args.no_biases, args=args) if(not tern_weights[2]) else TernaryConv2d_pad(int(a*128), int(a*256), (3,3), bias=not args.no_biases, args=args)
        self.conv4 = BinaryConv2d_pad(int(a*256), int(a*256), (3,3), bias=not args.no_biases, args=args) if(not tern_weights[3]) else TernaryConv2d_pad(int(a*256), int(a*256), (3,3), bias=not args.no_biases, args=args)
        self.conv5 = BinaryConv2d_pad(int(a*256), int(a*512), (3,3), bias=not args.no_biases, args=args) if(not tern_weights[4]) else TernaryConv2d_pad(int(a*256), int(a*512), (3,3), bias=not args.no_biases, args=args)
        self.conv6 = BinaryConv2d_pad(int(a*512), int(a*512), (3,3), bias=not args.no_biases, args=args) if(not tern_weights[5]) else TernaryConv2d_pad(int(a*512), int(a*512), (3,3), bias=not args.no_biases, args=args)
        self.fc1   = BinaryLinear(int(a*512*3*3), 1024, bias=not args.no_biases, args=args) if(not tern_weights[6]) else TernaryLinear(int(a*512*3*3), 1024, bias=not args.no_biases, args=args)
        self.fc2   = BinaryLinear(1024, 1024, bias=not args.no_biases, args=args) if(not tern_weights[7]) else TernaryLinear(1024, 1024, bias=not args.no_biases, args=args)
        if not args.ln_fp:
            self.fc3   = BinaryLinear(1024, 10, bias=not args.no_biases, args=args) if(not tern_weights[8]) else TernaryLinear(1024, 10, bias=not args.no_biases, args=args)
        else:
            self.fc3   =  nn.Linear(1024, 10)

        self.bn1 = nn.BatchNorm2d(int(a*128))
        self.bn2 = nn.BatchNorm2d(int(a*128))
        self.bn3 = nn.BatchNorm2d(int(a*256))
        self.bn4 = nn.BatchNorm2d(int(a*256))
        self.bn5 = nn.BatchNorm2d(int(a*512))
        self.bn6 = nn.BatchNorm2d(int(a*512))
        self.bn7 = nn.BatchNorm1d(1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.bn9 = nn.BatchNorm1d(10, affine=False)

        self.ste = args.gradient_estimator
        if self.ste == "vanilla":
            self.ste_forward = lambda x: x # anonymous function that returns unchanged input (aka "do_nothing()")
        elif self.ste == "bnn":
            self.ste_forward = hardtanh
        else:
            raise AttributeError("Straight-through estimator " + self.ste + " unknown or not implemented yet. Use vanilla or bnn.")

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        self.layer_order = args.layer_order
        if self.layer_order != "CBAP" and self.layer_order != "CPBA":
            raise AttributeError("Layer-order " + self.layer_order + " unknown or not implemented yet. Use CBAP or CPBA.")

        self.cuda = args.cuda

    def forward(self, x):
        if self.layer_order == "CBAP":
            return self.forward_cbap(x)
        elif self.layer_order == "CPBA":
            return self.forward_cpba(x)

    def forward_cbap(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[0] else ternarize(x.data, "tbn")

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[1] else ternarize(x.data, "tbn")
        x = max_pool2d(x, (2,2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[2] else ternarize(x.data, "tbn")

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[3] else ternarize(x.data, "tbn")
        x = max_pool2d(x, (2,2))

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[4] else ternarize(x.data, "tbn")

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[5] else ternarize(x.data, "tbn")
        x = max_pool2d(x, (2,2))

        x = x.view(-1, num_flat_features(x))

        x = self.fc1(x)
        x = self.bn7(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[6] else ternarize(x.data, "tbn")

        x = self.fc2(x)
        x = self.bn8(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[7] else ternarize(x.data, "tbn")

        x = self.fc3(x)
        x = self.bn9(x)
        return x

    def forward_cpba(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[0] else ternarize(x.data, "tbn")

        x = self.conv2(x)
        x = max_pool2d(x, (2,2))
        x = self.bn2(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[1] else ternarize(x.data, "tbn")

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[2] else ternarize(x.data, "tbn")

        x = self.conv4(x)
        x = max_pool2d(x, (2,2))
        x = self.bn4(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[3] else ternarize(x.data, "tbn")

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[4] else ternarize(x.data, "tbn")

        x = self.conv6(x)
        x = max_pool2d(x, (2,2))
        x = self.bn6(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[5] else ternarize(x.data, "tbn")

        x = x.view(-1, num_flat_features(x))

        x = self.fc1(x)
        x = self.bn7(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[6] else ternarize(x.data, "tbn")

        x = self.fc2(x)
        x = self.bn8(x)
        x = self.ste_forward(x)
        x.data = binarize(x.data) if not self.tern_acts[7] else ternarize(x.data, "tbn")

        x = self.fc3(x)
        x = self.bn9(x)
        return x

    def forward_xnor(self, x):
        pass
    def _to_bits(self, string):
        return [s=='1' for s in string]



class VGGNet7(nn.Module):
    def __init__(self, type):
        super(VGGNet7, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, (3,3))
        self.conv2 = Conv2d_pad(128, 128, (3,3))
        self.conv3 = Conv2d_pad(128, 256, (3,3))
        self.conv4 = Conv2d_pad(256, 256, (3,3))
        self.conv5 = Conv2d_pad(256, 512, (3,3))
        self.conv6 = Conv2d_pad(512, 512, (3,3))
        self.fc1   = nn.Linear(512*3*3, 1024)
        self.fc2   = nn.Linear(1024, 1024)
        self.fc3   = nn.Linear(1024, 10)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm1d(1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.bn9 = nn.BatchNorm1d(10, affine=False)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.conv5.weight)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = x.view(-1, num_flat_features(x))

        x = self.fc1(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn9(x)
        return x


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s

    return num_features
