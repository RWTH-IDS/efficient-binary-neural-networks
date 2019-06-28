# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torchvision
import sys

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    raise RuntimeError("Missing argument: name of dataset (mnist, cifar10)")

if dataset == "mnist":
    torchvision.datasets.MNIST(root="./", train=True, download=True)
    torchvision.datasets.MNIST(root="./", train=False, download=True)
elif dataset == "cifar10":
    torchvision.datasets.CIFAR10(root="CIFAR10/", train=True, download=True)
    torchvision.datasets.CIFAR10(root="CIFAR10", train=False, download=True)
