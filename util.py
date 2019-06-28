# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from os import path


def load_raw_data(dataset_str, data_path):
    # check if path exists
    if not path.isdir(data_path):
        raise AttributeError("Data path " + data_path + " does not exist or cannot be accessed.")

    # append / to path (pytorch is picky about that)
    if not data_path.endswith('/'):
        data_path = data_path + "/"

    # load data
    if dataset_str == "mnist":
        if not path.isdir(data_path+'MNIST'):
            data_path = path.split(path.split(data_path)[0])[0]  # user maybe gave something like "data/MNIST" -> need only "data/"
            if not path.isdir(data_path):
                raise AttributeError("Data path " + data_path + " does not exist or cannot be accessed. Please give folder containing MNIST folder.")

        transf = [transforms.ToTensor()] # transforms images to pytorch tensors in range [0,1]

        try:
            train_all = datasets.MNIST(root=data_path, train=True, download=False, transform=transforms.Compose(transf))
            test_all = datasets.MNIST(root=data_path, train=False, download=False, transform=transforms.Compose(transf))
        except RuntimeError as err:
            print("Did you forget to download the dataset? -> data/download.py")
            raise err
       
        # Split trainset into train (first 50k images) and validation data (last 10k images)
        trainset = torch.utils.data.Subset(train_all, range(50000))
        valset = torch.utils.data.Subset(train_all, range(50000,60000))

        return trainset, valset, test_all
    elif dataset_str == "cifar10":
        if path.isdir(data_path+'CIFAR10'):
            data_path = data_path+"CIFAR10/"

        transf = [transforms.RandomHorizontalFlip(), # randomly mirrors 50% of the images during loading
                transforms.RandomCrop(32, padding=4, padding_mode='edge'), # pads images with 4 pixels and randomly crops them during loading
                transforms.ToTensor(), # transforms images to pytorch tensors in range [0,1]
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))] # transforms data to [-1,1]
        transf_test = [transforms.ToTensor(), # transforms images to pytorch tensors in range [0,1]
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))] # transforms data to [-1,1]
        try:
            train_all = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transforms.Compose(transf))
            test_all = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms.Compose(transf_test))
        except RuntimeError as err:
            print("Did you forget to download the dataset? -> data/download.py")
            raise err

        # Split trainset into train (first 45k images) and validation data (last 5k images)
        trainset = torch.utils.data.Subset(train_all, range(45000))
        valset = torch.utils.data.Subset(train_all, range(45000,50000))

        return trainset, valset, test_all        
    else:
        raise AttributeError("Dataset " + dataset_str + " unknown. Use mnist or svhn.")


class SquareHingeLoss(nn.Module):
    def __init__(self, device, num_outputs=10): 
        super(SquareHingeLoss, self).__init__()
        self.device = device
        self.num_outputs = num_outputs

    def forward(self, input, target):
        target = torch.eye(self.num_outputs, device=self.device)[target] # one-hot the target vector
        target = 2*target - 1 # convert (0,1) to (-1,1)

        output = 1-input*target
        output[output.le(0)] = 0 # equals: max(0, output)
        return torch.mean(torch.sum(output*output, dim=1))


def reg_term(net, type):
    sum = 0
    for p in list(net.parameters()):
        if hasattr(p, 'org'):
            # Horrible implementation incoming
            # -> In order for loss.backward() to track the regularization loss,
            #    we need to calculate it using the current parameters of the
            #    network (as they are tracked themselves). At this stage, they
            #    are binary - but we need to calculate the loss using the
            #    full-precision weights. Current workaround: copy ALL
            #    full-precision weights into the networks parameters (which are
            #    saved in a temp-tensor), then calculate loss, then copy the
            #    back the binary parameters. :(
            temp = torch.Tensor(p.data.shape)
            temp.copy_(p.data)
            p.data.copy_(p.org)
            sum += (1 - p ** 2).sum()
            p.org.copy_(p.data)
            p.data.copy_(temp)
    return sum
