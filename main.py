# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import argparse
from datetime import datetime
import yaml
import warnings
#from operator import itemgetter
from copy import deepcopy
import random
from progress.bar import Bar
from models.lenet import *
from models.vgg import *
import warnings
import yaml
from util import SquareHingeLoss
from util import reg_term
from util import load_raw_data


def main():

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Binary Neural Network - Experiment')
    parser.add_argument('id', type=str,
                        help='unique id for saving the results')
    parser.add_argument('--params', type=str, default='config/ref_mnist.yml',
                        help='path to config file of parameters')
    ## Usual hyperparameters
    parser.add_argument('--epochs', type=int,
                        help='number of training epochs (default: 500)')
    parser.add_argument('--results', type=str,
                        help='folder to output the results to (default: results)')
    parser.add_argument('--dataset', type=str,
                        help='dataset to train on (default: mnist; others: cifar10')
    parser.add_argument('--datapath', type=str,
                        help='path to folder containing dataset (default: data)')
    parser.add_argument('--inference', action='store_true',
                        help="perform inference on pretrained model; given via --model-path")
    parser.add_argument('--model-path', dest='model_path', type=str,
                        default='', help="path to pretrained model for weights"
                        " initialization and to further train"
                        " (params.pt/result.pt required, default: empty)")
    parser.add_argument('--net', type=str,
                        help='cnn network architecture (default: LeNet5; other: VGGNet7')
    parser.add_argument('--lr', type=float,
                        help='learning rate (default: 0.01) (decayed in epochs 15,25)')
    parser.add_argument('--momentum', type=float,
                        help='SGD momentum (default: 0.9')
    parser.add_argument('--optimizer', type=str,
                        help='choose optimizer, see readme (default: SGD)')
    parser.add_argument('--batch-size', type=int,
                        help='training batch size (default: 50)')
    parser.add_argument('--weight-decay', type=float,
                        help='L2 regularization lambda term (default: 0.0001)')
    parser.add_argument('--lr-milestones', nargs='*',
                        help='epochs at which learn rate should be decayed (default: 15,25)')
    parser.add_argument('--lr-decay', nargs='*',
                        help='learn rate decay vector (default: [0.1])')
    parser.add_argument('--loss', type=str,
                        help='loss function type (default: SquareHingeLoss; others: CrossEntropyLoss)')
    parser.add_argument('--regularization', type=str,
                        help='type of regularization (default: L2; others: bin)')
    parser.add_argument('--num-workers', type=int,
                        help='proccess limit of multiprocessing, see readme (default: 4)')
    parser.add_argument('--undeterministic', action='store_true',
                        help='activates undeterministic CUDA functions for better performance')
    ## Binarization
    parser.add_argument('--binarize', action='store_true', 
                        help='enable binarization')
    parser.add_argument('--gradient-estimator', type=str, 
                        help='straight-through estimator of the binarize() gradient ' +
                             '(default: vanilla; others: bnn, bireal, swish)')
    parser.add_argument('--swish-beta', type=float, 
                        help='beta parameter for swish-like gradient estimator (default: 10)')
    parser.add_argument('--layer-order', type=str, 
                        help='order of conv, batch-norm, activation and pool layers (default: CBAP; others: CPBA)')
    ## CNN architecture
    parser.add_argument('--no-biases', action='store_true',
                        help='disable biases in conv and fc layers')
    parser.add_argument('--no-batchnorm', action='store_true', 
                        help='disables batch normalization')
    parser.add_argument('--activation', type=str, 
                        help='activation function of full precision model (default: relu; others: clip)')
    parser.add_argument('--width-infl', type=int, 
                        help='scaling factor for the LeNet-5 network width (default: 1)')
    ## System and output
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training ')
    parser.add_argument('--seed', type=int,
                        help='random seed (default: 1)')
    parser.add_argument('--gpu-ids', nargs='*',
                        help='gpus used for training (default: [0])')
    ## Dataset
    parser.add_argument('--binarize-input', action='store_true',
                        help='binarize network inputs')
    parser.add_argument('--binarize-input-threshold', type=float,
                        help='threshold for binarizing inputs (default: 0.5)')
    ## XNOR-Net experiments
    parser.add_argument('--apply-xnor-net', action='store_true',
                        help='apply relu directly on convolution outputs in bnn')
    parser.add_argument('--first-weights-fullprecision', action='store_true',
                        help='let weights of first layer in full precision')
    parser.add_argument('--last-weights-fullprecision', action='store_true',
                        help='let weights of last layer in full precision')
    ## Capacity experiments
    parser.add_argument('--tern-weights', type=str, 
                        help='which vggnet layers'' weights to ternarize (default: 000000000; fake one-hot coding, 1st digit -> layer 1, 2nd digit -> layer 2)')
    parser.add_argument('--tern-acts', type=str, 
                        help='which vggnet layers'' inputs to ternarize (default: 00000000; fake one-hot coding, 1st digit -> layer 1 outputs, 2nd digit -> layer 2 outputs; last layer outputs will not be ternarized)')
    parser.add_argument('--l1-fp', action='store_true', 
                        help='keep first vggnet layer in full-precision')
    parser.add_argument('--ln-fp', action='store_true', 
                        help='keep first vggnet layer in full-precision')
    parser.add_argument('--width-scale', type=int, 
                        help='scale number of vggnet conv feature maps per layer by this factor (default: 1)')

    print("Parsing input arguments...")
    args = parser.parse_args()

    # read YAML file
    with open(args.params, 'r') as stream:
        try:
            yaml_params = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    # loads params.yml into args
    for p in args.__dict__:  
        # if user defined param overrides param from config, save it in args and output it with background color
        if p != 'id' and p != 'params':
            color_prefix, color_postfix = '\x1b[6;30;42m', '\x1b[0m'  # cryptic string extensions for color output when printing
            if args.__dict__[p] is None or args.__dict__[
                p] is False:  # if parameter was not defined script startup (default case), load it from yaml and dont color it
                color_prefix, color_postfix = '', ''
                try:
                    args.__dict__[p] = yaml_params[p]
                except KeyError as e:
                    print("Couldn't find key in yaml file: " + str(e))
                    # raise
            else:
                yaml_params[p] = args.__dict__[p]   # overrides params.yml with user input
            output = '\t{0:35}' + color_prefix + '{1}' + color_postfix
        else:
            output = '\t{0:35}{1}'

        print(output.format(str(p), str(args.__dict__[p])))

    

    if args.dataset == "cifar10":
        if args.net != "VGGNet7":
            raise NotImplementedError("Dataset " + args.dataset + " is not supported by " + args.net + ".")
    elif args.dataset == "mnist":
        if args.net != "LeNet5":
            raise NotImplementedError("Dataset " + args.dataset + " is not supported by " + args.net + ".")


    # Set constants and create results folder
    DATA_PATH = args.datapath
    RESULTS_PATH = (args.results + "/" + args.id
                    + "/run" + datetime.now().strftime("%d%m_%H%M%S"))
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    # Create yaml file with parameters
    with open(RESULTS_PATH + '/params.yml', 'w') as file:
        documents = yaml.dump(yaml_params, file)

    if not args.inference:
        print(" > Results will be saved in %s..." % (RESULTS_PATH))

    # format list-like arguments to contain numbers, not strings. If 0 zero is given, ignore
    args.lr_milestones = list(map(int, args.lr_milestones)) if args.lr_milestones != ['0'] and args.lr_milestones != [
        0] else []
    args.lr_decay = list(map(float, args.lr_decay)) if args.lr_decay != ['0'] and args.lr_decay != [0] else []

    ## Warnings
    if args.regularization == "bin":
        print(
            "!!!!WARNING: implementation for -+1 regularization is EXTREMELY slow (increases time per epoch by 30%)!!!!")

    torch.save(args, RESULTS_PATH + "/params.pt")

    # Check GPU configuration
    if args.gpu_ids and len(args.gpu_ids)>1:
        print("GPU configuration check")
        for gpu_id in args.gpu_ids:
            if args.gpu_ids.count(gpu_id) > 1:
                print("Error: Choose different GPUs, e.g. (0 1) ")
                exit(1)
        else:
            print("GPU configured correctly")
            warnings.warn("Choosing more than one GPU leads to a parallelization through all installed GPUs")

    # Setup CUDA for GPU computations (or not)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        if args.gpu_ids == None:  # If no gpu-ids are given, use '0' as default
            args.gpu_ids = ['0']
        args.n_gpus = min(len(args.gpu_ids), torch.cuda.device_count())
    else:
        args.n_gpus = 0

    # Use first GPU  found or CPU
    device = torch.device("cuda:"+args.gpu_ids[0] if args.cuda else "cpu")

    if args.cuda:
        with warnings.catch_warnings(record=True) as w:  # catch system warning (thrown if device cuda version is < 3.5)
            device_cuda_v = torch.cuda.get_device_capability(
                device)  # check GPU's supported CUDA version (needs to be > 3.5)
        if device_cuda_v[0] < 3 or (device_cuda_v[0] == 3 and device_cuda_v[1] <= 5):
            print("Warning: GPU is of cuda capability %d.%d. PyTorch only supports version > 3.5." % (
            device_cuda_v[0], device_cuda_v[1]))
            device = torch.device("cpu")
            args.cuda = False

    if device.type == "cuda":
        print("Using %d GPU%s..." % (args.n_gpus,
                                     "s" if args.n_gpus != 1 else ""))
        for gpu_id in args.gpu_ids:
            print(" > " + torch.cuda.get_device_name(
                torch.device("cuda:"+gpu_id)))
    else:
        print("Using CPU...")

    # Set seeds and randomizer options for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # These backend options will result in better reproducibility when using GPUs, but 
        # also slower the computation speed
        if args.undeterministic:
            torch.backends.cudnn.deterministic = False  # enable usage of non-deterministic cudnn algorithms
            torch.backends.cudnn.benchmark = True  # enable dynamic search for best cudnn algorithms
        else:
            # According to pytorch.org/docs/stable/notes/randomness.html, these have to be set
            # to get results with higher reproducibility when using GPUs, but also slower the computation speed
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    # Setup CNN
    err_inf = None
    try:
        if args.binarize is True:
            if args.net == "LeNet5":
                cnn = LeNet5_binary
                net = cnn(args).to(device)
            elif args.net == "VGGNet7":
                cnn = VGGNet7_binary
                net = cnn(args).to(device)
        else:
            cnn = globals()[args.net]
            net = cnn(args).to(device)
        if args.inference:
            try:
                net.load_state_dict(torch.load(args.model_path + "/results.pt", map_location=device)['model_best']) 
            except Exception as err:
                err_inf = True
                raise AttributeError("The model path is missing or doesn't exist.")
    except Exception as err:
        if err_inf:
            raise err
        else:
            raise AttributeError("Network architecture " + args.net + " unknown. Use LeNet5 or VGGNet7.")


    # Load dataset and neural network model
    load_start=time.time()

    if args.n_gpus > 1:
        # Distribute model over multiple GPUs
        net = nn.DataParallel(net).cuda()

    # Setup optimizer
    l2_weight_decay = args.weight_decay if args.regularization == 'L2' else 0
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                               weight_decay=l2_weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=l2_weight_decay)

    elif args.optimizer=="RMSProp":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer=="ASGD":
        optimizer = optim.ASGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise AttributeError("optimizer " + args.optimizer + " unknown. Use adam, SGD, RMSProp or ASGD.")

   
    criterion = SquareHingeLoss(device=device) if args.loss == "SquareHingeLoss" else nn.CrossEntropyLoss()


    # Prepare result saving
    results = {'losses': [], 'accuracies': [],
               'model_initial': net.state_dict()}

    trainset, valset, testset = load_raw_data(args.dataset, DATA_PATH)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True)

    print('Loaded datasets in %.3f s'%(time.time()-load_start))

    # Inference
    if args.inference:
        print("Starting inference.....")
        print("### SUMMARY ###")
        validate(net, testloader, device, criterion)
        return

    # Resume training if error_checkpoint.pt exists or start training of fully trained network otherwise train from beginning
    if os.path.isfile(args.model_path + "/error_checkpoint.pt"):
        warnings.warn(
            "!!!!WARNING: resuming training only works if pretrained model was a full precision model. No implementation for resuming binary training yet!!!!")
        open(args.model_path + "/error_checkpoint.pt", 'r')
        checkpoint = torch.load(args.model_path + "/error_checkpoint.pt", map_location=device)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['epoch']
        print("=> loaded model '{}'".format(
            args.model_path + "/error_checkpoint.pt"))
    elif os.path.isfile(args.model_path + "/results.pt"):
        open(args.model_path + "/results.pt", 'r')
        checkpoint = torch.load(args.model_path + "/results.pt", map_location=device)
        net.load_state_dict(checkpoint['model_final'])
        optimizer.load_state_dict(checkpoint['optimizer_final'])
        start_epoch = 1
        print("=> loaded model '{}'".format(
            args.model_path + "/results.pt"))
    else:
            start_epoch = 1 
    
    # Training
    max_val_accuracy = 0
    max_val_accuracy_net = deepcopy(net.state_dict())
    ms_idx, decay_idx = 0, 0
    max_val_accuracy_optimizer = deepcopy(optimizer.state_dict())
    max_val_accuracy_epoch = 0
    last_val_acc = -1
    acc = 0
    t_epoch = []
    t_validation = []

    print("Starting training...")
    t_start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs+1):
            last_val_acc = acc
            if len(args.lr_milestones) > 0 and epoch == args.lr_milestones[ms_idx]:
                decay = args.lr_decay[decay_idx]
                for g in optimizer.param_groups:
                    g['lr'] = decay * g['lr']
                if (len(args.lr_milestones) > ms_idx + 1): ms_idx = ms_idx + 1
                if (len(args.lr_decay) > decay_idx + 1): decay_idx = decay_idx + 1


            # do train step on batch
            t_epoch_start = time.time()
            train_acc = train(net, trainloader, optimizer, device, criterion, args, epoch)
            t_epoch.append(time.time() - t_epoch_start)

            # get validation accuracy
            t_validate_start = time.time()
            acc = validate(net, valloader, device, criterion)
            t_validation.append(time.time() - t_validate_start)


            if acc > max_val_accuracy:
                max_val_accuracy = acc
                del max_val_accuracy_net
                del max_val_accuracy_optimizer
                max_val_accuracy_net = deepcopy(net.state_dict())
                max_val_accuracy_optimizer = deepcopy(optimizer.state_dict())
                max_val_accuracy_epoch = epoch
                print(" > New Maximum!")

            results['accuracies'].append(acc)

    except KeyboardInterrupt as err:
        torch.save({'epoch': epoch, 'model': net.state_dict(), 'opt': optimizer.state_dict(), 'error': str(err)}, RESULTS_PATH + "/error_checkpoint.pt")

        print(err)
    except Exception as err:
        torch.save({'epoch': epoch, 'model': net.state_dict(), 'opt': optimizer.state_dict(), 'error': str(err)}, RESULTS_PATH + "/error_checkpoint.pt")
        print("unknown error:",err)
    else:
        torch.save({'epoch': epoch, 'model': net.state_dict(), 'opt': optimizer.state_dict()}, RESULTS_PATH + "/last_checkpoint.pt")

    train_time = time.time() - t_start
    print("Training done in %.3f s" % train_time)
    print("Evaluating best model on testset...")

    test_net = cnn(args).to(device)
    if args.n_gpus > 1:
        test_net = nn.DataParallel(test_net).cuda()
    test_net.load_state_dict(max_val_accuracy_net)


    test_acc= validate(test_net, testloader, device, criterion)

    # Save results
    results['val_accuracy_best'] = max_val_accuracy
    results['epoch_best'] = max_val_accuracy_epoch
    results['model_best'] = max_val_accuracy_net
    results['optimizer_best'] = max_val_accuracy_optimizer
    # state of model at the last epoch
    results['model_final'] = net.state_dict()
    results['optimizer_final'] = optimizer.state_dict()
    results['epoch'] = epoch
    results['train_time'] = train_time
    results['epoch_times'] = t_epoch
    results['validation_times'] = t_validation

    np.savetxt(RESULTS_PATH + "/accuracies.txt",
               np.array(results['accuracies']))
    np.savetxt(RESULTS_PATH + "/test_accuracy.txt", np.array([test_acc]))
    torch.save(results, RESULTS_PATH + "/results.pt")

    # Output summary
    print("### SUMMARY ###")
    print(" > Results are saved in %s/results.pt" % (RESULTS_PATH))
    print(" > Maximum validation accuracy: %.2f %% in epoch %d"
          % (max_val_accuracy, max_val_accuracy_epoch))
    print(" > Accuracy of best model on testset: %.2f %%" % (test_acc))
    print("###############")

    os.system('echo 'r' "\e[?25h]"')


def train(net, trainloader, optimizer, device, criterion, args, epoch):
    # Set model in train mode (important for batch-normalization)
    bar = Bar('Epoch %d' % epoch, max=len(trainloader), suffix='%(elapsed)ds')
    net.train()

    acc, train_loss, correct, total = 0, 0, 0, 0
    for i, data in enumerate(trainloader):
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # MAIN TRAINING PART
        output = net(images)  # forward pass
        loss = criterion(output.to(device), labels)  # loss computation
        if args.regularization != 'L2':
            loss = loss + args.weight_decay * reg_term(net, args.regularization)
        loss.backward()  # backward pass

        if args.binarize:
            for p in list(net.parameters()):  # Copy full-precision weights into main params
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)

        optimizer.step()  # update set

        if args.binarize:
            for p in list(net.parameters()):
                if hasattr(p, 'org'):
                    p.data.clamp_(-1, 1)
                    p.org.copy_(p.data)  # Clip weights and save them in .org

        train_loss += float(loss.item())

        _, predicted = output.max(1)
        total += labels.size(0)
        correct += (predicted.to(device) == labels).sum().item()

        acc = 100*(correct/total)

        bar.next()
        print(' | Train-Loss: %.3f | Train-Acc: %.3f%% (%d/%d)' % (train_loss/(i+1), 100*(correct/total), correct, total), end="", flush=True)
    print()

    return acc


def validate(net, loader, device, criterion):
    net.eval()

    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, labels = data[0].to(device), data[1].to(device)

            # MAIN VALIDATION PART
            output = net(images)
            loss = criterion(output.to(device), labels)
            #

            val_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += (predicted.to(device) == labels).sum().item()


    acc = 100 * (correct / total)
    print(" > Val-Loss: %.3f | Val-Acc: %.3f%% (%d/%d)" % (val_loss / (i + 1), acc, correct, total), end="", flush=True)
    print()

    return acc


if __name__ == '__main__':
    
    main()