#!/bin/bash

# Copyright (c) IDS, RWTH Aachen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Last Modified : Jan. 22, 2024, T. Stadtmann

cd ..

N_SEEDS=5

exp_ids=( "$@" ) # Get input arguments as array
for ((seed=1; seed<=N_SEEDS; seed++)); do # Repeat each experiment N_SEEDS times
    for exp_id in "${exp_ids[@]}"; do # Conduct all experiments given as IDs
        case "$exp_id" in
            0)  python3 main.py CIFAR10_L1FP --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --l1-fp;;
            1)  python3 main.py CIFAR10_LNFP --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --ln-fp;;
            2)  python3 main.py CIFAR10_L1NFP --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --l1-fp --ln-fp;;
            3)  python3 main.py CIFAR10_L1T --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 00000000 --tern-weights 100000000;;
            4)  python3 main.py CIFAR10_LNT --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 00000000 --tern-weights 000000001;;
            5)  python3 main.py CIFAR10_L1NT --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 00000000 --tern-weights 100000001;;
            6)  python3 main.py CIFAR10_TERN1 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 11111111 --tern-weights 000000000;;
            7)  python3 main.py CIFAR10_TERN2 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 10000001 --tern-weights 000000000;;
            8)  python3 main.py CIFAR10_TERN3 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 11000011 --tern-weights 000000000;;
            9)  python3 main.py CIFAR10_TERN4 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 10000001 --tern-weights 100000001;;
            10)  python3 main.py CIFAR10_TERN5 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --tern-acts 11111111 --tern-weights 111111111;;
            11)  python3 main.py CIFAR10_WIDEN2 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --width-scale 2;;
            12)  python3 main.py CIFAR10_WIDEN4 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --width-scale 4;;
            13)  python3 main.py MNIST_TERN1 --params config/config_binary_mnist_opt.yml --seed $seed --undeterministic --tern-acts 101 --tern-weights 1001;;
            14)  python3 main.py MNIST_L1T --params config/config_binary_mnist_opt.yml --seed $seed --undeterministic --tern-acts 000 --tern-weights 1000;;
            15)  python3 main.py MNIST_LNT --params config/config_binary_mnist_opt.yml --seed $seed --undeterministic --tern-acts 000 --tern-weights 0001;;
            16)  python3 main.py MNIST_L1NT --params config/config_binary_mnist_opt.yml --seed $seed --undeterministic --tern-acts 000 --tern-weights 1001;;
        esac
    done
done
