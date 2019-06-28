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
            0)  python3 main.py CIFAR10_PTRELU --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --resume --model "pt_relu.pt" ;;
            1)  python3 main.py CIFAR10_REG0_00001 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --regularization bin --weight-decay 0.00001;;
            2)  python3 main.py CIFAR10_REG0_000001 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --regularization bin --weight-decay 0.000001;;
            3)  python3 main.py CIFAR10_REG0_0000001 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --regularization bin --weight-decay 0.0000005;;
            4)  python3 main.py CIFAR10_REG0_0000005 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --regularization bin --weight-decay 0.0000001;;
            5)  python3 main.py CIFAR10_REG0_00000001 --params config/config_binary_cifar10_larq_closer.yml --seed $seed --undeterministic --regularization bin --weight-decay 0.00000001;;
        esac
    done
done
