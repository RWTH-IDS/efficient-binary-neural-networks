## Efficient Binary Neural Networks

This repository demonstrates how to train binary & ternary neural networks using best 
practices and approaches aimed at preserving their efficiency. We provide pretrained models
of binarized CNNs for MNIST and CIFAR-10, reaching 99.16% and 89.85% accuracy respectively. 
The related techniques to train these models from scratch are reviewed and introduced in our paper: 

`Stadtmann, Tim, Cecilia Latotzke, and Tobias Gemmeke. "From quantitative analysis to synthesis of efficient binary neural networks." 2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 2020.`

Please cite the above publication in case you use this repository or the pre-trained models for your own research.

**Installation**

```
git clone git@github.com:RWTH-IDS/efficient-binary-neural-networks.git
cd efficient-binary-neural-networks
pip install -r requirements

# optional: download datasets for training and testing
cd data
python download.py mnist
python download.py cifar10
```

**Usage**

For both MNIST and CIFAR10, we provide pretrained models in binary and full-precision.


| architecture  | dataset |  precision | accuracy |
| ------------  | ------- |  --------- | -------- |
| LeNet5        | mnist   |  float     | 99.50    |
| LeNet5        | mnist   |  binary    | 99.16    |
| VGGNet7       | cifar10 |  float     | 94.05    |
| VGGNet7       | cifar10 |  binary    | 89.85    |


Run inference on the test sets with pretrained binary networks as follows:

```
# MNIST binary inference
python main.py ID --inference --params config/binary_mnist.yml --model-path pretrained/LeNet5/binary
# CIFAR10 binary inference
python main.py ID --inference --params config/binary_cifar10.yml --model-path pretrained/VGGNet7/binary
```

You can train these networks from scratch as follows:

```
# MNIST training
python main.py binary_mnist_from_scratch --params config/binary_mnist.yml
# CIFAR10 training
python main.py binary_cifar_from_scratch --params config/binary_cifar10.yml
```

Similar workflow for inference on pretrained floating point networks:

```
# MNIST float inference
python main.py ID --inference --params config/ref_mnist.yml --model-path pretrained/LeNet5/float
# CIFAR10 float inference
python main.py ID --inference --params config/ref_cifar10.yml --model-path pretrained/VGGNet7/float
```

The script **main.py** runs a training process based on a number of options. The 
default options enable a full-precision training on MNIST using the LeNet5 CNN architecture: `python main.py ID`.
This will start a training over 500 epochs on MNIST. The only necessary option is a string of your choice that 
acts as an ID which will be used to save the training data (chosen 
hyperparameters, initial network weights, accuracies at each epoch, ...) in a 
unique folder. 

Use `python main.py --help` for information on all possible options. Presets for these options can be given in yml-files, with some examples provided in the `config` folder. The parameters in these config files can be overwritten individually, e.g.: `python main.py CIFAR10_test --params config/ref_cifar10.yml --lr 0.2`

In the folder `experiments`, you will find two exemplary training scripts. Scripts in this style were used to produce all results in the 
our publication.
