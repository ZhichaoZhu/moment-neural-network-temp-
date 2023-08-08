# Quick Start

The codes in this directory aims for reproducing the results in the paper entitled "Convergence timing in spiking neural network indicates prediction uncertainty"

* To train MLP model for MNIST, please first copy the files, *nips_mnist_training.py*, *nips_mnist.yaml* to the root directory and run the following command 
```
    python nips_mnist_training.py --config=./nist_mnist.yaml
```
* For ResNet20 on CIFAR100, the codes can be found at the authors repository: [ANN2SNN_SRP](https://github.com/hzc1208/ANN2SNN_SRP)
* For ResNet34 and VGG16BN on ImageNet, the codes can be found at the authros repository: [SNN_Calibration](https://github.com/yhhhli/SNN_Calibration)

Due to the License issue, the code for running experiments on LOIHI is not available currently.

