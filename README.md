# Simple Diffusion Models
Building simple diffusion models for image generation. More so for understanding and learning.

## Simple script to train and evaluate a tiny U-Net model on CIFAR-10 and MNIST.

### Train Command
```
python diffusion_tinyunet_mnist_cifar10.py --dataset "mnist" --batch_size 64 --epochs 100 --lr 1e-3 --log_interval 100
```

### Test Command
```
python diffusion_tinyunet_mnist_cifar10.py --dataset "mnist" --inference --num_samples 16
```

### Random Qualitative Results on MNIST

| ![](samples/mnist_step_1.png) | ![](samples/mnist_step_2.png) | ![](samples/mnist_step_3.png) | ![](samples/mnist_step_4.png) | ![](samples/mnist_step_5.png) |
|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| ![](samples/mnist_step_6.png) | ![](samples/mnist_step_7.png) | ![](samples/mnist_step_8.png) | ![](samples/mnist_step_9.png) | ![](samples/mnist_step_10.png) |


Easy peasy lemon squeezey &#x1F34B;
