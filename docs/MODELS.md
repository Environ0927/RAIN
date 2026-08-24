# Frozen model definitions

- `lr`: one affine layer over the flattened input (7,850 parameters for
  MNIST/Fashion-MNIST with 10 classes).
- `SimpleCNN1C`: 32/64-channel 3x3 convolutions, one 2x2 max-pool, and
  256/10 fully connected layers (3,232,906 parameters).
  The same frozen body with a 62-class FEMNIST head has 3,246,270 parameters.
- `FashionNet`: Conv(1,32,3)-BN-ReLU-Pool, then
  Conv(32,64,3)-BN-ReLU-Pool, flatten, FC(3136,128)-ReLU-Dropout(0.2), and
  FC(128,10). Convolution and linear weights use Kaiming-normal initialization;
  biases are zero. The model has 421,738 parameters.
- `resnet18-cifar`: BasicBlock counts `[2,2,2,2]`, 64/128/256/512 channels,
  a 3x3 stride-one stem, no initial max-pool, adaptive average pooling, and a
  10-class head. It has 11,173,962 parameters.
- `resnet34-small`: BasicBlock counts `[3,4,6,3]` with the same small-image
  stem and a configurable head. The CIFAR-100 instance has 21,328,292 parameters.
- `resnet50-small`: Bottleneck counts `[3,4,6,3]`, small-image stem, and a
  configurable head. The Tiny-ImageNet instance has 23,910,152 parameters.

All parameter counts include trainable normalization and bias parameters and
were checked by the model smoke test. Config names are frozen in `configs/`.
