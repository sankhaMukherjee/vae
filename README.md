# VAE

Overview of different types of autoencoders.

In this repository, we shall create a number of different types ofautoencoders. Some of the autoencoders are written in 
TensorFlow, and some in pyTorch. Care has been taken to make sure that the modelsare easy to understand rather than whether
they are efficient or accurate. Also most of thiscode does not follow good software engineering practices whatsoever. This 
repo is not intended tobe production quality code. This is expected to be experimental software, that can form the basis for
rapid prototyping and experimentation.

Of special note is the fact that none of this code uses any form of regularization, batch normalization, and the like. Neither
does this code contain any information for saving models, creating checkpoints, loading from checkpoints, etc. If you wish to
use any of these features, you will need to add them yourself.


# Examples

The following examples are present


|         command        | model | backend | data | comments |
|------------------------|-------|---------|------|----------|
|`python3 testTFvae.py`  | [TF/VAE.py](https://github.com/sankhaMukherjee/vae/blob/master/models/TF/VAE.py) | TensorFlow | MNIST | Both the encoder and the decoder are `Dense` layers. Reconstruction is simply based upon a `sigmoid_cross_entropy_with_logits`. MNIST digits are unraveled into a 784 dimensional vector. |
|`python3 testTFcvae.py`  | [TF/CVAE.py](https://github.com/sankhaMukherjee/vae/blob/master/models/TF/CVAE.py) | TensorFlow | MNIST | Conditional variational autoencoder. Both the encoder and the decoder are `Dense` layers. Reconstruction is simply based upon a `sigmoid_cross_entropy_with_logits`. MNIST digits are unraveled into a 784 dimensional vector. |
|`python3 testTFcoercevae.py`  | [TF/coerceVAE.py](https://github.com/sankhaMukherjee/vae/blob/master/models/TF/coerceVAE.py) | TensorFlow | MNIST | coerced variational autoencoder. Both the encoder and the decoder are `Dense` layers. Reconstruction is simply based upon a `sigmoid_cross_entropy_with_logits`. MNIST digits are unraveled into a 784 dimensional vector. In this variation, there is some coersion while creating the latent space so that there is greater separation between members of the group that are known to be in different groups. |
|`python3 testTFConvVAE.py`  | [TF/ConvVAE.py](https://github.com/sankhaMukherjee/vae/blob/master/models/TF/ConvVAE.py) | TensorFlow | MNIST | Convolutional variaitonal autoencoder. Instead of assuming that the image is based upon a flattened representation, this method simply uses a set of convolution layers as part of the encoder and the decoder. |

# Requirements

The current version is written with the following configuration:

 - `CudaToolkit 11.0`
 - `cuDNN 8.`
 - `TensorFlow 2.4.1`
 - `torch 1.8.0+cu11`

The code has been tested on a GPU with the following configuration: 

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 2070    Off  | 00000000:01:00.0  On |                  N/A |
|  0%   47C    P8    21W / 175W |   1456MiB /  7979MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

For some reason, the current version of tensorflow overflows in memory usage and
errors out for RTX 2070 seres. For that reason, you will need to add the following
lines to your TensorFlow code to prevent that from happening.

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
```

## Authors

Sankha S. Mukherjee - Initial work (2021)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
