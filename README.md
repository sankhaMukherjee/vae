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
|`python3 testTemporalVAE.py`  | [TF/TemporalVAE.py](https://github.com/sankhaMukherjee/vae/blob/master/models/TF/TemporalVAE.py) | TensorFlow | Generated Data | An Autoencoder that looks like a Hidden Markov Model (HMM). If the number of states are very high, this might be a good method of handling the matter ![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/temporal.png). Note that its best not to use this as a VAE but as an ordinary AE|
|`python3 testTFsigmaVae.py`  | [TF/sigmaVAE.py](https://github.com/sankhaMukherjee/vae/blob/master/models/TF/sigmaVAE.py) | TensorFlow | MNIST | The simple VAE example updated so that the loss function resembles that of the ??-VAE [1] |


# Example Results

## VAE latent reconstruction with `tanh` activation: 

![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/vae-tanh/00099_LatentReconstruction.png)

## VAE latent reconstruction with `relu` activation: 

![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/vae-relu/00099_LatentReconstruction.png)

## ConvVAE latent reconstruction with `tanh` activation: 

![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/ConvVAE/00099_LatentReconstruction.png)


## Coerce VAE

In this case, we ant to _coerce_ the latent space such that it is easier to discriminate between the different labels
when the label data is available. 

| latent space | reconstruction |
|--------------|----------------|
| ![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/Coerce/00149_LatentSpace.png) | ![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/Coerce/00149_LatentReconstruction.png) |


## C-VAE

| 1 | 2 | 3 |  5 |
|---|---|---|----|
| ![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/CVAE/00199_LatentReconstruction_01.png) | ![img](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/CVAE/00199_LatentReconstruction_02.png) | ![img](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/CVAE/00199_LatentReconstruction_03.png) | ![img](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/CVAE/00199_LatentReconstruction_05.png) | 


## ??-VAE

The ??-VAE is a variant on the ??-VAE in that the parameter ?? is no longer a parameter that needs to be tuned
by hand, but can be learned end-to-end. This follows from the work of Rybkin et al. [1], and is supposed to
yield much better reconstructions in comparison to ??-VAEs. Note that all the VAE's that are shown above have
some form of manual ??-tuning that has been performed at run-time. Compare reconstruction results form the
??-VAE and the ??-VAE below:

| VAE type| reconstruction |
|---------|------------|
|  ??-VAE  | ![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/vae-relu/00099_reconstruction.png) |
|  ??-VAE  | ![image](https://raw.githubusercontent.com/sankhaMukherjee/vae/master/results/sigmaVae/00099_reconstruction.png) |



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

## References

1. [Simple and Effective VAE Training with Calibrated Decoders](https://arxiv.org/pdf/2006.13202.pdf)