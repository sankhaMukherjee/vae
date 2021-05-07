import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras import Model, layers


class ConvEncoder(layers.Layer):

    def __init__(self, nInpX, nInpY, nInpCh, nLatent, nFilters, kernelSizes, strideSizes, activations, paddings, name='encoder'):

        super(ConvEncoder, self).__init__(name=name)

        # Figure input features
        self.nInpX   = nInpX
        self.nInpY   = nInpY
        self.nInpCh  = nInpCh

        convSpecs    = zip(nFilters, kernelSizes, strideSizes, activations, paddings)
        self.convs   = [ Conv2D( filters=f, kernel_size=k, strides=(s, s), padding=p, activation=a) for f, k, s, a, p in convSpecs ]
        self.flat    = Flatten()

        self.mean    = Dense(nLatent)
        self.logVar  = Dense(nLatent)

    def call(self, inputs):

        # Go through the Convolution layers
        x = inputs * 1
        for conv in self.convs:
            x = conv(x)
        
        x = self.flat(x)

        # Create the latent layer (z)
        zMean   = self.mean(x)
        zLogVar = self.logVar(x)
        epsilon = tf.random.normal( shape=zMean.shape, mean=0, stddev=1 )
        z       = zMean + tf.exp( 0.5 * zLogVar )*epsilon

        return zMean, zLogVar, z

    def describe(self, inputs):

        print(f'+--------------- [Encoder Details Start] -----------------')

        print(f'| Shape of the input: {inputs.numpy().shape}')

        # Go through the Convolution layers
        x = inputs * 1
        for i, conv in enumerate(self.convs):
            x = conv(x)
            print(f'| Shape after the {i:4d}tn convolution: {x.numpy().shape}')
        
        x = self.flat(x)
        print(f'| Shape after flattening: {x.numpy().shape}')

        # Create the latent layer (z)
        zMean   = self.mean(x)
        zLogVar = self.logVar(x)
        epsilon = tf.random.normal( shape=zMean.shape, mean=0, stddev=1 )
        z       = zMean + tf.exp( 0.5 * zLogVar )*epsilon

        print(f'| Shape of the latent space: {x.numpy().shape}')

        print(f'+--------------- [Encoder Details End] -----------------')
        


        return


class ConvDecoder(layers.Layer):

    def __init__(self, nInpX, nInpY, nInpCh, nLatent, nFilters, kernelSizes, strideSizes, activations, paddings, name='decoder'):

        super(ConvDecoder, self).__init__(name=name)

        self.resize  = Dense(nLatent*nLatent*nInpCh)
        self.reshape = Reshape((nLatent, nLatent, nInpCh), input_shape=( nLatent*nLatent*nInpCh, ))

        deconvSpecs  = zip(nFilters, kernelSizes, strideSizes, activations, paddings)
        self.deconvs = [ Conv2DTranspose( filters=f, kernel_size=k, strides=(s, s), padding=p, activation=a) for f, k, s, a, p in deconvSpecs ]

        self.resize  = layers.experimental.preprocessing.Resizing( nInpY, nInpX )
        self.flat    = Flatten()
        self.result  = Dense( nInpX*nInpY*nInpCh, activation=None )
        

    def call(self, inputs):

        x = inputs * 1

        # First convert it into a square image
        x = self.resize( x )
        x = self.reshape( x )

        # Go through the deconvolution layers
        for deconv in self.deconvs:
            x = deconv(x)
        
        # flatten it and convert it into something that can be used
        x = self.resize(x)
        x = self.flat(x)
        
        result = self.result( x )

        return result

    def describe(self, inputs):
        '''describe the shapes of the different layers once created

        This will be useful in creating the best estimation of the type of encoders
        and decoders that can be created for a convolutional encoder and decoder.
        '''

        
        # First convert it into a square image
        x = self.resize( x )
        x = self.reshape( x )

        # Go through the deconvolution layers
        for deconv in self.deconvs:
            x = deconv(x)
        
        # flatten it and convert it into something that can be used
        x = self.resize(x)
        x = self.flat(x)
        
        result = self.result( x )


        return

class ConvVAE(Model):

    def __init__(self, nInpX, nInpY, nInpCh, nLatent, encoderSpecs, decoderSpecs, lr = 1e-3, name='vae'):
        
        super(ConvVAE, self).__init__(name=name)
        
        self.nInpX   = nInpX
        self.nInpY   = nInpY
        self.nInpCh  = nInpCh
        self.nLatent = nLatent

        self.encoderSpecs = encoderSpecs
        slef.decoderSpecs = decoderSpecs

        self.encoder = ConvEncoder(nInpX, nInpY, nInpCh, nLatent, **encoderSpecs)
        self.decoder = ConvDecoder(nInpX, nInpY, nInpCh, nLatent, **decoderSpecs)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    def call(self, inputs):

        zMean, zLogVar, z = self.encoder(inputs)
        reconstructed     = self.decoder(z)
        
        return reconstructed

    def step(self, x):

        with tf.GradientTape() as tape:

            zMean, zLogVar, z = self.encoder(x)
            xHat = self.decoder( z )

            # Reconstruction Loss
            reconLoss = tf.nn.sigmoid_cross_entropy_with_logits( x, xHat )
            reconLoss = tf.reduce_sum( reconLoss, 1 )
            reconLoss = tf.reduce_mean( reconLoss )
            reconLoss = reconLoss

            # KL - divergence loss
            klLoss    = - 0.5 * tf.reduce_sum(zLogVar - tf.square(zMean) - tf.exp(zLogVar) + 1, 1)
            klLoss    = tf.reduce_mean( klLoss )
            klLoss    = klLoss

            # Calculate the total loss
            loss      = reconLoss + klLoss

            # Optimize
            grads     = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return reconLoss.numpy(), klLoss.numpy(), loss.numpy()

