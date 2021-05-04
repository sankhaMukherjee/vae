import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers


class coerceVAE_Encoder(layers.Layer):

    def __init__(self, nLatent, layers, activations, name='encoder'):

        assert len(layers) == len(activations), "In the encoder, the number of layers and activations must be the same"
        
        super(coerceVAE_Encoder, self).__init__(name=name)

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.mean    = Dense(nLatent)
        self.logVar  = Dense(nLatent)
        self.coerce  = Dense(1)

    def call(self, inputs):

        
        # Go through the Dense layers
        x = inputs * 1
        for i, dl in enumerate(self.layers):
            x = dl(x)
        
        # Create the coerceing layer
        coerce = self.coerce(x)

        # Create the latent layer (z)
        zMean   = self.mean(x)
        zLogVar = self.logVar(x)
        epsilon = tf.random.normal( shape=zMean.shape, mean=0, stddev=1 )
        z       = zMean + tf.exp( 0.5 * zLogVar )*epsilon

        return zMean, zLogVar, z, coerce

class coerceVAE_Decoder(layers.Layer):

    def __init__(self, nInp, layers, activations, name='encoder'):

        assert len(layers) == len(activations), "In the decoder, the number of layers and activations must be the same"
        
        super(coerceVAE_Decoder, self).__init__(name=name)

        layers      = list(reversed(layers))
        activations = list(reversed(activations))

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.result  = Dense(nInp, activation=None)

    def call(self, inputs):

        # Go through the Dense layers
        x = inputs * 1
        for i, dl in enumerate(self.layers):
            x = dl(x)

        result = self.result(x)

        return result

class coerceVAE(Model):

    def __init__(self, nInp, layers, activations, nLatent, lr = 1e-3, name='coerceVAE'):
        '''[summary]

        Parameters
        ----------
        nInp : [type]
            [description]
        layers : [type]
            [description]
        activations : [type]
            [description]
        nLatent : [type]
            [description]
        concatLayer : [type]
            The concat
        lr : [type], optional
            [description], by default 1e-3
        name : str, optional
            [description], by default 'vae'
        '''

        assert len(layers) == len(activations), "The number of layers and activations must be the same"
        

        super(coerceVAE, self).__init__(name=name)
        
        nLayers = len(layers)

        self.nInp    = nInp
        self.encoder = CVAE_Encoder(nLatent=nLatent, layers=layers, activations=activations)
        self.decoder = CVAE_Decoder(nInp, layers=layers, activations=activations)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    def call(self, inputs, condition):

        zMean, zLogVar, z, coerce = self.encoder(inputs)
        reconstructed             = self.decoder(z)
        
        return reconstructed

    def step(self, x, y):

        with tf.GradientTape() as tape:

            zMean, zLogVar, z, coerce = self.encoder(x)
            xHat = self.decoder(z)

            # Reconstruction Loss
            reconLoss = tf.nn.sigmoid_cross_entropy_with_logits( x, xHat )
            reconLoss = tf.reduce_sum( reconLoss, 1 )
            reconLoss = tf.reduce_mean( reconLoss )

            # KL - divergence loss
            klLoss    = - 0.5 * tf.reduce_sum(zLogVar - tf.square(zMean) - tf.exp(zLogVar) + 1, 1)
            klLoss    = tf.reduce_mean( klLoss )

            # Coerce Loss
            coerceLoss = tf.reduce_mean( (coerce - y)**2 )

            # Calculate the total loss
            loss      = reconLoss + klLoss

            # Optimize
            grads     = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return reconLoss.numpy(), klLoss.numpy(), loss.numpy(), coerceLoss.numpy()

