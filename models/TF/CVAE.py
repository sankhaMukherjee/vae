import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers


class CVAE_Encoder(layers.Layer):

    def __init__(self, nLatent, layers, activations, concatLayer, name='encoder'):

        assert len(layers) == len(activations), "In the encoder, the number of layers and activations must be the same"
        assert concatLayer < len(activations), "Concatination must happen within the layers of the encoder"

        super(CVAE_Encoder, self).__init__(name=name)

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.mean    = Dense(nLatent)
        self.logVar  = Dense(nLatent)
        self.nLayers = len(layers)
        self.concatLayer = concatLayer

    def call(self, inputs, condition):

        
        # Go through the Dense layers
        x = inputs * 1
        for i, dl in enumerate(self.layers):

            if i == self.concatLayer:
                x = dl(x)
                x = tf.concat([x, condition], axis=1)
            else:
                x = dl(x)
        
        # Create the latent layer (z)
        zMean   = self.mean(x)
        zLogVar = self.logVar(x)
        epsilon = tf.random.normal( shape=zMean.shape, mean=0, stddev=1 )
        z       = zMean + tf.exp( 0.5 * zLogVar )*epsilon

        return zMean, zLogVar, z

class CVAE_Decoder(layers.Layer):

    def __init__(self, nInp, layers, activations, concatLayer, name='encoder'):

        assert len(layers) == len(activations), "In the decoder, the number of layers and activations must be the same"
        assert concatLayer < len(activations), "Concatination must happen within the layers of the decoder"

        super(CVAE_Decoder, self).__init__(name=name)

        layers      = list(reversed(layers))
        activations = list(reversed(activations))

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.result  = Dense(nInp, activation=None)
        self.nLayers = len(layers)
        self.concatLayer = concatLayer

    def call(self, inputs, condition):

        # Go through the Dense layers
        x = inputs * 1
        for i, dl in enumerate(self.layers):

            if i == self.concatLayer:
                x = tf.concat([x, condition], axis=1)
                x = dl(x)
            else:
                x = dl(x)

        result = self.result(x)

        return result

class CVAE(Model):

    def __init__(self, nInp, layers, activations, nLatent, concatLayer, lr = 1e-3, name='vae'):
        '''[summary]

        The concat layer is specified with relationship to the decoder, and is zero-indexed. Assume
        that the encoder and decoder layers look like the following:

        [700, 500, 300, 100] -> z -> [100, 300, 500, 700]

        and the conidtional vector should be added to the layer 300 as shown below:

        [700, 500, (300,c), 100] -> z -> [100, (300,c), 500, 700]

        then you would specify ``concatLayer`` as 1, since the layer with 300 nodes is the layer with
        index 1 (assuming the zero-indexed layer). The CVAE will automatically concatinate the input
        vector to the right layer both in the encoder and the decoder, based upon the encoder and the
        decoder. In this specific case where ``concatLayer`` = 1,

        for the decoder, the concatLayer = 1,
        for the encoder, the concatLayer = 4 - concatLayer - 1(for the zero-indexing) = 2

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
        assert concatLayer < len(activations),  "Concatination must happen within the layers of the decoder"

        super(CVAE, self).__init__(name=name)
        
        nLayers = len(layers)

        self.nInp    = nInp
        self.encoder = CVAE_Encoder(nLatent=nLatent, layers=layers, activations=activations, concatLayer = nLayers - concatLayer - 1)
        self.decoder = CVAE_Decoder(nInp, layers=layers, activations=activations, concatLayer=concatLayer)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    def call(self, inputs, condition):

        zMean, zLogVar, z = self.encoder(inputs, condition)
        reconstructed     = self.decoder(z, condition)
        
        return reconstructed

    def step(self, x, condition):

        with tf.GradientTape() as tape:

            zMean, zLogVar, z = self.encoder(x, condition)
            xHat = self.decoder( z, condition)

            # Reconstruction Loss
            reconLoss = tf.nn.sigmoid_cross_entropy_with_logits( x, xHat )
            reconLoss = tf.reduce_sum( reconLoss, 1 )
            reconLoss = tf.reduce_mean( reconLoss )

            # KL - divergence loss
            klLoss    = - 0.5 * tf.reduce_sum(zLogVar - tf.square(zMean) - tf.exp(zLogVar) + 1, 1)
            klLoss    = tf.reduce_mean( klLoss )

            # Calculate the total loss
            loss      = reconLoss + klLoss

            # Optimize
            grads     = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return reconLoss.numpy(), klLoss.numpy(), loss.numpy()

