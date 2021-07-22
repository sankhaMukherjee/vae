import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model, layers


class Encoder(layers.Layer):

    def __init__(self, nLatent, layers, activations, name='encoder'):

        super(Encoder, self).__init__(name=name)

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.mean    = Dense(nLatent)
        self.logVar  = Dense(nLatent)

    def call(self, inputs):

        # Go through the Dense layers
        x = inputs * 1
        for dl in self.layers:
            x = dl(x)
        
        # Create the latent layer (z)
        zMean   = self.mean(x)
        zLogVar = self.logVar(x)
        epsilon = tf.random.normal( shape=zMean.shape, mean=0, stddev=1 )
        z       = zMean + tf.exp( 0.5 * zLogVar )*epsilon

        return zMean, zLogVar, z

class Decoder(layers.Layer):

    def __init__(self, nInp, layers, activations, name='encoder'):

        super(Decoder, self).__init__(name=name)

        layers      = list(reversed(layers))
        activations = list(reversed(activations))

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.result  = Dense(nInp, activation=None)

    def call(self, inputs):


        # Go through the Dense layers
        x = inputs * 1
        for dl in self.layers:
            x = dl(x)
        
        result = self.result(x)

        return result

class SigmaVAE(Model):

    def __init__(self, nInp, layers, activations, nLatent, lr = 1e-3, name='vae'):
        
        super(SigmaVAE, self).__init__(name=name)
        
        self.nInp    = nInp
        self.nLatent = nLatent
        self.encoder = Encoder(nLatent=nLatent, layers=layers, activations=activations)
        self.decoder = Decoder(nInp, layers=layers, activations=activations)

        # This is going to be a learned parameter
        self.sigma   = tf.Variable(1, trainable=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    def call(self, inputs):

        zMean, zLogVar, z = self.encoder(inputs)
        reconstructed     = self.decoder(z)
        
        return reconstructed

    def step(self, x):

        with tf.GradientTape() as tape:

            # This is the dimensionality of the input
            D = self.nInp

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
            # The log sigma might cause things to become 
            loss      =  D * tf.log( self.sigma ) +  reconLoss / (2 * self.sigma**2)  + klLoss

            # Optimize
            grads     = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return reconLoss.numpy(), klLoss.numpy(), loss.numpy()

    def checkpoint(self, folder):

        folder = os.path.join( folder, 'model', 'modelData' )
        os.makedirs( folder, exist_ok=True )
        self.save_weights( folder )
        return
