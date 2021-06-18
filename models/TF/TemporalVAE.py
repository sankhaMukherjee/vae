import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras         import Model, layers
from tensorflow.keras.layers  import Dense, Input
from tensorflow.keras.losses  import MeanSquaredError

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

class Transition(layers.Layer):

    def __init__(self, nLatent, layers, activations, name='encoder'):

        # Here, nLatent is the size of the latent layer
        super(Transition, self).__init__(name=name)

        self.layers  = [ Dense(l, activation=a) for l, a in zip(layers, activations) ]
        self.tr      = Dense(nLatent)

    def call(self, s_t):

        # Go through the Dense layers
        x = s_t * 1
        for dl in self.layers:
            x = dl(x)
        
        # Create the latent layer (z)
        s_tP1 = self.tr(x)
    
        return s_tP1

class TemporalVAE(Model):

    def __init__(self, nInp, layersEnc, activationsEnc, nLatent, layersTrans, activationsTrans, lr = 1e-3, name='vae'):
        
        super(TemporalVAE, self).__init__(name=name)
        
        self.nInp    = nInp
        self.nLatent = nLatent
        self.encoder = Encoder(nLatent=nLatent, layers=layersEnc, activations=activationsEnc)
        self.decoder = Decoder(nInp, layers=layersEnc, activations=activationsEnc)

        self.transition = Transition(nLatent=nLatent, layers=layersTrans, activations=activationsTrans)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    def call(self, inputs):

        zMean, zLogVar, z = self.encoder(inputs)
        reconstructed1    = self.decoder(zMean)
        z_p1              = self.transition(zMean)
        reconstructed2    = self.decoder(z_p1)
        
        return reconstructed, reconstructed2, z_p1

    def step(self, x):

        x1, x2 = x

        with tf.GradientTape() as tape:

            zMean, zLogVar, z = self.encoder(x1)
            reconstructed1    = self.decoder(z)
            z_p1              = self.transition(z)
            reconstructed2    = self.decoder(z_p1)

            # Reconstruction Loss 1
            reconLoss1 = (x1 - reconstructed1)**2
            reconLoss1 = tf.reduce_sum( reconLoss1, 1 )
            reconLoss1 = tf.reduce_mean( reconLoss1 )

            # Reconstruction Loss 2
            reconLoss2 = (x2 - reconstructed2)**2
            reconLoss2 = tf.reduce_sum( reconLoss2, 1 )
            reconLoss2 = tf.reduce_mean( reconLoss2 )

            # KL - divergence loss
            klLoss    = - 0.5 * tf.reduce_sum(zLogVar - tf.square(zMean) - tf.exp(zLogVar) + 1, 1)
            klLoss    = tf.reduce_mean( klLoss )
            klLoss    = klLoss

            # Calculate the total loss
            loss      = reconLoss1 + reconLoss2  #+ klLoss

            # Optimize
            grads     = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return reconLoss1.numpy(), reconLoss2.numpy(), klLoss.numpy(), loss.numpy()

    def checkpoint(self, folder):

        folder = os.path.join( folder, 'model', 'modelData' )
        os.makedirs( folder, exist_ok=True )
        self.save_weights( folder )
        return
