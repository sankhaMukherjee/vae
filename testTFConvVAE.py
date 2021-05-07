import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import ConvVAE 
from datetime import datetime as dt
import numpy as np

from utils import plotUtils as pU
from utils import dataUtils as dU

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-ConvVAE')
    os.makedirs(f'results/{now}')

    batchSize = 1024
    EPOCHS    = 200

    # --------- [ Generate the data ] ---------------------
    (x_train, y_train), (x_test, y_test) = dU.getMNISTData(reshape=False)
    x_train1 = x_train.reshape( -1, 28, 28, 1 )
    x_test1  = x_test.reshape( -1, 28, 28, 1 )
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train1)
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batchSize)
    
    # --------- [ Generate the model ] ---------------------
    nInpX  = 28
    nInpY  = 28
    nInpCh = 1
    
    nLatent = 2
    
    encoderSpecs = {
        'nFilters'    : [2, 5, 10], 
        'kernelSizes' : [3, 3, 3], 
        'strideSizes' : [1, 1, 1], 
        'activations' : [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh], 
        'paddings'    : ['same', 'same', 'same'],
    }

    decoderSpecs = {
        'nFilters'    : [10, 5, 5, 5, 1], 
        'kernelSizes' : [5, 7, 7, 6, 6], 
        'strideSizes' : [1, 1, 1, 1, 1], 
        'activations' : [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.tanh], 
        'paddings'    : ['valid', 'valid', 'valid', 'valid', 'valid'],
    }

    vae = ConvVAE.ConvVAE(nInpX, nInpY, nInpCh, nLatent, encoderSpecs, decoderSpecs)

    # --------- [ Train the model ] ---------------------
    losses = []
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,), end='-> ')

        # Iterate over the batches of the dataset.
        for step, x in enumerate(train_dataset):
            reconLoss, klLoss, loss = vae.step( x )
            losses.append([reconLoss, klLoss, loss])

            if step % 100 == 0:
                print(reconLoss, klLoss, loss)

    # ------------- [plot everything] -----------------
    losses = np.array(losses).T
    losses = {
        'reconstruction' : losses[0],
        'KL Divergence'  : losses[1],
        'Total'          : losses[2]}

    pU.plotLosses(losses, folder=now)
    pU.plotMNISTLatentSpace(epoch, vae, x_test1, y_test, folder=now)
    pU.plotMNISTImages(epoch, vae, x_test1, y_test, logits=True, folder=now)
    pU.plotMNISTLatentReconstruction(epoch, vae, extent=(-3, 3), nSteps=21, logits=True, folder=now)

    return

if __name__ == "__main__":
    main()
    