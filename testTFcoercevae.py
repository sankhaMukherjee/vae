import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import coerceVAE 
from datetime import datetime as dt
import numpy as np

from utils import plotUtils as pU
from utils import dataUtils as dU

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-coerce')
    os.makedirs(f'results/{now}')

    nInp      = 784  # (28*28) shaped images 
    batchSize = 1024
    EPOCHS    = 150

    # --------- [ Generate the data ] ---------------------
    (x_train, y_train), (x_test, y_test) = dU.getMNISTData()
    y_train1 = y_train.reshape(-1, 1).astype( np.float32 ) # required for the concatenation
    y_test1  = y_test.reshape(-1, 1).astype(  np.float32 ) # required for the concatenation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train1))
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batchSize)
    
    # --------- [ Generate the model ] ---------------------
    layers      = [700, 500, 100]
    activations = ['tanh', 'tanh', 'tanh']
    nLatent     = 2

    coerceVae = coerceVAE.CoerceVAE(nInp, layers=layers, activations = activations, nLatent = nLatent)

    # --------- [ Train the model ] ---------------------
    losses = []
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,), end='-> ')

        # Iterate over the batches of the dataset.
        for step, (x, y) in enumerate(train_dataset):
            reconLoss, klLoss, coerceLoss, loss = coerceVae.step( x, y )
            losses.append([reconLoss, klLoss, coerceLoss, loss])

            if step % 100 == 0:
                print(reconLoss, klLoss, coerceLoss, loss)

    # ------------- [plot everything] -----------------
    losses = np.array(losses).T
    losses = {
        'reconstruction' : losses[0],
        'KL Divergence'  : losses[1],
        'coerce Loss'    : losses[2],
        'Total'          : losses[3],
        }

    pU.plotLosses(losses, folder=now)
    pU.plotMNISTLatentSpaceCoerce(epoch, coerceVae, x_test, y_test, folder=now)
    pU.plotMNISTImages(epoch, coerceVae, x_test, y_test, logits=True, folder=now, condition=False)
    pU.plotMNISTLatentReconstruction(epoch, coerceVae, extent=(-2, 2), nSteps=21, logits=True, folder=now, condition=False)
    
    return

if __name__ == "__main__":
    main()
    