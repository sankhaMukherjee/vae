import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

print( tf.config.list_physical_devices() )

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import sigmaVAE
from datetime import datetime as dt
import numpy as np

from utils import plotUtils as pU
from utils import dataUtils as dU

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-vae')
    os.makedirs(f'results/{now}')

    nInp      = 784  # (28*28) shaped images 
    batchSize = 1024
    EPOCHS    = 100

    # --------- [ Generate the data ] ---------------------
    (x_train, y_train), (x_test, y_test) = dU.getMNISTData()
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batchSize)
    
    # --------- [ Generate the model ] ---------------------
    layers      = [700, 500, 100]
    activations = ['relu', 'relu', 'relu']
    # activations = ['tanh', 'tanh', 'tanh']
    nLatent     = 2

    vae = sigmaVAE.SigmaVAE(nInp, layers=layers, activations = activations, nLatent = nLatent)

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


    # vae.predict(x_train[:10])
    vae.checkpoint( f'results/{now}')

    pU.plotLosses(losses, folder=now)
    pU.plotMNISTLatentSpace(epoch, vae, x_test, y_test, folder=now)
    pU.plotMNISTImages(epoch, vae, x_test, y_test, logits=True, folder=now)
    pU.plotMNISTLatentReconstruction(epoch, vae, extent=(-3, 3), nSteps=21, logits=True, folder=now)

    return

if __name__ == "__main__":
    main()
    