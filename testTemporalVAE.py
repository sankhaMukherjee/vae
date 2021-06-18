import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import TemporalVAE 
from datetime import datetime as dt
import numpy as np

from utils import plotUtils as pU
from utils import dataUtils as dU

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S-Temporalvae')
    os.makedirs(f'results/{now}')

    nInp      = 3    # mse values
    batchSize = 512
    EPOCHS    = 200

    # --------- [ Generate the data ] ---------------------
    mVals = np.load('data/mVals.npy')
    sVals = np.load('data/sVals.npy') # This is the latent data. We basically dont need this

    mVals = mVals.astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices((mVals[:-1], mVals[1:]))
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batchSize)
    
    # --------- [ Generate the model ] ---------------------
    layers           = [20, 40, 20]
    activations      = ['tanh', 'tanh', 'tanh']
    layersTrans      = [20, 40, 20]
    activationsTrans = ['tanh', 'tanh', 'tanh']
    nLatent          = 2

    temporalVae = TemporalVAE.TemporalVAE(
        nInp, # This isthe shape of the MSE
        layersEnc=layers, activationsEnc = activations, nLatent = nLatent, # Hidden layers for the encoder/decoders
        layersTrans=layersTrans, activationsTrans=activationsTrans,  # Hidden layers for the transition
    )

    # --------- [ Train the model ] ---------------------
    losses = []
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,), end='-> ')

        # Iterate over the batches of the dataset.
        for step, x in enumerate(train_dataset):
            reconLoss1, reconLoss2, klLoss, loss = temporalVae.step( x )
            losses.append([reconLoss1, reconLoss2, klLoss, loss])

            if step % 100 == 0:
                # print(f'{reconLoss1:03e}, {reconLoss2:03e}, {klLoss:03e}, {loss:03e}')
                print(f'{reconLoss1:03e}, {reconLoss2:03e}, {klLoss:03e}, {loss:03e}')

    # ------------- [plot everything] -----------------
    losses = np.array(losses).T
    losses = {
        'recon1' : losses[0],
        'recon2' : losses[1],
        'klLoss' : losses[2],
        'total'  : losses[3]}


    # # vae.predict(x_train[:10])
    # vae.checkpoint( f'results/{now}')

    pU.plotLosses(losses, folder=now)
    # pU.plotMNISTLatentSpace(epoch, vae, x_test, y_test, folder=now)
    # pU.plotMNISTImages(epoch, vae, x_test, y_test, logits=True, folder=now)
    # pU.plotMNISTLatentReconstruction(epoch, vae, extent=(-3, 3), nSteps=21, logits=True, folder=now)

    return

if __name__ == "__main__":
    main()
    