import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import VAE 
from datetime import datetime as dt

from utils import plotUtils as pU
from utils import dataUtils as dU

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    nInp      = 784  # (28*28) shaped images 
    batchSize = 1024
    EPOCHS    = 20

    # --------- [ Generate the data ] ---------------------
    (x_train, y_train), (x_test, y_test) = dU.getMNISTData()
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batchSize)
    
    # --------- [ Generate the model ] ---------------------
    layers      = [700, 500, 100]
    activations = ['tanh', 'tanh', 'tanh']
    nLatent     = 2

    vae = VAE.VAE(nInp, layers=layers, activations = activations, nLatent = nLatent)

    # --------- [ Train the model ] ---------------------
    for epoch in range(EPOCHS):
        print('Start of epoch %d' % (epoch,), end='-> ')

        # Iterate over the batches of the dataset.
        for step, x in enumerate(train_dataset):
            reconLoss, klLoss, loss = vae.step( x )

            if step % 100 == 0:
                print(reconLoss, klLoss, loss)

        if epoch % 2 == 0:
            pU.plotMNISTImages(epoch, vae, x_test, y_test, now)


    return

if __name__ == "__main__":
    main()
    