import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from utils import dataUtils as dU

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import VAE_test
from datetime import datetime as dt
import numpy as np

def main():

    nInp      = 784  # (28*28) shaped images 
    batchSize = 1024
    EPOCHS    = 100

    layers      = [700, 500, 100]
    activations = ['relu', 'relu', 'relu']
    # activations = ['tanh', 'tanh', 'tanh']
    nLatent     = 2

    # --------- [ Generate the data ] ---------------------
    (x_train, y_train), (x_test, y_test) = dU.getMNISTData()
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(batchSize)

    vae = VAE_test.VAE(nInp, layers=layers, activations = activations, nLatent = nLatent)
    for step, x in enumerate(train_dataset):
        vae.checkWeights(x)
        break
    

    return


if __name__ == "__main__":
    main()
