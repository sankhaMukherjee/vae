import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from models.TF import VAE 
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def getMNISTData():

    # Create the data
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(60000, 784).astype('float32')
    x_test  = x_test.reshape(-1, 784).astype('float32')

    return (x_train, y_train), (x_test, y_test)

def plotImages(epoch, vae, x_test, y_test, now):

    z_mean, z_log_var, z = vae.encoder( x_test )
    z = z.numpy()
    
    sel_ys = [ np.where(y_test == i)[0][0] for i in range(10) ]
    sel_xs = x_test[sel_ys, :]
    
    fig = plt.figure(figsize=(10,2))

    x_hat = vae( sel_xs ).numpy()
    # convert to a sigmoid ...
    x_hat = x_hat.clip(-1e-2, 1e2)
    x_hat = 1/( 1 + np.exp( -x_hat ) )

    for i, (x1, x2) in enumerate(zip( sel_xs, x_hat )):
        plt.subplot(2,10, i+1)
        plt.imshow( x1.reshape( 28, 28 ) )
        plt.subplot(2,10, i+11)
        plt.imshow( x2.reshape( 28, 28 ) )

    plt.savefig(f'results/{now}_{epoch:05d}_imgs.png')

    return

def main():

    now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')

    nInp      = 784  # (28*28) shaped images 
    batchSize = 1024
    EPOCHS    = 20

    # --------- [ Generate the data ] ---------------------
    (x_train, y_train), (x_test, y_test) = getMNISTData()
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
            plotImages(epoch, vae, x_test, y_test, now)


    return

if __name__ == "__main__":
    main()
    