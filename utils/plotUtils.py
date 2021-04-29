import matplotlib.pyplot as plt
import numpy as np

def plotMNISTImages(epoch, vae, x_test, y_test, logits=True, folder=None):
    
    _, _, z = vae.encoder( x_test )
    z = z.numpy()
    
    sel_ys = [ np.where(y_test == i)[0][0] for i in range(10) ]
    sel_xs = x_test[sel_ys, :]
    
    fig = plt.figure(figsize=(10,2))
    axes = []
    for i in range(20):
        x = (i//2)/10
        y = i%2
        axes.append(plt.axes([x, y*0.5, 0.1, 0.5]))

    x_hat = vae( sel_xs ).numpy()

    if logits:
        # convert to a sigmoid ...
        x_hat = x_hat.clip(-1e-2, 1e2)
        x_hat = 1/( 1 + np.exp( -x_hat ) )

    for i, (x1, x2) in enumerate(zip( sel_xs, x_hat )):
        axes[i*2].imshow(   x2.reshape( 28, 28 ), cmap='gray' )
        axes[i*2+1].imshow( x1.reshape( 28, 28 ), cmap='gray' )
        axes[i*2].axis('off')
        axes[i*2+1].axis('off')

    if folder is None:
        outFile = f'results/{epoch:05d}_reconstruction.png'
    else:
        outFile = f'results/{folder}/{epoch:05d}_reconstruction.png'

    plt.savefig(outFile)

    return

def plotLosses(losses, folder=None):



    return
