import matplotlib.pyplot as plt
import numpy as np

def plotMNISTImages(epoch, vae, x_test, y_test, logits=True, folder=None):
    
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

    plt.figure()

    for k, v in losses.items():
        plt.plot( v, label=k )
    plt.legend()
    plt.yscale('log')

    if folder is None:
        outFile = f'results/losses.png'
    else:
        outFile = f'results/{folder}/losses.png'

    plt.savefig(outFile)

    return

def plotMNISTLatentSpace(epoch, vae, x_test, y_test, folder=None):

    _, _, z = vae.encoder( x_test )
    z = z.numpy()
    
    sel_ys = [ np.where(y_test == i)[0] for i in range(10) ]
    colors = [ plt.cm.viridis(i)  for i in np.linspace(0.1, 0.9, 10)]

    plt.figure()
    for i, sel_y in enumerate(sel_ys):
        tempZ = z[ sel_y, : ]
        plt.plot( tempZ[:, 0], tempZ[:, 1], 's', mec='None', mfc=colors[i], label=f'{i}', alpha=0.2 )
    plt.legend()

    if folder is None:
        outFile = f'results/{epoch:05d}_LatentSpace.png'
    else:
        outFile = f'results/{folder}/{epoch:05d}_LatentSpace.png'

    plt.savefig(outFile)
    
    return 

def plotMNISTLatentReconstruction(epoch, vae, extent=(-3, 3), nSteps=10, logits=True, folder=None):

    
    plt.figure(figsize=(5,5), facecolor='black')
    axes, zVals, keys = {}, [], []
    
    for x in np.linspace(extent[0], extent[1], nSteps):
        for y in np.linspace(extent[1], extent[0], nSteps):
            zVals.append([x, y])

    zVals = np.array(zVals)
    xHats = vae.decoder( zVals ).numpy()
    if logits:
        # convert to a sigmoid ...
        xHats = xHats.clip(-1e-2, 1e2)
        xHats = 1/( 1 + np.exp( -xHats ) )

    for i in range(nSteps):
        for j in range(nSteps):

            tempI, tempJ = i/nSteps, j/nSteps
            temp = plt.axes( [tempI, tempJ, 1/nSteps, 1/nSteps] )
            axes[(i,j)] = temp

            keys.append((i,j))

            temp.imshow( xHats[ i*nSteps + j ].reshape(28,28), cmap='gray' )
            temp.axis('off')
            
    if folder is None:
        outFile = f'results/{epoch:05d}_LatentReconstruction.png'
    else:
        outFile = f'results/{folder}/{epoch:05d}_LatentReconstruction.png'

    plt.savefig(outFile)
    
    return 
