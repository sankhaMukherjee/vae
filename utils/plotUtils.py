import matplotlib.pyplot as plt
import numpy as np

def plotMNISTImages(epoch, vae, x_test, y_test, now):
    '''[summary]

    [extended_summary]

    Parameters
    ----------
    epoch : [type]
        [description]
    vae : [type]
        [description]
    x_test : [type]
        [description]
    y_test : [type]
        [description]
    now : [type]
        [description]
    '''

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
