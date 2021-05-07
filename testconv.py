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
        'nFilters'    : [2, 5, 10], 
        'kernelSizes' : [3, 3, 3], 
        'strideSizes' : [1, 1, 1], 
        'activations' : [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh], 
        'paddings'    : ['same', 'same', 'same'],
    }

    cv = ConvVAE.ConvEncoder(nInpX, nInpY, nInpCh, nLatent, **encoderSpecs)

    return

if __name__ == "__main__":
    main()
