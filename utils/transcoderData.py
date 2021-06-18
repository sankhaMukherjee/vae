import numpy as np
from tqdm import tqdm

def main():

    s = np.array([0.5, 0.12])
    S = np.random.rand(2,2) 
    D = np.random.rand(2,3)
    sVals = [s]

    print('Generating data for the latent space ...')
    print('----------------------------------------')
    for i in tqdm(range(10000)):
        s = s@S
        s = np.log(np.abs(s))
        sVals.append(s)
    
    sVals = np.vstack(sVals)
    mVals = sVals @ D

    np.save( 'data/sVals.npy', sVals)
    np.save( 'data/mVals.npy', mVals)
    
    return 

if __name__ == "__main__":
    main()
