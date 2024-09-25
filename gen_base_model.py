import argparse

from pennylane import broadcast
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int, help="increase debug verbosity", default=1)


    parser.add_argument("--expName",  default=None,help='(optional) ')

    # IO paths
    parser.add_argument("--basePath",default=None,help="head path for set of experiments, or 'env'")
    parser.add_argument("--outPath",default='out/',help="(optional) redirect all outputs ")
    
    args = parser.parse_args()
    return args

def get_dataset():

    # Set random seeds
    torch.manual_seed(149)
    np.random.seed(149)

    X, y = make_moons(n_samples=400, noise=0.1)
    y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
    y_hot = torch.scatter(torch.zeros((400, 2)), 1, y_, 1)

    # Adjusted color scheme for better contrast
    c = ["#3498db" if y_ == 0 else "#e74c3c" for y_ in y]  # New colors for each class
    plt.axis("off")
    plt.scatter(X[:, 0], X[:, 1], c=c)
    plt.show()


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    bigD=importImage(args)   
    MD=buildMeta_CannedImage(args,bigD)
    pprint(MD)
    
    prepImageQCrankInput(MD,bigD)
    
      
    print('M:inp bigD:',sorted(bigD))
    
    outF=MD['short_name']+'.qcrank_inp.h5'
    fullN=os.path.join(args.outPath,outF)
    write4_data_hdf5(bigD,fullN,metaD=MD)

    pprint(MD)

    print('local sim for cpu:\n time  ./run_aer_job.py --cannedExp   %s   -n 300   -E \n'%(MD['short_name'] ))
    print('local sim for 4 gpus:\n mpirun -np 4  ./run_cudaq_job.py --circName   %s   -n 300   \n'%(MD['short_name'] ))
    print('M:done')