#!/usr/bin/env python3
""" 
 plot dataset
"""

__author__ = "Ziqing Guo"
__email__ = "ziqguo@ttu.edu"

import numpy as np
import  time
import sys,os
from pprint import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'toolbox')))
from PlotterBackbone import PlotterBackbone
from Util_IOfunc import  read_yaml
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import torch
from sklearn.datasets import make_moons

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-s", "--shift", type=bool, default=True, help="whether shift the dots")

    parser.add_argument("--outPath",default='out',help="all outputs from experiment")
       
    args = parser.parse_args()
    # make arguments  more flexible
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    args.showPlots=''.join(args.showPlots)

    return args

def get_moon():

    # Set random seeds
    torch.manual_seed(149)
    np.random.seed(149)

    X, y = make_moons(n_samples=600, noise=0.1)
    y_ = torch.unsqueeze(torch.tensor(y), 1)  # used for one-hot encoded labels
    y_hot = torch.scatter(torch.zeros((600, 2)), 1, y_, 1)

    # Adjusted color scheme for better contrast
    c = ["#3498db" if y_ == 0 else "#e74c3c" for y_ in y]  # New colors for each class
    return X, c

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)


#...!...!....................
    def compute_time(self,bigD,tag1,figId=1,shift=False):
        nrow,ncol=1,1       
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white',figsize=(5.5,7))        
        ax = self.plt.subplot(nrow,ncol,1)
       
        # ax.plot(nqV_shifted, runtV_shifted, marker=marker_style, linestyle='-', markerfacecolor=isFilled, color=dCol,label=dLab,markersize=9)     
        X, c = get_moon()
        ax.scatter(X[:, 0], X[:, 1], c=c)
              
        tit='Compute Moon'
        # Place the title above the legend
        ax.axis('off')
        ax.set_title(tit, pad=50)  # Adjust the pad value as needed

#...!...!....................
def find_yaml_files(directory_path, vetoL=None):
    """
    Scans the specified directory for all files with a .h5 extension,
    rejecting files whose names contain any of the specified veto strings.

    Args:
    directory_path (str): The path to the directory to scan.
    vetoL (list): A list of strings. Files containing any of these strings in their names will be rejected.

    Returns:
    list: A list of paths to the .yaml files found in the directory, excluding vetoed files.
    """
    if vetoL is None:
        vetoL = []
   
    h5_files = []
    print('scan path:',directory_path)
    for root, dirs, files in os.walk(directory_path):
        print('found %d any files'%len(files))
        for file in files:
            if file.endswith('.yaml') and not any(veto in file for veto in vetoL):
                h5_files.append(os.path.join(root, file))
    return h5_files

#...!...!....................            
def sort_end_lists(d, parent_key='', sort_key='nq', val_key='runt'):
    """
    Recursively prints all keys in a nested dictionary.
    Once the sort_key is in dict it triggers sorting both keys.

    Args:
    d (dict): The dictionary to traverse.
    parent_key (str): The base key to use for nested keys (used for recursion).
    sort_key (str): The key indicating the list to sort by.
    val_key (str): The key indicating the list to sort alongside.
    """
    if sort_key in d:
        xV = d[sort_key]
        yV = d[val_key]
        xU, yU = map(list, zip(*sorted(zip(xV, yV), key=lambda x: x[0])))
        print(' %s.%s:%d' % (parent_key, sort_key, len(xU)))
        d[sort_key]=np.array(xU)
        d[val_key]=np.array(yU)
        return
    
    for k, v in d.items():
        full_key = '%s.%s' % (parent_key, k) if parent_key else k
        print(full_key)
        if isinstance(v, dict):
            sort_end_lists(v, full_key, sort_key, val_key)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    # ----  just plotting
    args.prjName='mar24'
    plot=Plotter(args)
    corePath='/pscratch/sd/g/gzquse/quantDataVault2024/dataPenny'  # bare metal Martin
    if 'a' in args.showPlots:
        plot.compute_time(None,'moon', figId=1, shift=args.shift)
    if 'b' in args.showPlots:
        plot.compute_time(None,'spiral',figId=2, shift=args.shift)
    plot.display_all(png=1)
    
