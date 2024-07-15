import os
import sys
import glob
import logging
import shutil
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

logger = logging.getLogger('ceciestunepipe.util.information')

def discrete_mutual_info(p_x: np.array, p_y: np.array, p_xy: np.array) -> float:
    # just compute the mutual information between the two marginal pdfs given the joint pdf p_xy (for instance, as computed in discrete_pdf)
    # an array with all the marginals
    n = p_xy.shape[0] # n of bins
    px = p_x.reshape(n, 1)
    py = p_y.reshape(1, n)
    px_py = px @ py
    
    # element by element multiplication of p(x,y) * log(p(x,y)/(p(x)*p(y)))
    mutual_info_arr = p_xy * np.log(p_xy/px_py)
    
    return np.ma.masked_invalid(mutual_info_arr).sum()


def discrete_pdf(xy: np.array, plot=True) -> pd.DataFrame:
    # x: (2, m) array with two samples of natural numbers
    # given two lists of integers, get the distributions of each, the joint pdf and the mutual information
    # the limits of the histograms for the distributions
    edges = np.arange(np.min(xy), np.max(xy)+1)
    bins = edges[:-1]
    #print(edges.size)
    
    p_x, _ = np.histogram(xy[0], bins=edges, density=True)
    p_y, _ = np.histogram(xy[1], bins=edges, density=True)
    p_xy, _, _ = np.histogram2d(xy[0], xy[1], bins=[edges, edges], density=True)
    
    # an ordered one by rank of probabilites
    x_rank, y_rank = [np.argsort(x)[::-1] for x in [p_x, p_y]]
    f_xy, _, _ = np.histogram2d(xy[0], xy[1], bins=[edges, edges], density=True)
    
    
    if plot:
        fig = plt.figure(figsize=(6, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.1, hspace=0.1)

        ax_histxy = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_histxy)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_histxy)
        
        ax_histx.plot(bins, p_x)
        ax_histx.plot(bins, p_x[x_rank])
        ax_histx.set_title('p(x)')
        ax_histx.set_xticks([])

        ax_histy.plot(p_y, bins)
        ax_histy.plot(p_y[y_rank], bins)
        ax_histy.set_title('p(y)')
        ax_histy.set_yticks([])

        ax_histxy.imshow(p_xy[x_rank, :][:, y_rank].T)
        ax_histxy.set_title('p(x,y)')
        
        ax_dict = {'x': ax_histx, 'y': ax_histy, 'xy': ax_histxy}
    else:
        ax_dict = {}
            
    pdf_df = pd.DataFrame({'x': bins, 'y': bins, 'p_x':p_x, 'p_y': p_y, 'x_rank': x_rank, 'y:rank': y_rank})
        
    return pdf_df, f_xy, ax_dict
        