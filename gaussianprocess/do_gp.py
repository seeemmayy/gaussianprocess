import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from binomGP import regionandlockdown, BinomGP, plotPrediction



def do_gp(rawdataset, lockdowndataset, regiondataset, task_config, mindate, ax):

    """
    Calls on the functions in the binomGP file using the inputs given from the gp_array.py file.
    Formats raw datasets, runs gaussian process and produces plots.

    Returns nothing.
    
    """

    formatteddf = regionandlockdown(rawdataset,
                                    lockdowndataset,
                                    regiondataset,
                                    task_config)
    
    gaussianprocess = BinomGP(y=formatteddf[task_config['mpc']].to_numpy(),
                              N=formatteddf['total_count'].to_numpy(),
                              time=formatteddf['day'].to_numpy(),
                              time_pred=formatteddf['day'].to_numpy(),
                              l=formatteddf['Lockdownoffset'].to_numpy(),
                              mcmc_iter=5000)

    plotPrediction(ax,
                   formatteddf['day'],
                   formatteddf[task_config['mpc']],
                   formatteddf['total_count'],
                   gaussianprocess['pred']['pi_star'],
                   mindate,
                   task_config,
                   lag=None,
                   prev_mult=1000,
                   plot_gp=False)


