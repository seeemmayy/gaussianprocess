import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import time
from datetime import date
import s3fs
import os
import boto3
from botocore.client import Config
import json


s3 = s3fs.S3FileSystem(profile="agile-upload",
                       client_kwargs=dict(
                         endpoint_url="https://fhm-chicas-storage.lancs.ac.uk"
                                         )
                      )


def regionandlockdown(df, lockdowndataset, regiondataset, task_config):
    """
    Takes the base dataset and adds in a binary lockdown information
    column and a region column based on the local authority codes for the GP

    Returns a dataframe to be input into the GP function
    """

    #aggregates lockdown dataset by year and week number
    #takes the average lockdown for the week
    #lockdowndataset['consult_date']= lockdowndataset['consult_date'].dt.strftime('%d/%m/%Y')
    #lockdowndataset['year'] = lockdowndataset['consult_date'].dt.year
    lockdown = lockdowndataset.groupby(['year','week_number']).mean()
    lockdown['Lockdown'] = lockdown['Lockdown'].apply(lambda x: 1 if x >= 0.5 else 0)
    lockdown = lockdown.reset_index()

    df['week_number'] = df['date'].dt.week
    df['year'] = (df['date']).dt.year
    df = df.loc[(df['year'] >= 2019)]
    
    #merging region column
    regiondataset = regiondataset.rename(columns={"LAD20CD":"location"})
    df = df.merge(regiondataset, on='location', how='left')
    df = df.rename(columns= {"ITL121NM": "region", "LAD20NM":"local authority", "ITL221NM":"nuts2"})
    df = df[df['species'] == task_config['species']]

    if task_config['location'] is not None:
        df = df[df['region'] == task_config['location']]

    df = df.groupby(['year','week_number']).sum()
    df = df.reset_index()
    df['day'] = df.index * 7
    
    #merging lockdown column
    df = pd.merge(df,lockdown[["year","week_number","Lockdown"]], how="left")
    df['Lockdown'] = df['Lockdown'].fillna(0)
    Lockdownmean = np.mean(df['Lockdown'])
    df['Lockdownoffset'] = df['Lockdown'] - Lockdownmean
    df = df.dropna(how='any', axis=0)

    with s3.open(f"savsnet-agile-artefacts/archive/"+str(time.strftime("%Y-%m-%d"))+"/consult_data/input_data/"+task_config['species']+"/"+task_config['mpc']+"/"+task_config['plot_filename']+".csv", "w") as f:
        df.to_csv(f)

                                
    return df
    

__refdate = pd.to_datetime('2000-01-01')

def BinomGP(y, N, time, time_pred, l, mcmc_iter, start={}):
    """Fits a logistic Gaussian process regression timeseries model.
    Details
    =======
    Let $y_t$ be the number of cases observed out of $N_t$ individuals tested.  Then
    $$y_t \sim Binomial(N_t, p_t)$$
    with 
    $$logit(p_t) \sim s_t$$
    $s_t$ is modelled as a Gaussian process such that
    $$s_t \sim \mbox{GP}(\mu_t, \Sigma)$$
    with a mean function capturing a linear time trend
    $$\mu_t = \alpha + \beta X_t$$
    and a periodic covariance plus white noise
    $$\Sigma_{i,j} = \sigma^2 \exp{ \frac{2 \sin^2(|x_i - x_j|/365)}{\phi^2}} + \tau^2$$
    Parameters
    ==========
    y -- a vector of cases
    N -- a vector of number tested
    X -- a vector of times at which y is observed
    l -- a vector of 0's and 1's whether United Kingdom was in lockdown or not
    T -- a vector of time since recruitment
    X_star -- a vector of times at which predictons are to be made
    start -- a dictionary of starting values for the MCMC.
    Returns
    =======
    A tuple of (model,trace,pred) where model is a PyMC3 Model object, trace is a PyMC3 Multitrace object, and pred is a 5000 x X_star.shape[0] matrix of draws from the posterior predictive distribution of $\pi_t$.
    """
    time = np.array(time)[:, None]  # Inputs must be arranged as column vector
    offset = np.mean(time)
    time = time - offset  # Center time
    model = pm.Model()

    with model:
        alpha = pm.Normal('alpha', 0, 1000, testval=0.)
        beta = pm.Normal('beta', 0, 100, testval=0.)
        sigmasq_s = pm.HalfNormal('sigmasq_s', 5., testval=0.1)
        phi_s = 0.32
        tau2 = pm.HalfNormal('tau2', 5., testval=0.1)
        delta = pm.Normal('delta', 0, 100, testval=0.) 
        

        # Construct GPs
        cov_t = sigmasq_s * pm.gp.cov.Periodic(1, 365., phi_s)
        mean_t = pm.gp.mean.Linear(coeffs=beta, intercept=alpha)
        gp_period = pm.gp.Latent(mean_func=mean_t, cov_func=cov_t)

        cov_nugget = pm.gp.cov.WhiteNoise(tau2)
        gp_nugget = pm.gp.Latent(cov_func=cov_nugget)

        gp_t = gp_period + gp_nugget
        s = gp_t.prior('gp_t', X=time[:, None])
        eta = (delta * l) + s

        Y_obs = pm.Binomial('y_obs', N, pm.invlogit(eta), observed=y)

        # Sample
        trace = pm.sample(mcmc_iter,
                          chains=1,
                          start=start,
                          tune=1000)
                          

        # Prediction
        time_pred_centered = time_pred - offset
        s_star = gp_t.conditional('s_star', time_pred_centered[:, None])
        eta_star = (delta * l) + s_star
        pi_star = pm.Deterministic('pi_star', pm.invlogit(eta_star))
        pred = pm.sample_posterior_predictive(trace, var_names=['y_obs', 'pi_star'])



        return {'model': model, 'trace': trace, 'pred': pred}


    
def plotPrediction(ax,X,y,N,pred,mindate,task_config,lag=None,prev_mult=1,plot_gp=False):
    """Predictive time series plot with (by default) the prediction
    summarised as [0.01,0.05,0.5,0.95,0.99] quantiles, and observations colour-coded
    by tail-probability.
    Parameters
    ==========
    ax -- a set of axes on which to plot
    X  -- 1D array-like of times of length n
    y  -- 1D array-like of observed number of cases at each time of length n
    N  -- 1D array-like of total number at each time of length n
    pred -- 2D m x n array with numerical draws from posterior
    mindate -- a pandas.Timesav xxtamp representing the time origin wrt X
    lag     -- how many days prior to max(X) to plot
    prev_mult -- prevalence multiplier (to get in, eg. prev per 1000 population)
    plot_gp -- plots a GP smudge-o-gram rather than 95% and 99% quantiles.
    Returns
    =======
    Nothing.   Just modifies ax
    """

    fig,ax=plt.subplots(figsize=(10,5))

    # Time slice
    ts = slice(0,X.shape[0])
    if lag is not None:
        ts = slice(X.shape[0]-lag, X.shape[0])
    
    # Data
    x = np.array([mindate + pd.Timedelta(d,unit='D') for d in X[ts]])
    pbar = np.array(y/N)[ts] * prev_mult    


    # Prediction quantiles
    phat = pred[:,ts] * prev_mult
    pctiles = np.percentile(phat, [1,5,50,95,99], axis=0)

    # Tail probabilities for observed p
    prp = np.sum(pbar > phat, axis=0)/phat.shape[0]
    prp[prp > .5] = 1. - prp[prp > .5]
    
    # Risk masks
    red = prp <= 0.01
    orange = (0.01 < prp) & (prp <= 0.05)
    green = 0.05 < prp

    # Construct plot
    if plot_gp is True:
        from pymc3.gp.util import plot_gp_dist
        plot_gp_dist(ax,phat,x,plot_samples=False, palette="Blues")
    else:
        ax.fill_between(x, pctiles[4,:], pctiles[0,:], color='lightgrey',alpha=.5,label="99% credible interval")
        ax.fill_between(x, pctiles[3,:], pctiles[1,:], color='lightgrey',alpha=1,label='95% credible interval')
        grey = mpatches.Patch(color='silver', label='At Least one lockdown restriction in place')
        blue = mpatches.Patch(color='cornflowerblue', label='National Lockdown in place')
        ax.axvspan('2020-03-21','2020-07-03', color='royalblue', alpha = .5, zorder = -1)
        ax.axvspan('2020-09-17','2020-12-03', color='royalblue', alpha = .5, zorder = -1)
        ax.axvspan('2021-01-04','2021-03-31', color='royalblue', alpha = .5, zorder = -1)
        ax.axvspan('2020-03-21','2021-03-31', color='lightsteelblue', alpha = .4)
        ax.plot(x, pctiles[2,:], c='grey', ls='-', label="Predicted prevalence")
        ax.scatter(x[green],pbar[green],c='green',s=8,alpha=0.5,label='0.05<p')
        ax.scatter(x[orange],pbar[orange],c='orange',s=8,alpha=0.5,label='0.01<p<=0.05')
        ax.scatter(x[red],pbar[red],c='red',s=8,alpha=0.5,label='p<=0.01')
        legend_elements = [Line2D([0], [0], color='grey', lw=2, label="Predicted Prevalence"),
                           Patch(facecolor='lightgrey', alpha=.5, label = "99% credible interval"),
                           Patch(facecolor='lightgrey', alpha=.2, label = "95% credible interval"),
                           Line2D([0], [0], color='green', marker='o', linestyle='None', label='0.05<p'),
                           Line2D([0], [0], color='orange', marker='o', linestyle='None',label='0.01<p<=0.05'),
                           Line2D([0], [0], color='red', marker='o', linestyle='None', label='p<=0.01'),
                           Patch(facecolor='royalblue', alpha=.5, label = "National Lockdown in Place"),
                           Patch(facecolor='lightsteelblue', alpha=.5, label = "At Least One Lockdown in Place")
                          ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.04,1), loc="upper left")
        ax.set_xlabel('Date')
        ax.yaxis.set_label_text('Prevalence / 1000 records')
        ax.set_title(task_config['plot_title'])
        ax.set_xlim(pd.Timestamp('2021-01-01'), time.strftime("%Y-%m-%d"))
        ax.figure.savefig(task_config['plot_filename'], format='png', dpi=300, bbox_inches='tight')


        s3 = s3fs.S3FileSystem(profile="agile-upload",
                       client_kwargs=dict(
                         endpoint_url="https://fhm-chicas-storage.lancs.ac.uk"
                                         )
                      )

        with s3.open(f"savsnet-agile-artefacts/archive/"+str(time.strftime("%Y-%m-%d"))+"/consult_data/plots/"+task_config['species']+"/"+task_config['mpc']+"/"+task_config['plot_filename']+".png", "wb") as f:
            ax.figure.savefig(f, dpi=300, bbox_inches='tight')


        with s3.open(f"savsnet-agile-artefacts/public/consult_data/latest/"+task_config['plot_filename']+".png", "wb") as f:
            ax.figure.savefig(f, dpi=300, bbox_inches='tight')

        os.remove(task_config['plot_filename'])



