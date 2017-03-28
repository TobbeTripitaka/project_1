from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

import glob
import cProfile

from functools import partial
import matplotlib.cm as cm

from scipy import stats

def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    
    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = observation_likelihood.pdf(x)
        
        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))
       
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        
        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)
    
        maxes[t] = R[:, t].argmax()
    return R, maxes


def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)


class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data, 
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0



def import_data(fname, header_rows=1, step=1, normalize=False):
    '''
    fname: string, file to read
    header rows: integer 
    step: integer Subsampling of dataset to increase speed
    normalize: boolean Normalize dataset to the range (0,1)
    Read csv file. Subsample if wanted to speed up and normalize all columns except 2 first. 
    '''
    print fname
    df = pd.read_csv(fname)
    # Subsampling
    if step <> 1:
        df = df.iloc[::step,:]
        if verbose:
            print 'step: ', step
    #Normalize 0 to 1 but exclude 2 first columns, that should be X and Y ccordinates
    if normalize:
        df.iloc[:, 2:] = df.apply(lambda x: (x - x.min()) / (x.max() - x.min())).iloc[:, 2:]
    if verbose:
        print df.describe()
    return df
 

def data_to_online_change_point_detection(data, ol=500): 
    '''
    data = 1D array 
    ol = observation likelihood
    Call function and return R and maxes. 
    student's T distributions, constant hazard
    '''
    if verbose:
            print 'Run changepoint detection. OL: ', ol
    R, maxes = online_changepoint_detection(data, 
            partial(constant_hazard, ol), 
            StudentT(0.1, .01, 1, 0))
    if verbose:
            print 'Done!'
    return R, maxes

def run_and_plot(file_names,Nw,step,min_change,ol,plot_param):
    
    for data_file in file_names:    
        df = import_data(data_file, step=step) 
    
        parameters = list(df)
        if verbose:
            print data_file
            print parameters
        changes = pd.DataFrame().reindex_like(df[:])
        changes['X'], changes['Y'] = df['X'], df['Y']

        df['Aus_Amag_g'] = df['Aus_Amag_g'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        R, maxes = data_to_online_change_point_detection(df['Aus_Amag_g'].as_matrix(), ol = ol)
        changes['Aus_Amag_g'].iloc[Nw//2:-Nw//2] = R[Nw,Nw:-1]


                
        fig, ax = plt.subplots(figsize=[18, 9]) #or width_factor*n 
        ax = [plt.subplot(3,1,i+1) for i in range(3)]
        plt.subplots_adjust(hspace=0)

        n = len(df.index)

#        for a in ax:
#            a.set_yticklabels([])
#            a.set_xlim([0,n])
#            a.tick_params(labelleft='off')
        color = 'black'
                    
            
        ax[0].plot(df['Aus_Amag_g'], alpha = 1, c=color)

        sparsity = 5  # for faster display
        ax[1].pcolor(np.array(range(0, len(R[:,0]), sparsity)), 
                            np.array(range(0, len(R[:,0]), sparsity)), 
                            -np.log(R[0:-1:sparsity, 0:-1:sparsity]), 
                            cmap=cm.Greys, vmin=0, vmax=300)
                
        ax[2].plot(R[Nw,Nw:-1], alpha = 1, c=color)
        plt.show()
    
        changes = changes[(changes['Aus_Amag_g'] >= min_change).T.any()]
        changes.to_csv('div/ch_OL_%s_Nw_%s_%s'%(ol, Nw, data_file[-6:]),float_format='%.6f')

        if verbose:
            print 'ch_OL_%s_Nw_%s_%s'%(ol, Nw, data_file[-6:]) , ' stored.'
        gc.collect()
  
in_path = 'data'
file_names = glob.glob('%s/*.csv'%in_path)

#Uncomment to test one file
file_names = ['data/ga_line_12.csv']


verbose = False

#Samples to wait before evaluate
Nw = 80

#Subsample dataset to speed up
step = 1

#Min changes to store to output file
min_change =  1e-8  

#Observation likelehood
ol= 100

plot_param = 'Aus_Amag_g'
    
color = 'k'

run_and_plot(file_names,Nw,step,min_change,ol,plot_param)


