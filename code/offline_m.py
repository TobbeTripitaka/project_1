from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import sys
import glob
import getopt

#from numba import jit

import cProfile
from functools import partial
from scipy.special import gammaln, multigammaln
from scipy.misc import comb
from decorator import decorator

# This makes the code compatible with Python 3
# without causing performance hits on Python 2
#TO DO: implement print function!
try:
    xrange
except NameError:
    xrange = range




try:
    from sselogsumexp import logsumexp
except ImportError:
    from scipy.misc import logsumexp
    print("Use scipy logsumexp().")
else:
    print("Use SSE accelerated logsumexp().")

##########################################################################################


##########################################################################################

def main(argv):
    '''
    Get command line arguments. Name input dataset.
    
    Parameters from shell
    ----------	
    Returns
    -------
    Returns file_name (for labels etc), subsampling steplenght, truncation value, 
    how many cordniated in x-axis, width factor for figue whidt in relation to datalenght
    min_change to save row in output datafile, coordinate names in indata header
    '''
    def str2bool(thoughts): 
        '''
        Setting in string variable to boolean
        '''
        if type(thoughts) == bool:
            return thoughts
        else:   
            return thoughts.lower() in ("yes", "y", "true", "t", "1", "oui")
     
    # Set defaults: 
    step = 50              #Subsampling for speed
    trunc = -100            #Truncate
    verbose = True          #Talk to me?
    n_ticks = 25            #How many tics in the plot    
    width_factor = 0.025    #Width factor of plot, unless fixed
    min_change =  1e-6      #Min changes to store to output file
    cordinates=['X', 'Y']   #Header names for import data
    geo=['DESC_','NAME']    #Header names for geo data
    no_use = ''           #Excluded datasets
    
    
    
    file_name = sys.stdin.name
    
    try:
        opts, args = getopt.getopt(argv,"i:v:t:s:n:",["input=", "verbose=", "truncate=", "sub_sampling=", "no_use="],)
    except getopt.GetoptError:
        print 'offline_m.py -i <input csv-file> -v <verbose> -t <truncate> -s <subsampling -n <no use>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'Where is Your God Now?'
            sys.exit()
        elif opt in ("-i", "--in_file"): 
            file_name = arg
        elif opt in ("-v", "--verbose"): 
            verbose = arg
        elif opt in ("-t", "--truncate"): 
            trunc = arg
        elif opt in ("-s", "--sub_sampling"): 
            step = arg           
        elif opt in ("-n", "--no_use"): 
            no_use = arg 
    return file_name, int(step), int(trunc), str2bool(verbose), int(n_ticks), width_factor, min_change, cordinates, geo, no_use.split(' ')
    

#@jit
def _dynamic_programming(f, *args, **kwargs):
    if f.data is None:
        f.data = args[0]

    if not np.array_equal(f.data, args[0]):
        f.cache = {}
        f.data = args[0]

    try:
        f.cache[args[1:3]]
    except KeyError:
        f.cache[args[1:3]] = f(*args, **kwargs)
    return f.cache[args[1:3]]

#@jit
def dynamic_programming(f):
    f.cache = {}
    f.data = None
    return decorator(_dynamic_programming, f)

def offline_changepoint_detection(data, prior_func,
                                  observation_log_likelihood_function,
                                  truncate=-np.inf,
                                  verbose = True):
    """Compute the likelihood of changepoints on data.

    Keyword arguments:
    data                                -- the time series data
    prior_func                          -- a function given the likelihood of a changepoint given the distance to the last one
    observation_log_likelihood_function -- a function giving the log likelihood
                                           of a data part
    truncate                            -- the cutoff probability 10^truncate to stop computation for that changepoint log likelihood

    P                                   -- the likelihoods if pre-computed
    """

    
    n = len(data)
    Q = np.zeros((n,))
    g = np.zeros((n,))
    G = np.zeros((n,))
    P = np.ones((n, n)) * -np.inf

    # save everything in log representation
    for t in range(n):        
        g[t] = np.log(prior_func(t))
        if t == 0:
            G[t] = g[t]
        else:
            G[t] = np.logaddexp(G[t-1], g[t])

    P[n-1, n-1] = observation_log_likelihood_function(data, n-1, n)
    Q[n-1] = P[n-1, n-1]

    for t in reversed(range(n-1)):
        P_next_cp = -np.inf  # == log(0)
      
        for s in range(t, n-1):
            P[t, s] = observation_log_likelihood_function(data, t, s+1)

            # compute recursion
            summand = P[t, s] + Q[s + 1] + g[s + 1 - t]
            P_next_cp = np.logaddexp(P_next_cp, summand)

            # truncate sum to become approx. linear in time (see
            # Fearnhead, 2006, eq. (3))
            if summand - P_next_cp < truncate:
                break

        P[t, n-1] = observation_log_likelihood_function(data, t, n)

        # (1 - G) is numerical stable until G becomes numerically 1
        if G[n-1-t] < -1e-15:  # exp(-1e-15) = .99999...
            antiG = np.log(1 - np.exp(G[n-1-t]))
        else:
            # (1 - G) is approx. -log(G) for G close to 1
            antiG = np.log(-G[n-1-t])

        Q[t] = np.logaddexp(P_next_cp, P[t, n-1] + antiG)

    Pcp = np.ones((n-1, n-1)) * -np.inf
    for t in range(n-1):
        Pcp[0, t] = P[0, t] + Q[t + 1] + g[t] - Q[0]
        if np.isnan(Pcp[0, t]):
            Pcp[0, t] = -np.inf
    for j in range(1, n-1):

        for t in range(j, n-1):
            tmp_cond = Pcp[j-1, j-1:t] + P[j:t+1, t] + Q[t + 1] + g[0:t-j+1] - Q[j:t+1]
            Pcp[j, t] = logsumexp(tmp_cond.astype(np.float32))
            if np.isnan(Pcp[j, t]):
                Pcp[j, t] = -np.inf

    return Q, P, Pcp

@dynamic_programming
def gaussian_obs_log_likelihood(data, t, s):
    s += 1
    n = s - t
    mean = data[t:s].sum(0) / n

    muT = (n * mean) / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = 1 + 0.5 * ((data[t:s] - mean) ** 2).sum(0) + ((n)/(1 + n)) * (mean**2 / 2)
    scale = (betaT*(nuT + 1))/(alphaT * nuT)

    # splitting the PDF of the student distribution up is /much/ faster.
    # (~ factor 20) using sum over for loop is even more worthwhile
    prob = np.sum(np.log(1 + (data[t:s] - muT)**2/(nuT * scale)))
    lgA = gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT/2)

    return np.sum(n * lgA - (nuT + 1)/2 * prob)

#@jit
def ifm_obs_log_likelihood(data, t, s):
    '''Independent Features model from xuan et al'''
    s += 1
    n = s - t
    x = data[t:s]
    if len(x.shape)==2:
        d = x.shape[1]
    else:
        d = 1
        x = np.atleast_2d(x).T

    N0 = d          # weakest prior we can use to retain proper prior
    V0 = np.var(x)
    Vn = V0 + (x**2).sum(0)

    # sum over dimension and return (section 3.1 from Xuan paper):
    return d*( -(n/2)*np.log(np.pi) + (N0/2)*np.log(V0) - \
        gammaln(N0/2) + gammaln((N0+n)/2) ) - \
        ( ((N0+n)/2)*np.log(Vn) ).sum(0)

#@jit
def fullcov_obs_log_likelihood(data, t, s):
    '''Full Covariance model from xuan et al'''
    s += 1
    n = s - t
    x = data[t:s]
    if len(x.shape)==2:
        dim = x.shape[1]
    else:
        dim = 1
        x = np.atleast_2d(x).T

    N0 = dim          # weakest prior we can use to retain proper prior
    V0 = np.var(x)*np.eye(dim)
    
    # Improvement over np.outer
    # http://stackoverflow.com/questions/17437523/python-fast-way-to-sum-outer-products
    # Vn = V0 + np.array([np.outer(x[i], x[i].T) for i in xrange(x.shape[0])]).sum(0)
    Vn = V0 + np.einsum('ij,ik->jk', x, x)

    # section 3.2 from Xuan paper:
    return -(dim*n/2)*np.log(np.pi) + (N0/2)*np.linalg.slogdet(V0)[1] - \
        multigammaln(N0/2,dim) + multigammaln((N0+n)/2,dim) - \
        ((N0+n)/2)*np.linalg.slogdet(Vn)[1]

#@jit        
def const_prior(r, l):
    return 1/(l)

#@jit
def geometric_prior(t, p):
    return p * ((1 - p) ** (t - 1))

#@jit
def neg_binominal_prior(t, k, p):
    return comb(t - k, k - 1) * p ** k * (1 - p) ** (t - k)

def import_data(file_name, header_rows=1, step=1, normalize=False, verbose=False, no_norm=['X', 'Y'], parameters=[]):
    '''
    fname: string, file to read
    header rows: integer 
    step: integer Subsampling of dataset to increase speed
    normalize: boolean Normalize dataset to the range (0,1)
    Read csv file. Subsample if wanted to speed up and normalize all columns cordinates
    '''
    df = pd.read_csv(file_name)
    include = [col for col in df.columns if col not in no_norm]


    print 'no_norm', no_norm
    print 'include', include

    if step <> 1:
        df[include] = df[include].rolling(step, center=True, min_periods=0).mean()
        df = df.iloc[::step,:]
        if verbose:
            print 'step: ', step
    
    if normalize:
        df[include] = df[include].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    if verbose:
        print 'File:', file_name, 'Subsampling: ', step
    if parameters == []:
        parameters = list(df.drop(cordinates + geo, axis=1))
    # Define parameters 
    return df, parameters

# Import data from standard in generate array and define no-data
#in_file = 'data/ga_vpt_line_15.csv'
in_file = sys.stdin


# Read command line arguments
if __name__ == "__main__":
    file_name, step, trunc, verbose, n_ticks, width_factor, min_change, cordinates, geo, no_use = main(sys.argv[1:])

exclude_list = geo + cordinates

print 'exclude list', exclude_list

df, parameters = import_data(in_file, 
                    step=step, 
                    normalize=True, 
                    verbose=verbose, 
                    no_norm= exclude_list)



print 'parameters', parameters

nansum = df.isnull().sum().sum()
if nansum>0:
    print 'Warning, contains %s Nans! Nans are interpolated.' %nansum #in_file.name

parameters_to_process = [x for x in parameters if x not in no_use]

data = df[parameters_to_process].interpolate(method='linear').as_matrix()

n, p = np.shape(data)

print n, p

# Some fix: Change order of some datasets to plot the radiometry on top of each other
col_list = list(df)
col_list[3], col_list[4] = col_list[4], col_list[3]
df.columns = col_list
    
    
#FIX ADJUST HIGHT TO NUMBER OF PARAMETERS    
fig, ax = plt.subplots(figsize=[18, 9]) #or width_factor*n 
ax = [plt.subplot(2,1,i+1) for i in range(2)]
plt.subplots_adjust(hspace=0)

for a in ax:
    a.set_yticklabels([])
    a.set_xlim([0,n])
    a.tick_params(labelleft='off')
 
for i in range(p):
    h_pos = 1.5*i
    ax[0].set_yticks([])
    label = parameters_to_process[i]
    ax[0].text(-2, h_pos, label, ha = "right", fontsize=14)
    ax[0].axhspan(h_pos, h_pos+1, facecolor='g', alpha=0.1)
    ax[0].plot(h_pos+data[:,i], c='k', lw=2)


#y_values_masked = np.ma.masked_where(df[parameters[i]].isnull(), y_values)

ax[0].spines['bottom'].set_visible(False)

Q_full, P_full, Pcp_full = offline_changepoint_detection(data,
                                                partial(const_prior, 
                                                l=(len(data)+1)),
                                                fullcov_obs_log_likelihood, 
                                                truncate=trunc)

    
P = np.exp(Pcp_full).sum(0)
P = np.append(P, P[-1]) #UGLY FIX!!!
  
boundary_id = 1
list_of_boundaries = ['From left: ', df.iloc[1]['DESC_']]
for pt in xrange(n-1):
    if df.iloc[pt]['DESC_'] != df.iloc[pt+1]['DESC_']:
        ax[1].axvline(x=pt, ymax = 0.85, color='green')
        ax[1].text(pt, 0.9, 
            boundary_id,
            fontsize=12,
            horizontalalignment = 'center',
            style='italic', 
            color='green')      
        list_of_boundaries.extend((', %s: '%boundary_id, df.iloc[pt+1]['DESC_']))
        boundary_id += 1  
ax[1].set_ylabel('Pcp', color = 'black', fontsize=20)

ax[1].plot(P, c='k', lw=2)
ax[1].spines['top'].set_visible(False)
ax[1].set_ylim([0,1])
    
# Print coordinates along x-axis   
tick_jumps = n//n_ticks
lon_labels = [ '%.3f' % elem for elem in df['X'].iloc[::tick_jumps]]
lat_labels = [ '%.3f' % abs(elem) for elem in df['Y'].iloc[::tick_jumps]]
pos_labels = ['%s E\n%s S' % elem for elem in zip(lon_labels, lat_labels)]
label_ticks = range(0, n, tick_jumps)
    
plt.xticks(label_ticks, 
        pos_labels, 
        size=10, 
        rotation=45, 
        ha= 'right', 
        color='black', 
        alpha=0.8)
plt.setp(ax[1].get_xticklabels()[::2], visible=False)

#plt.show()

# save figure
#plt.savefig('test.pdf', format='pdf',bbox_inches='tight') 
plt.savefig(sys.stdout, format='pdf',bbox_inches='tight')


# Save .csv file
change_prob = pd.DataFrame(P, index=df.index.values, columns=['Pcp'])
change_prob[['X', 'Y']] = df[['X', 'Y']]
change_prob = change_prob[(change_prob['Pcp'] >= min_change)]
change_prob.to_csv(sys.stdout, float_format='%.6f')#'result/data/P_%s_S%s_T%s.csv'%(file_name, step, trunc), float_format='%.6f')
    
fig1=plt.plot(Q_full) 
#plt.savefig('fig/Q_full.pdf', format='pdf')


fig2=plt.plot(P_full)    
#plt.savefig('fig/P_full.pdf', format='pdf',bbox_inches='tight')

np.savetxt('tex/captions/%soff_m_caption.tex'%file_name, [''.join(list_of_boundaries).replace('&', "and")], fmt='%s')

    
# Clean up
plt.clf()
gc.collect()
