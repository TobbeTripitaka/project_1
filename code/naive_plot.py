import pandas as pd
import numpy as np
import sys, gc
import matplotlib.pyplot as plt

def normalize(data):
    try:
        return data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    except: 
        return (data - data.min()) / (data.max() - data.min())

step = 1

df = pd.read_csv(sys.stdin)
df = df.iloc[::step,:]

geo_data = ['Aus_u_ga@2', 'Aus_Ka_ga@', 'Aus_Amag_g', 'Aus_Th_ga@', 'Aus_BA267_']
data = normalize(df[geo_data])
size = df.shape

print 'size: ', size


fig, ax = plt.subplots(figsize=[15, 15])

ax = [plt.subplot(6,1,i+1) for i in range(6)]

for a in ax:
    a.set_yticklabels([])
    a.set_xticklabels([])
    a.set_xlim([0,size[0]])
    a.set_ylim([0,1])

#Plot data
ax[0].plot(data, color='black', alpha=0.4)
ax[0].set_ylabel('Normalized data')

 
#Plot rolling mean
window = 120
rm = data.rolling(window = window, center=True).mean()
ax[1].plot(normalize(rm), color='green', alpha=0.4)
mean = rm.mean(axis=1, skipna=None)
ax[1].plot(normalize(mean), color='black', alpha=1, lw=1)
ax[1].set_ylabel('Rolling mean = %s'%window)

#Rolling mean diff
window = 120
rmd = data.diff(periods=3, axis=0).rolling(window=window, center=True).mean().abs()
ax[2].plot(normalize(rmd), color='green', alpha=0.4)
mean = rmd.mean(axis=1, skipna=None)
ax[2].plot(normalize(mean), color='black', alpha=1, lw=1)
ax[2].set_ylabel('Diff mean')


#Rolling variance
window = 100
rv = data.rolling(window=window, center=True).std()**2
ax[3].plot(normalize(rv), color='green', alpha=0.4)
mean = rv.mean(axis=1, skipna=None)
ax[3].plot(normalize(mean), color='black', alpha=1, lw=1)
ax[3].set_ylabel('Rolling variance')


#Rolling variance diff
rvd = rv.diff(periods=3, axis=0).abs()
ax[4].plot(normalize(rvd), color='green', alpha=0.4)
mean = rvd.mean(axis=1, skipna=None)
ax[4].plot(normalize(mean), color='black', alpha=1, lw=1)
ax[4].set_ylabel('Diff variance')


boundary_id = 1
list_of_boundaries = ['From left: ', df.iloc[1]['DESC_']]

ax[5].plot(mean, color='green', alpha=0)

for pt in xrange(size[0]-1):

    if df.iloc[pt]['DESC_'] != df.iloc[pt+1]['DESC_']:
        ax[5].axvline(x=pt, color='green', ymax = 0.85)
        ax[5].text(pt, 0.9, 
            boundary_id,
            fontsize=12,
            horizontalalignment = 'center',
            style='italic', 
            color='green')     
        list_of_boundaries.extend((', %s: '%boundary_id, df.iloc[pt+1]['DESC_']))
#        
        boundary_id += 1  
ax[5].set_ylabel('Geological domains')          

#plt.show()
      
np.savetxt('tex/captions/naive_caption.tex', [''.join(list_of_boundaries).replace('&', "and")], fmt='%s')

plt.savefig(sys.stdout, format='pdf',bbox_inches='tight')

# Clean up
plt.clf()
gc.collect()


