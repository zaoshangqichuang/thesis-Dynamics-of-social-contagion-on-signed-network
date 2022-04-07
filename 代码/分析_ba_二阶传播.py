# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 00:13:00 2022

@author: 201921250004
"""
from BA_me_analysis import BA_me_analysis
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
beta_2_list = np.linspace(0.01,1/1.5,5)
viridis = cm.get_cmap('viridis', 5)
colors = viridis(np.linspace(0.3, 1, 5))
k = 0
#fig = plt.figure()
#ax = fig.add_subplot(111)
for nn in range(1,11):
    for beta_2 in beta_2_list:
        network={'NetworkType':'ba',
             'N':100,
             'k':6,
             'p_n':0.2,
             'strategy':'random'
        }
        param ={
        'rho0':0.1,
        't0':1000,
        'lambda1':1,
        'mu':0.3,
        'p_n':0.2,
        'simplex':2,
        'gamma1':1.5,
        'gamma2':0.5,
        'alpha1':0.5,
        'beta_2':beta_2
        }
        ba0 = BA_me_analysis(network,param)
        #rho_list_ba = []
        for lambda1 in np.linspace(0.1,1,10):
            ba0.lambda1 = lambda1
            ba0.beta_1 = ba0.lambda1*ba0.mu/ba0.k_po_mean
            ba0.me_sim()
            #rho_list_ba.append(ba0.rho)
            pickle.dump(ba0, open(dir1+"simplex_2_ba\\"+"simplex2_ba_"+str(round(beta_2,2))+"_"+str(nn)+"_"+str(round(lambda1,2))+".pkl", 'wb'))
    #ax.plot(np.linspace(0.1,1,10),rho_list_ba,'o-',color=colors[k],label='$\\beta_{\\Delta+}$'+'={:.2f}'.format(round(beta_2,2)))
    #k += 1
#fig.legend()