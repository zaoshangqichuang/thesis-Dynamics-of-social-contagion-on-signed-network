# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:30:58 2022

@author: 201921250004
"""
from BA_me_analysis import BA_me_analysis
import pickle
import numpy as np
import matplotlib.pyplot as plt
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
for nn in range(1,11):
    for p_n in np.linspace(0,0.9,20):
        network={'NetworkType':'ba',
         'N':100,
         'k':3,
         'p_n':p_n,
         'p_ws':0.5,
         'strategy':'random'
    }
        param ={
        'rho0':0.1,
        't0':1000,
        'lambda1':1,
        'mu':0.3,
        'p_n':p_n,
        'simplex':1,
        'gamma1':1,
        'gamma2':1,
        'alpha1':0.5
        }
        ba0 = BA_me_analysis(network,param)
        for lambda1 in np.linspace(0.65,1.5,20):
            ba0.lambda1 = lambda1
            ba0.beta_1 = ba0.lambda1*ba0.mu/ba0.k_po_mean
            ba0.me_sim()
            pickle.dump(ba0, open(dir1+"ba_simplex_1\\"+"simplex1_ba_"+str(round(p_n,2))+"_"+str(nn)+"_"+str(round(lambda1,2))+".pkl", 'wb'))
'''
for m in range(1,7):
    for nn in range(1,11):
        for p_n in np.linspace(0,0.9,20):
            param ={
            'rho0':0.1,
            't0':1000,
            'beta_1':0.3,
            'mu':0.3,
            'p_n':p_n,
            'simplex':1,
            'gamma1':1,
            'gamma2':1,
            'alpha1':0.5
            }
            network={'NetworkType':'ba',
             'N':100,
             'k':m,
             #'p_n':0.2,
             'strategy':'random'
             }
            ba0 = BA_me_analysis(network,param)
            ba0.me_sim()
            pickle.dump(ba0, open(dir1+"new_simplex_bam\\"+"simplex1_ba_"+str(round(p_n,2))+"_"+str(nn)+"_"+str(round(m,0))+".pkl", 'wb'))
'''

'''
fig = plt.figure()
ax1= fig.add_subplot(121)
hmf_list = []
for p_n in np.linspace(0,1,10):
    steady_list = []
    for lambda1 in np.linspace(0.5,1,10):
        ba0 = pickle.load(open(dir1+"simplex1_ba_"+str(round(p_n,2))+"_"+str(round(lambda1,2))+".pkl",'rb'))
        steady_list.append(ba0.rho)
    ax1.plot(np.linspace(0.5,1,10),steady_list)
    hmf_list.append(ba0.hmf)
ax2 = fig.add_subplot(122)
ax2.plot(np.linspace(0,1,10),hmf_list,'o')
'''