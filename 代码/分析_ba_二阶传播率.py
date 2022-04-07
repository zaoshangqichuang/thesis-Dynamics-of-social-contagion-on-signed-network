# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:07:56 2022

@author: 201921250004
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
#from SIS_sim import SIS_sim
from BA_me_analysis import BA_me_analysis
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
N = 100
k = 6
NetworkType = 'ba'
network={'NetworkType':NetworkType,
             'N':N,
             'k':k,
            # 'p_ws':p_ws,
             #'p_n':p_n,
             'strategy':'random'
             #'p_ws':p_ws,
             #'p':p,
             #'p_n':p_n
        }
rho_array = np.arange(1,30,2)/100
param ={'rho0':0.5,
            't0':1000,
            'lambda1':0.5,
            'lambda2':1,
            #'beta_1':0.2,
            #'lambda2':lambda2,
            #'beta_2':0.2, 
            'mu':0.1,
            'p_n':0.5,
            'simplex':2,
            'gamma1':1.5,
            'gamma2':0,
            'gamma3':0.5
            #'alpha1':alpha1,    
            }
        #lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
        #lambda1_array = np.arange(0.25,2.25,0.25)
sis = BA_me_analysis(network,param)
k_delta_mean = sis.k_delta_mean*(sis.p*sis.gamma1+(1-sis.p)*sis.q*sis.gamma3)
k_mean_2 = sis.k_po_mean_2
k_mean = sis.k_po_mean
rho_df = np.zeros(shape=(20,20),dtype='float')
k1 = 0




for lambda1 in np.linspace(0.2,0.9,20):
    sis.lambda1 = lambda1
    sis.beta_1 = sis.lambda1*sis.mu/sis.k_po_mean
    k2 = 0
    for lambda2 in np.linspace(0,5,20):
        sis.lambda2 = lambda2
        sis.beta_2=sis.lambda2*sis.mu/(sis.k_delta_mean*(sis.p*sis.gamma1+(1-sis.p)*sis.q*sis.gamma3))
        sis.me_sim()
        print('%.2f'%lambda1,'%.2f'%lambda2,sis.rho)
        rho_df[k1,k2] = sis.rho
        k2 += 1
    k1 += 1
#np.savez(dir1+"chuanbo_ba",rho_df=rho_df,lambda1=np.linspace(0.2,0.9,20),lambda2=np.linspace(0,5,20))
lambda1_array = np.linspace(0.2,0.9,20)
lambda2_array = np.linspace(0,5,20)
lambda2_linjie = []
for k1 in range(20):
    pre = 0 
    for k2 in range(20):
        if rho_df[k1][k2]>0:
            lambda2_linjie.append(pre)
            break
        else:
            pre = lambda2_array[k2]
plt.plot(lambda1_array,lambda2_linjie,'o')            
