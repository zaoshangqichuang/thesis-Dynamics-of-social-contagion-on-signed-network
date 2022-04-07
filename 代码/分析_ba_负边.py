# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 22:13:40 2022

@author: 201921250004
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
#from SIS_sim import SIS_sim
from BA_me_analysis import BA_me_analysis
from me_analysis import me_analysis
#N = 100
#k = 6
#NetworkType = 'ba'
def sim_ba_p(NetworkType,k,n):
    network={'NetworkType':NetworkType,
                 'N':100,
                 'k':k,
                 'p_ws':0.5,
                 #'p_n':p_n,
                 'strategy':'random'
                 #'p_ws':p_ws,
                 #'p':p,
                 #'p_n':p_n
            }
    dir1 = 'npzfile\\'
    dir2 = 'pngfile\\'
    p_n_array = np.linspace(0,1,10)[:-1]
    rho_df = np.zeros(shape=(10,6,5),dtype='float')
    k1 = 0
    for p_n in p_n_array:
        param ={'rho0':0.5,
                    't0':1000,
                    'lambda1':0.5,
                    'lambda2':3,
                    #'beta_1':0.2,
                    #'lambda2':lambda2,
                    #'beta_2':0.2, 
                    'mu':0.1,
                    'p_n':p_n,
                    'simplex':2,
                    'gamma1':1.5,
                    'gamma2':0,
                    'gamma3':0.5
                    #'alpha1':alpha1,    
                    }
                #lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
                #lambda1_array = np.arange(0.25,2.25,0.25)
        if NetworkType == 'ba':
            sis = BA_me_analysis(network,param)
        else:
            sis = me_analysis(network,param)
        gamma1_array =list(np.linspace(1,2,5))+[0]
        gamma3_array = np.linspace(0,1,5)
        k2 = 0
        for gamma1 in gamma1_array:
            sis.gamma1 = gamma1
            k3 = 0
            for gamma3 in gamma3_array:
                sis.gamma3 = gamma3
                
                if NetworkType == 'ba':
                    sis.me_sim()
                    rho_df[k1][k2][k3] = sis.rho
                else:
                    sis.run()
                    rho_df[k1][k2][k3] = sis.rho_list[-1]
                print(k1,k2,k3)
                k3 += 1
            k2 += 1
        k1 += 1
    np.savez(dir1+"p_n_ba_sim"+str(n)+'_'+str(k)+'_'+str(NetworkType),p_n_array=p_n_array,gamma1_array=gamma1_array,
             gamma3_array=gamma3_array,rho_df=rho_df)
    
for NetworkType in ['ws']:
    for k in [6,11,16]:
        for n in range(1,11):
            sim_ba_p(NetworkType,k,n)








