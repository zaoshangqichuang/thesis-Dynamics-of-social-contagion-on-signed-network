# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:08:41 2022

@author: huangguo
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from SIS_sim import SIS_sim
from me_analysis import me_analysis
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
N = 100
k = 10
NetworkType='er'
p_ws = 0.1
alpha1 = 0.5
q_array = [10**a for a in np.linspace(-3,0,100)]
lambda1 = 0.8
lambda2 = 2.5
rho2_df = np.zeros(shape=(100,100))
rho3_df = np.zeros(shape=(100,100))


network={'NetworkType':NetworkType,
         'N':N,
         'k':k,
         'p_ws':p_ws,
         #'p_n':p_n,
         'strategy':'random',
         #'p_ws':p_ws,
         #'p':p,
         'p_n':0.2
    }
param ={'rho0':0.1,
        't0':1000,
        'lambda1':lambda1,
        'lambda2':lambda2,
        #'beta_1':0.2,
        #'lambda2':lambda2,
        #'beta_2':0.2,
        'mu':0.3,
        'p_n':0.2,
        'simplex':2,
        'gamma1':1,
        'gamma2':1,
        'gamma3':1,
        'alpha1':alpha1,
        
        }
#lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
#lambda1_array = np.arange(0.25,2.25,0.25)
sis = me_analysis(network,param)
k1 = 0
for q in q_array:
    sis.p = 0.02
    sis.q = q
    k2 = 0
    sis.beta_1 = 0.2
    for a in np.linspace(-1,1,100):
        sis.gamma1 = 10**a
        sis.gamma2 = 0
        sis.gamma3 = sis.gamma1/10
        sis.cal_p_n()
        sis.cal_lambda_1()
        sis.cal_beta_2()
        sis.cal_lambda_2()
        sis.cal_rho_steady()
        rho2_df[k1,k2]=sis.rho2
        rho3_df[k1,k2]=sis.rho3
        k2 += 1
    k1 += 1
beta_delta=[10**a*sis.beta_1 for a in np.linspace(-6,-1,100)]
np.savez(dir1+"rho_df_q_1",rho2_df = rho2_df,rho3_df=rho3_df,q_array=q_array,beta_delta=beta_delta)
