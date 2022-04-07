# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 00:17:34 2022

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
p_n_array = np.linspace(0.1,0.5,100)
lambda1 = 0.8
lambda2 = 2.5
rho2_df = np.zeros(shape=(100,100))
rho3_df = np.zeros(shape=(100,100))
k1 = 0
for p_n in p_n_array:
    network={'NetworkType':NetworkType,
             'N':N,
             'k':k,
             'p_ws':p_ws,
             #'p_n':p_n,
             'strategy':'random',
             #'p_ws':p_ws,
             #'p':p,
             'p_n':p_n
        }
    param ={'rho0':0.1,
            't0':1000,
            'lambda1':lambda1,
            'lambda2':lambda2,
            #'beta_1':0.2,
            #'lambda2':lambda2,
            #'beta_2':0.2,
            'mu':0.3,
            'p_n':p_n,
            'simplex':2,
            'gamma1':1,
            'gamma2':1,
            'gamma3':1,
            'alpha1':alpha1,
            
            }
    #lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
    #lambda1_array = np.arange(0.25,2.25,0.25)
    sis = me_analysis(network,param)
    k2 = 0
    sis.beta_1 = 0.0001
    for gamma1 in np.linspace(1000,1/sis.beta_1,100):
        sis.gamma1 = gamma1
        sis.gamma2 = 0
        sis.gamma3 = 1
        sis.cal_lambda_1()
        sis.cal_beta_2()
        sis.cal_lambda_2()
        sis.cal_rho_steady()
        rho2_df[k1,k2]=sis.rho2
        rho3_df[k1,k2]=sis.rho3
        k2 += 1
    k1 += 1
beta_delta=[gamma1*sis.beta_1 for gamma1 in np.linspace(10,1/sis.beta_1,100)]
np.savez(dir1+"rho_df",rho2_df = rho2_df,rho3_df=rho3_df,p_n_array=p_n_array,beta_delta=beta_delta)



#初始密度的相变现象
network={'NetworkType':NetworkType,
             'N':N,
             'k':k,
             'p_ws':p_ws,
             #'p_n':p_n,
             'strategy':'random',
             #'p_ws':p_ws,
             #'p':p,
             'p_n':p_n
        }
param ={'rho0':0.1,
        't0':1000,
        'lambda1':lambda1,
        'lambda2':lambda2,
        #'beta_1':0.2,
        #'lambda2':lambda2,
        #'beta_2':0.2,
        'mu':0.3,
        'p_n':2/6,
        'simplex':2,
        'gamma1':1,
        'gamma2':1,
        'gamma3':1,
        'alpha1':alpha1,
        
        }
    #lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
    #lambda1_array = np.arange(0.25,2.25,0.25)
sis = me_analysis(network,param)
sis.gamma1 = 2
sis.gamma2 = 1
sis.gamma3 = 0.25
sis.cal_beta_2()
sis.cal_lambda_2()
sis.cal_rho_steady()
rho_steady_me_list = []
for rho0 in [0.05,0.1,0.15,0.16,0.2,0.5,0.7]:
    sis.rho0 = rho0
    rho_steady_me = sis.me_sim()
    rho_steady_me_list.append(rho_steady_me)
np.savez(dir1+"rho_steady_me",rho_steady_me_list=rho_steady_me_list,rho0_list=[0.05,0.1,0.2,0.5,0.7])
