# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:42:56 2022

@author: 201921250004
"""
from BA_me_analysis import BA_me_analysis
from me_analysis import me_analysis
from SignedNetwork import SignedNetwork
import pickle
import numpy as np
import matplotlib.pyplot as plt
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'

network={'NetworkType':'ba',
         'N':100,
         'k':3,
         'p_n':0.5,
         'p_ws':0.5,
         'strategy':'prior'
    }
for nn in range(6,11):
    for alpha in np.arange(0,1,0.1):
         param ={
            'rho0':0.1,
            't0':1000,
            'lambda1':1,
            'mu':0.3,
            'p_n':0.5,
            'simplex':1,
            'gamma1':1,
            'gamma2':1,
            'alpha1':alpha
            }
         sn = BA_me_analysis(network,param)
         sn.run()
         pickle.dump(sn, open(dir1+"ba_prior\\"+"simplex1_ba_"+str(nn)+"_"+str(round(alpha,2))+".pkl", 'wb'))
     
network={'NetworkType':'ws',
         'N':100,
         'k':3,
         'p_n':0.5,
         'p_ws':0.5,
         'strategy':'prior'
    }
for nn in range(6,11):
    for alpha in np.arange(0,1,0.1):
         param ={
            'rho0':0.1,
            't0':1000,
            'lambda1':1.25,
            'mu':0.3,
            'p_n':0.5,
            'simplex':1,
            'gamma1':1,
            'gamma2':1,
            'alpha1':1
            }
         sn = me_analysis(network,param)
         sn.run()
         pickle.dump(sn, open(dir1+"ba_prior\\"+"simplex1_ws_"+str(nn)+"_"+str(round(alpha,2))+".pkl", 'wb'))