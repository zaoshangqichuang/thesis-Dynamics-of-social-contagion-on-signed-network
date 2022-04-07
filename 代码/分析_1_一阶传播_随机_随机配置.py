# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:29:10 2022

@author: huangguo
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from SIS_sim import SIS_sim
from me_analysis import me_analysis
N = 100
k = 10
p = 0.1
#p_n = 0.2
p_ws = 0.5
dir1 = 'npz文件\\'
dir2 = 'png文件\\'
for NetworkType in ['er','ws','re']:
    for nn in range(1,11):
        for lambda1 in np.arange(0.5,3.5,0.5):
            network={'NetworkType':NetworkType,
             'N':N,
             'k':k,
             'p_ws':p_ws,
             #'p_n':p_n,
             'strategy':'random'
             #'p_ws':p_ws,
             #'p':p,
            # 'p_n':p_n
            }
            param ={'rho0':0.1,
                        't0':1000,
                        'lambda1':lambda1,
                        #'beta_1':0.2,
                        #'lambda2':lambda2,
                        #'beta_2':0.2,
                        'mu':0.3,
                        'p_n':0.5,
                        'simplex':1,
                        'gamma1':1,
                        'gamma2':1,
                        'alpha1':0
                        }
            sis = SIS_sim(network,param)
            sis.run()
            pickle.dump(sis, open(dir1+"random1_"+NetworkType+"_"+str(round(lambda1,1))+'_'+str(nn)+".pkl", 'wb'))
            print("random1_"+NetworkType+"_"+str(round(lambda1,1))+'_'+str(nn))
