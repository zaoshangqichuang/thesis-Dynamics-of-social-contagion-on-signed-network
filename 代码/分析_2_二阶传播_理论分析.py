# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:00:35 2022

@author: huangguo
"""
#二阶传播
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
network={'NetworkType':NetworkType,
         'N':N,
         'k':k,
         'p_ws':p_ws,
         #'p_n':p_n,
         'strategy':'random',
         #'p_ws':p_ws,
         #'p':p,
         'p_n':2/6
    }
lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
#lambda1_array = np.arange(0.25,2.25,0.25)
lambda2_array = np.arange(0.5,3.5,0.5)
param_list=[]
kk = 1

for lambda1 in lambda1_array:
    for lambda2 in lambda2_array:
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
                'alpha1':alpha1
                }
        sis = SIS_sim(network,param)
        sis.run()
        pickle.dump(sis, open(dir1+"2simplex_new_"+str(round(lambda1,2))+"_"+str(round(lambda2,1))+".pkl", 'wb'))
        print("lambda1:{0:.2f},lambda2:{1:.1f}".format(lambda1,lambda2)+",进度{0}/{1}".format(kk,len(lambda1_array)*len(lambda2_array)))
        param_list.append((sis.lambda1,sis.lambda2,sis.rho_steady_from_sim,sis.rho_steady))
        kk += 1
        
plot_dict1={}
me_dict1={}
for lambda2 in lambda2_array:
    plot_dict1[str(round(lambda2,1))]=[]
    me_dict1[str(round(lambda2,1))]=[]
    for lambda1 in lambda1_array:
        sis = pickle.load(open(dir1+"2simplex_"+str(round(lambda1,2))+"_"+str(round(lambda2,1))+".pkl", 'rb'))
        plot_dict1[str(round(lambda2,1))].append(sis.rho_steady_from_sim)
        me_dict1[str(round(lambda2,1))].append(sis.rho_steady)

        
        
        
'''
lambda1_array_me = np.linspace(0.25,2.25,50)
lambda2_array = np.arange(0.5,3.5,0.5)
N = 100
k = 6
NetworkType='er'
p_ws = 0.1
alpha1 = 0.5
me_dict1={}
network={'NetworkType':NetworkType,
         'N':N,
         'k':k,
         'p_ws':p_ws,
         #'p_n':p_n,
         'strategy':'random',
         #'p_ws':p_ws,
         #'p':p,
         'p_n':2/6
    }

for lambda2 in lambda2_array:
    me_dict1[str(round(lambda2,1))]=[]
    for lambda1 in lambda1_array_me:
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
                'alpha1':alpha1
                }
        sis = SIS_sim(network,param)
        me_dict1[str(round(lambda2,1))].append(sis.rho_steady)

import numpy as np
np.savez(dir1+"me_dict1"+".npz",me_dict1=me_dict1)

'''

