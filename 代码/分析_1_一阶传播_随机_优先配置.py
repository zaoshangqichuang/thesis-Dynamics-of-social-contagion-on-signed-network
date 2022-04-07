# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:45:44 2022

@author: huangguo
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from SIS_sim import SIS_sim
from me_analysis import me_analysis
NetworkType = 'ws'
N = 100
k = 6
p = 0.1
#p_n = 0.2
p_ws = 0.5
dir1 = 'npz文件\\'
dir2 = 'png文件\\'
network={'NetworkType':NetworkType,
         'N':N,
         'k':k,
         'p_ws':p_ws,
         #'p_n':p_n,
         'strategy':'prior'
         #'p_ws':p_ws,
         #'p':p,
        # 'p_n':p_n
    }
lambda1 = 1.5
for nn in range(1,5):
    for alpha1 in np.arange(0,3.25,0.25)/3:
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
                    'alpha1':alpha1
                    }
        sis = SIS_sim(network,param)
        sis.run()
        pickle.dump(sis, open(dir1+"prior_"+str(round(alpha1,2))+"_"+str(nn)+".pkl", 'wb'))
        print("prior_"+str(round(alpha1,2))+"_"+str(nn))
'''     
    plt.plot(sis.rho_list)
    net = sis.signedNetwork
    sis.get_degree() 
'''
'''
import pickle
pickle.dump(sis, open(dir1+"prior_6.pkl", 'wb'))  # 序列化
sis_load = pickle.load(open(dir1+'myclass.pkl', 'rb'))  # 反序列化
plt.plot(sis_load.rho_list)
'''

