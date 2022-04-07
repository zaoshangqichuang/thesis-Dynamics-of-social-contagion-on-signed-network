# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:11:09 2022

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
#p_n = 0.2
p_ws = 0.5
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
network={'NetworkType':NetworkType,
         'N':N,
         'k':k,
         'p_ws':p_ws,
         #'p_n':p_n,
         'strategy':'prior',
         'p_n':0.5
         #'p_ws':p_ws,
         #'p':p,
        # 'p_n':p_n
    }
lambda1 = 1.5
lambda2 = 3
param ={'rho0':0.1,
                    't0':500,
                    'lambda1':lambda1,
                    'lambda2':lambda2,
                    #'beta_1':0.2,
                    #'lambda2':lambda2,
                    #'beta_2':0.2,
                    'mu':0.3,
                    'p_n':0.5,
                    'simplex':2,
                    'gamma1':1,
                    'gamma2':1,
                    'alpha1':0.5
                    }
sis = SIS_sim(network,param)


for nn in range(1,11):
    for alpha1 in np.arange(0,3,0.25)/3:
        sis.alpha1 = alpha1
        sis.add_Sign(sis.p_n,sis.alpha1)
        sis.run()
        pickle.dump(sis, open(dir1+"prior_simplex2_"+str(round(alpha1,2))+"_"+str(nn)+".pkl", 'wb'))
        print("prior_simplex_2_sparse"+str(round(alpha1,2))+"_"+str(nn))
'''
alpah1_array = np.arange(0,25,1)/3
rho_steady_list = [np.zeros(4) for i in range(13)]
k1 = 0
for nn in range(1,5):
    k2 = 0
    for alpha1 in np.arange(0,3.25,0.25)/3:
        sis_load = pickle.load(open(dir1+"prior_simplex2_"+str(round(alpha1,2))+"_"+str(nn)+".pkl", 'rb'))
        rho_steady_list[k2][k1]=sis_load.rho_steady_from_sim*1.6
        k2 += 1
    k1 += 1
plt.plot([l.mean() for l in rho_steady_list])
'''
alpah1_array = np.arange(0,3,0.25)/3
rho_steady_list = []
for alpha1 in alpah1_array:
    sis_load = pickle.load(open(dir1+"prior_simplex2_"+str(round(alpha1,2))+"_"+str(1)+".pkl", 'rb'))
    rho_steady_list.append(sis_load.rho_steady_from_sim*1.5)
plt.plot(alpah1_array,rho_steady_list,'o')




import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from SIS_sim import SIS_sim
from me_analysis import me_analysis
NetworkType = 'ws'
N = 100
k = 6
#p_n = 0.2
p_ws = 0.5
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
network={'NetworkType':NetworkType,
         'N':N,
         'k':k,
         'p_ws':p_ws,
         #'p_n':p_n,
         'strategy':'prior',
         'p_n':0.5
         #'p_ws':p_ws,
         #'p':p,
        # 'p_n':p_n
    }
lambda1 = 3.57
lambda2 = 3
param ={'rho0':0.1,
                    't0':1000,
                    'lambda1':lambda1,
                    'lambda2':lambda2,
                    #'beta_1':0.2,
                    #'lambda2':lambda2,
                    #'beta_2':0.2,
                    'mu':0.3,
                    'p_n':0.5,
                    'simplex':1,
                    'gamma1':1,
                    'gamma2':1,
                    'alpha1':0.5
                    }
sis = SIS_sim(network,param)

for nn in range(1,11):
    for alpha1 in np.arange(0,3,0.25)/3:
        sis.alpha1 = alpha1
        sis.add_Sign(sis.p_n,sis.alpha1)
        sis.run()
        pickle.dump(sis, open(dir1+"prior_simplex1_sparse_"+str(round(alpha1,2))+"_"+str(nn)+".pkl", 'wb'))
        print("prior_"+str(round(alpha1,2))+"_"+str(nn))































