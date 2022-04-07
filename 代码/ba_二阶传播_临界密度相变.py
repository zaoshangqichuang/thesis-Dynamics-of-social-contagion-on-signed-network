# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 21:59:13 2022

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
N = 500
k = 8
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
param ={'rho0':0.1,
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
sis.cal_rho_steady_me()
print(sis.rho2,sis.rho3,sis.beta_2)
rho0=0.2
sis.rho0 = rho0
sis.rho_list = [rho0]
#sis.me_sim()
#rho_steady_me_list.append(sis.rho)
#sis.run()
rho_l= []
plot_dict4 = {}
for lambda2 in np.arange(1.4,2.2,0.2):
    sis.lambda2 = 1.38
    sis.beta_2=sis.lambda2*sis.mu/(sis.k_delta_mean*(sis.p*sis.gamma1+(1-sis.p)*sis.q*sis.gamma3))
    print(sis.beta_2)
    theta0_list = []
    for rho0 in np.arange(1,20,1)/100:
        sis.rho0=rho0
        sis.cal_theta_final(rho0)
        sis.rho_list_me = [rho0]
        sis.me_sim()
        print(rho0,sis.theta_yin_list[-1],sis.theta)
        theta0_list.append(sis.theta)
    plot_dict4[str(round(sis.lambda2,2))] = [[sis.theta_yin_list[-1]],theta0_list]
    plt.plot(np.arange(1,20,1)/100,theta0_list,'o-')

np.savez(dir1+"rho_linjie",plot_dict4=plot_dict4)


sis.t0=2000
sis.lambda2 = 1.38
sis.beta_2=sis.lambda2*sis.mu/(sis.k_delta_mean*(sis.p*sis.gamma1+(1-sis.p)*sis.q*sis.gamma3))
plot_dict5 = {}
#rho0_list = [0.26,0.27,0.28]
for rho0 in np.arange(0.08*500,0.12*500+1,1)/500:
    sis.rho0=rho0
    sis.rho_list_me = [rho0]
    sis.me_sim()
    plt.plot(sis.theta_list)
    plot_dict5[str(round(rho0*500,0))] = sis.theta_list
    
np.savez(dir1+"rho_linjie_re",plot_dict5=plot_dict5)



#和均质网络进行对比

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
#from SIS_sim import SIS_sim
from me_analysis import me_analysis
dir1 = 'npzfile\\'
dir2 = 'pngfile\\'
N = 500
k = 8
NetworkType = 're'
p_n = 0.2
network={'NetworkType':NetworkType,
             'N':N,
             'k':k,
            # 'p_ws':p_ws,
             'p_n':p_n,
             'strategy':'random'
             #'p_ws':p_ws,
             #'p':p,
             #'p_n':p_n
        }
rho_array = np.arange(1,30,2)/100
param ={'rho0':0.1,
            't0':1000,
            'lambda1':0.6,
            'beta_2':0.6,
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
sis = me_analysis(network,param)
print(sis.rho2,sis.rho3,sis.beta_2,sis.lambda2)
rho_l= []
plot_dict5 = {}
for lambda2 in np.arange(1.4,3.2,0.2):
    sis.lambda2 = lambda2
    sis.beta_2=sis.lambda2*sis.mu/(sis.k_mean_2*(sis.p*sis.gamma1+(1-sis.p)*sis.q*sis.gamma3))
    print(sis.beta_2)
    theta0_list = []
    for rho0 in np.arange(1,51,1)/100:
        sis.rho0=rho0
        #sis.cal_theta_final(rho0)
        #sis.rho_list_me = [rho0]
        sis.me_sim()
        print(rho0,sis.rho_steady,sis.rho_list_me[-1])
        theta0_list.append(sis.rho_list_me[-1])
    plot_dict5[str(round(sis.lambda2,2))] = [[sis.rho_steady],theta0_list]
    plt.plot(np.arange(1,51,1)/100,theta0_list,'o-')

sis.lambda2 = 2.69
sis.beta_2=sis.lambda2*sis.mu/(sis.k_mean_2*(sis.p*sis.gamma1+(1-sis.p)*sis.q*sis.gamma3))
print(sis.beta_2)  
#sis.beta_2= 0.09
plot_dict6 = {}
sis.t0=2000
#rho0_list = [0.26,0.27,0.28]
for rho0 in np.arange(0.33*500,0.35*500+1,1)/500:
    sis.rho0=rho0
    sis.me_sim()
    plt.plot(sis.rho_list_me)
    plot_dict6[str(round(rho0*500,0))] = sis.rho_list_me
np.savez(dir1+"rho_linjie_re",plot_dict5=plot_dict6)

'''
    theta0_list.append(sis.theta)
    sis.rho0 = rho0
    sis.rho_list = [rho0]
    sis.run()
    sis.me_sim()
    rho_l.append(sis.rho_steady_from_sim)
  '''  
'''
for beta_2 in np.linspace(0.06,0.1,10):
    param ={'rho0':0.05,
            't0':500,
            'lambda1':0.001,
            'beta_2':beta_2,
            #'beta_1':0.2,
            #'lambda2':lambda2,
            #'beta_2':0.2,
            'mu':0.1,
            'p_n':2/6,
            'simplex':2,
            'gamma1':1.5,
            'gamma2':0,
            'gamma3':0.5
            #'alpha1':alpha1,
            
            }
        #lambda1_array = [0.25,0.75,0.8,0.9,1,1.5]
        #lambda1_array = np.arange(0.25,2.25,0.25)
    sis = BA_me_analysis(network,param)
    sis.cal_rho_steady_me()
    print(sis.rho2,sis.rho3)
    rho_steady_me_list = []
    for nn in range(1,2):
        for rho0 in rho_array:
            sis.rho0 = rho0
            sis.rho_list = [rho0]
            #sis.me_sim()
            #rho_steady_me_list.append(sis.rho)
            sis.run()
            sis.cal_theta_final(rho0)
            rho_steady_me_list.append(sis.rho_steady_from_sim)
            pickle.dump(sis, open(dir1+"simplex_2_ba\\simplex2_rho\\"+"simplex2_ba_me_"+'_'+str(nn)+'_'+str(round(rho0,2))+"_"+str(round(beta_2,2))+".pkl", 'wb'))
#np.savez(dir1+"rho_steady_me",rho_steady_me_list=rho_steady_me_list,rho0_list=[0.05,0.1,0.2,0.5,0.7])
'''