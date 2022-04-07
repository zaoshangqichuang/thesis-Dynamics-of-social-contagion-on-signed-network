# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:34:34 2022

@author: 201921250004
"""
from SignedNetwork import SignedNetwork
import numpy as np
#import pulp
import networkx as nx
import random
from SIS_sim import SIS_sim
class BA_me_analysis(SignedNetwork,SIS_sim):
    def __init__(self,Network,param):
        super().__init__(Network)
        if self.NetworkType in ['er','ws','re','ba']:
            self.p_n = param.get('p_n',0)
            self.add_Sign(param['p_n'],param.get('alpha1',None))
            #self.mu = param.get('mu',1)
            self.k_density_list=[z/self.N for z in nx.degree_histogram(self.nosignNetwork)]
        self.signNetwork =self.signedNetwork
        self.k_density_list=[z/self.N for z in nx.degree_histogram(self.signNetwork)]
        self.k_mean = self.cal_k_mean()
        self.simplex = param.get('simplex',1)
        self.gamma1 = param.get('gamma1',1) 
        self.gamma2 = param.get('gamma2',1) 
        self.gamma3 = param.get('gamma3',1)
        
        self.mu = param.get('mu',1)
        self.k_mean_2 = self.cal_k_2_mean()
        self.p = self._cal_p()
        self.q = self._cal_q()
        self.k_delta_mean = self.cal_k_delta_mean()
        self.hmf = self.k_mean**2/self.k_mean_2
        self.cal_k_po_p()
        self.cal_k_po_mean()
        self.hmf1 = self.k_po_mean**2/self.k_po_mean_2
        self.beta_1 = param.get('beta_1',0)
        self.lambda1 = param.get('lambda1',self.cal_lambda_1())
        #self.lambda2 = param['lambda2']
        self.beta_1 = param.get('beta_1',self.lambda1*self.mu/self.k_po_mean)
        self.rho0 = param.get('rho0',0.1)
        self.t0 = param.get('t0',1000) #迭代次数
        self.rho_list = [self.rho0]
        if self.simplex ==2:
            self.beta_2 = param.get('beta_2',0)
            self.lambda2 = param.get('lambda2',self.cal_lambda_2())
            self.beta_2 = param.get('beta_2',self.lambda2*self.mu/(self.k_delta_mean*(self.p*self.gamma1+(1-self.p)*self.q*self.gamma3)))
            self.gamma1 = param.get('gamma1',1)
            self.gamma2 = 0
            self.gamma3 = param.get('gamma3',1)
    def cal_lambda_1(self):
        self.lambda1 = self.beta_1*self.k_mean*(1-self.p_n)/self.mu
        return self.lambda1
    def cal_lambda_2(self):
        self.k_delta_mean = self.cal_k_delta_mean()
        self.lambda2 = self.beta_2*self.k_delta_mean*(self.p*self.gamma1+(1-self.p)*self.q*self.gamma3)/self.mu
        return self.lambda2
    def cal_k_mean(self):
        self.k_mean = 0
        for k in range(len(self.k_density_list)):
            self.k_mean += k*self.k_density_list[k]
        return self.k_mean
    def cal_k_2_mean(self):
        self.k_mean_2 = 0
        for k in range(len(self.k_density_list)):
            self.k_mean_2 += k**2*self.k_density_list[k]
        return self.k_mean_2
    def cal_k_delta_mean(self):
        if self.simplex == 2:
            self.k_delta_mean = 0
            for k in range(len(self.k_density_list)):
                self.k_delta_mean += self.k_density_list[k]*self.cal_k_delta(k)
        else:
            self.k_delta_mean = 0
        return self.k_delta_mean
    def cal_k_po_p(self):
        k_po_dict = {}
        for node in self.signedNetwork.nodes():
            k_po = 0
            k = nx.degree(self.signedNetwork,node)
            for neighbor in nx.neighbors(self.signedNetwork,node):
                if self.signedNetwork[node][neighbor]['po_weight'] == 1:
                    k_po += 1
            if k_po_dict.get(k,-1)==-1:
                k_po_dict[k] = [k_po]
            else:
                k_po_dict[k].append(k_po)
        self.k_po_p = np.zeros(shape=(len(self.k_density_list),len(self.k_density_list)),dtype=float)
        for key in k_po_dict.keys():
            for k_po in k_po_dict[key]:
                self.k_po_p[key][k_po] += 1
        self.k_po_p =  self.k_po_p/self.k_po_p.sum(axis=1).reshape(-1,1)
        self.k_po_p[np.isnan(self.k_po_p)] = 0
        return self.k_po_p
    
    def cal_k_po_mean(self):
        self.k_po_list = []
        for k in range(len(self.k_density_list)):
            k_po = 0
            for i in range(len(self.k_density_list)):
                k_po += self.k_po_p[k][i]*i
            self.k_po_list.append(k_po)
        self.k_po_mean = 0
        for k_po,k_p in zip(self.k_po_list,self.k_density_list):
            self.k_po_mean += k_po*k_p
        self.k_po_mean_2 = 0
        for k_po,k_p in zip(self.k_po_list,self.k_density_list):
            self.k_po_mean_2 += k_po**2*k_p
        return 1
            
    
    
    def cal_k_I_rho(self):
        self.k_I_array = np.zeros(len(self.k_density_list),dtype=float)
        for node in self.signedNetwork.nodes():
            k = nx.degree(self.signedNetwork,node)
            if self.signedNetwork.nodes[node]['state'] == 'I':
                self.k_I_array[k] += 1
        self.k_I_rho = self.k_I_array / np.array(nx.degree_histogram(self.nosignNetwork))
        self.k_I_rho[np.isnan(self.k_I_rho)] = 0
        return self.k_I_rho
    
    
    def cal_theta(self):
        self.theta= 0
        for k in range(len(self.k_density_list)): 
            self.theta += k*self.k_density_list[k]*self.k_I_rho[k]
        self.theta = self.theta/self.k_mean
        if self.theta > (self.N-1)/self.N:
            self.theta = 1
        elif self.theta < 1/self.N:
            self.theta = 0
        return self.theta
    def cal_rho_steady(self):
        self.rho = 0
        for k in range(len(self.k_density_list)):
            self.rho += self.k_density_list[k]*self.k_I_rho[k]
        if self.rho < 1/self.N:
            self.rho = 0
        elif self.rho > self.N-1/self.N:
            self.rho = 1
        return self.rho
    def setup(self):
        #初始化节点感染概率
        random_select_list = random.sample(self.signedNetwork.nodes,int(self.rho0*self.N))
        for node in self.signedNetwork.nodes:
            if node in random_select_list:
                self.signedNetwork.nodes[node]['state']='I'
            else:
                self.signedNetwork.nodes[node]['state']='S'
        return 1
    def cal_k_delta(self,k):
        if k*(1-self.p_n)*(k*(1-self.p_n)-1)/2 > 0:
            return k*(1-self.p_n)*(k*(1-self.p_n)-1)/2
        else:
            return 0
    
    def _cal_q(self):
        return self.k_mean*self.p_n/((self.N-1)*(1-self.p))
    def _cal_p(self):
        return (1-self.p_n)*self.k_mean/(self.N-1)
    def cal_k_q(self):
        return self.k_mean*self.p_n/((self.N-1)*(1-self.p))
    def cal_k_p(self):
        return (1-self.p_n)*self.k_mean/(self.N-1)
    
    def cal_rho_steady_me(self):
        if self.simplex == 1 or self.lambda2 == 0:
            if self.lambda1 > 1:
                self.rho_steady = 1-1/self.lambda1
            else:
                self.rho_steady = 0
                self.rho2 = None
                self.rho3 = None
        elif self.simplex == 2:
            #rho1 = 0
            self.delta = (self.lambda2-self.lambda1)**2-4*self.lambda2*(1-self.lambda1)
            if self.delta>0:
                self.rho2 = (self.lambda2-self.lambda1-np.sqrt(self.delta))/(2*self.lambda2)
                self.rho3 = (self.lambda2-self.lambda1+np.sqrt(self.delta))/(2*self.lambda2)
                if self.rho2 >= 0 and self.rho3 <= 1:
                    if self.rho0 < self.rho2:
                        self.rho_steady = 0
                    else:
                        self.rho_steady = self.rho3
                elif self.rho2 >= 1:
                    self.rho_steady = 0
                    self.rho2 = None
                    self.rho3 = None
                elif self.rho2 < 0 and self.rho3 <= 1 and self.rho3 > 0:
                    self.rho_steady = self.rho3
                    self.rho2 = None
                else:
                    self.rho_steady = 0
                    self.rho2 = None
                    self.rho3 = None
            else:
                self.rho_steady = 0
                self.rho2 = None
                self.rho3 = None
        return 1
    
    def me_sim(self):
        self.rho_list_me = [self.rho0]
        self.setup()
        self.cal_k_I_rho()
        self.theta_list = [self.cal_theta()]
        if self.simplex == 1 or self.beta_2 == 0 or self.gamma1==0:
            for t in range(self.t0):
                for k in range(len(self.k_density_list)):
                    rho_d = -self.mu*self.k_I_rho[k]+self.beta_1*k*(1-self.p_n)*self.theta_list[-1]*(1-self.k_I_rho[k])
                    self.k_I_rho[k] += rho_d
                    if self.k_I_rho[k] < 0:
                        self.k_I_rho[k] = 0
                    elif self.k_I_rho[k] > 1:
                        self.k_I_rho[k] = 1
                self.theta_list.append(self.cal_theta())
                self.rho_list_me.append(self.cal_rho_steady())
                if self.rho >= 1 or self.rho <= 0:
                    break
        elif self.simplex == 2:
            for t in range(self.t0):
                for k in range(len(self.k_density_list)):
                    rho_d = -self.mu*self.k_I_rho[k]+self.beta_1*k*(1-self.p_n)*self.theta_list[-1]*(1-self.k_I_rho[k])
                    rho_d += self.beta_2*self.cal_k_delta(k)*self.theta_list[-1]**2*(1-self.k_I_rho[k])*(self.p*self.gamma1+(1-self.p)*self.q*self.gamma3)
                    self.k_I_rho[k] += rho_d
                    if self.k_I_rho[k] < 0:
                        self.k_I_rho[k] = 0
                    elif self.k_I_rho[k] > 1:
                        self.k_I_rho[k] = 1
                self.theta_list.append(self.cal_theta())
                self.rho_list_me.append(self.cal_rho_steady())
                if self.rho >= 1 or self.rho <= 0:
                    break
    def cal_k_lambda1(self,k):
        return self.beta_1*k*(1-self.p_n)/self.mu
    def cal_k_lambda2(self,k):
        return self.beta_2*self.cal_k_delta(k)*(self.p*self.gamma1+(1-self.p)*self.q*self.gamma3)/self.mu
    def cal_theta_final(self,theta0):
        self.theta_yin_list = [theta0]
        for t in range(self.t0):
            a = 0
            for k in range(len(self.k_density_list)):
                a += k*self.k_density_list[k]*(self.cal_k_lambda1(k)*theta0+self.cal_k_lambda2(k)*theta0**2)/(1+self.cal_k_lambda1(k)*theta0+self.cal_k_lambda2(k)*theta0**2)
            theta0 = a/self.k_mean
            if theta0 > (self.N-1)/self.N:
                theta0 = 1
                self.theta_yin_list.append(1)
                break
            elif theta0 < 1/self.N:
                theta0 = 0
                self.theta_yin_list.append(0)
                break
            else:
                self.theta_yin_list.append(theta0)
                continue

'''
network={'NetworkType':'ba',
         'N':100,
         'k':6,
         'p_n':0.2,
         'strategy':'random'
    }
param ={
'rho0':0.1,
't0':1000,
'lambda1':1,
'mu':0.3,
'p_n':0.2,
'simplex':2,
'gamma1':1.5,
'gamma2':0.5,
'alpha1':0.5,
'beta_2':0.02
}
ba0 = BA_me_analysis(network,param)
fig = plt.figure()
ax = fig.add_subplot(111)
rho_list_ba = []
for lambda1 in np.linspace(0.1,1,10):
    ba0.lambda1 = lambda1
    ba0.beta_1 = ba0.lambda1*ba0.mu/ba0.k_po_mean
    ba0.me_sim()
    rho_list_ba.append(ba0.rho)
ax.plot(np.linspace(0.5,1,10),rho_list_ba,'o')
'''                  
        
        
        
 