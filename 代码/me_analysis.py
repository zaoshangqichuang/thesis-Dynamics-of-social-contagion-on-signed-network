# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:13:38 2021

@author: 201921250004
"""
from SignedNetwork import SignedNetwork
import numpy as np
import pulp
from SIS_sim import SIS_sim
class me_analysis(SignedNetwork,SIS_sim):
    def __init__(self,Network,param):
        
        super().__init__(Network)
        
        self.simplex = param.get('simplex',1)
        self.gamma1 = param.get('gamma1',1) 
        self.gamma2 = param.get('gamma2',1) 
        self.gamma3 = param.get('gamma3',1)
        self.p_n = param.get('p_n',0)
        self.mu = param.get('mu',1)
        self.k_mean = self.k*(1-self.p_n)
        self.p = self._cal_p()
        self.q = self._cal_q()
        self.beta_1 = param.get('beta_1',0)
        self.lambda1 = param.get('lambda1',self.cal_lambda_1())
        #self.lambda2 = param['lambda2']
        self.beta_1 = param.get('beta_1',self.lambda1*self.mu/self.k_mean)
        
        if self.simplex == 2:
            self.k_mean_2 = self.cal_k_mean_2()
            self.beta_2 = param.get('beta_2',self.beta_1*(self.gamma1*self.p+self.gamma2*(1-self.p)*(1-self.q)+self.gamma3*(1-self.p)*self.q))
            self.lambda2 = param.get('lambda2',self.beta_2*(self.gamma1*self.p+self.gamma3*(1-self.p)*self.q)*self.k_mean_2/self.mu)
            #self.lambda2 = param['lambda2']
            self.beta_2 = param.get('beta_2',self.lambda2*self.mu/(self.k_mean_2*(self.gamma1*self.p+self.gamma3*(1-self.p)*self.q)))
            #self.gamma1 = (self.beta_2/(self.beta_1)-2*self.p_n)/(1-2*self.p_n)
            #self.gamma1,self.gamma2,self.gamma3 = self._get_paramter(0.5)
        self.add_Sign(param['p_n'],param.get('alpha1',None))
        self.rho0 = param['rho0']
        self.t0 = param['t0'] #迭代次数
        self.rho_list = [self.rho0]
        if self.simplex == 1 or self.lambda2 == 0:
            if self.lambda1 > 1:
                self.rho_steady = 1-1/self.lambda1
            else:
                self.rho_steady = 0
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
      
    def _get_paramter(self,g3=0.5):
        prob = pulp.LpProblem(sense=pulp.LpMinimize)
        gamma1 = pulp.LpVariable('gamma1', lowBound=1,upBound=1/self.beta_1)
        gamma2 = pulp.LpVariable('gamma2', lowBound=1,upBound=1/self.beta_1)
        gamma3 = pulp.LpVariable('gamma3', lowBound=0.1,upBound=1)
        prob += gamma1-2*gamma2
        prob += (gamma1-2*gamma2>=0)
        prob += (gamma3-g3>=0)
        prob += (gamma1*self.beta_1*self.p+gamma2*self.beta_1*(1-self.p)*(1-self.q)+gamma3*self.beta_1*(1-self.p)*self.q==self.beta_2)
        prob.solve()
        return pulp.value(gamma1),pulp.value(gamma2),pulp.value(gamma3)
    def cal_k_mean(self):
        self.cal_k()
        if self.k*(1-self.p_n) < 0:
            self.k_mean = 0
        else:
            self.k_mean = self.k*(1-self.p_n)
        return self.k_mean
    def cal_k_mean_2(self):
        self.cal_k()
        if self.k*(1-self.p_n) < 0 or self.k*(1-self.p_n)-1<0:
            self.k_mean_2 = 0
        else:
            self.k_mean_2 = self.k*(1-self.p_n)*(self.k*(1-self.p_n)-1)/2
        return self.k_mean_2
    def _cal_q(self):
        return self.k_mean*self.p_n/((self.N-1)*(1-self.p))
    def _cal_p(self):
        return (1-self.p_n)*self.k_mean/(self.N-1)
    def cal_k(self):
        self.k=(self.N-1)*(self.p+(1-self.p)*self.q)
        return self.k
    def cal_p_n(self):
        self.p_n = (1-self.p)*self.q/((1-self.p)*self.q+self.p)
        return self.p_n
    def cal_beta_2(self):
        self.k_mean = self.cal_k_mean()
        self.k_mean_2 = self.cal_k_mean_2()
        self.beta_2 = self.beta_1*(self.gamma1*self.p+self.gamma2*(1-self.p)*(1-self.q)+self.gamma3*(1-self.p)*self.q)
        return self.beta_2
    def cal_lambda_2(self):
        self.k_mean = self.cal_k_mean()
        self.k_mean_2 = self.cal_k_mean_2()
        self.lambda2 = self.beta_2*self.k_mean_2/self.mu
    def cal_lambda_1(self):
        self.k_mean = self.cal_k_mean()
        self.k_mean_2 = self.cal_k_mean_2()
        self.lambda1 = self.beta_1*self.k_mean/self.mu
        return self.lambda1
    def cal_rho_steady(self):
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
        self.rho_list_me=[self.rho0]
        if  self.simplex == 1 or self.beta_2 == 0 or self.gamma1 == 0:
            for t in range(self.t0):
                rho_d = -self.rho_list_me[-1]*self.mu+self.beta_1*self.k_mean*self.rho_list_me[-1]*(1-self.rho_list_me[-1])
                if self.rho_list_me[-1]+rho_d <= (self.N-1)/self.N and self.rho_list_me[-1]+rho_d >= 1/self.N:
                    self.rho_list_me.append(self.rho_list_me[-1]+rho_d)
                elif self.rho_list_me[-1]+rho_d > (self.N-1)/self.N:
                    self.rho_list_me.append(1)
                    break
                elif self.rho_list_me[-1]+rho_d < 1/self.N:
                    self.rho_list_me.append(0)
                    break
        elif self.simplex == 2 :
            for t in range(self.t0):
                rho_d = -self.rho_list_me[-1]*self.mu+self.beta_1*self.k_mean*self.rho_list_me[-1]*(1-self.rho_list_me[-1])+self.beta_2*(self.gamma1*self.p+self.gamma3*(1-self.p)*self.q)*self.k_mean_2*self.rho_list_me[-1]**2*(1-self.rho_list_me[-1])
                if self.rho_list_me[-1]+rho_d <= (self.N-1)/self.N and self.rho_list_me[-1]+rho_d >= 1/self.N:
                    self.rho_list_me.append(self.rho_list_me[-1]+rho_d)
                elif self.rho_list_me[-1]+rho_d > (self.N-1)/self.N:
                    self.rho_list_me.append(1)
                    break
                elif self.rho_list_me[-1]+rho_d < 1/self.N:
                    self.rho_list_me.append(0)
                    break
        return self.rho_list_me
    