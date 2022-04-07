# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:03:42 2021

@author: 201921250004
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import networkx as nx
import random
from pick_edges import pick_edges
from pick_edges import random_sample_edges
from pick_edges import random_sample_triangle
class SIS_sim(object):
    '''
    def __init__(self,Network,param):
        super().__init__(Network)
        self.add_Sign(param['p_n'])
        self.rho0 = param['rho0']
        self.t0 = param['t0'] #迭代次数
        self.beta_1 = param.get('beta_1',self.beta_1)
        self.beta_2 = param.get('beta_2',None)
        self.mu = param.get('mu',None)
        self.rho_list = [self.rho0]
        self.lambda1 = self.beta_1*(self.k*(1-self.p_n))/self.mu
    '''
    def __setup(self):
        #初始化节点感染概率
        random_select_list = random.sample(self.signedNetwork.nodes,int(self.rho0*self.N))
        for node in self.signedNetwork.nodes:
            if node in random_select_list:
                self.signedNetwork.nodes[node]['state']='I'
            else:
                self.signedNetwork.nodes[node]['state']='S'
        self.I_list = list(filter(lambda x: self.signedNetwork.nodes[x]['state']=='I',self.signedNetwork.nodes))
        self.S_list = list(filter(lambda x: self.signedNetwork.nodes[x]['state']=='S',self.signedNetwork.nodes))
        _,_ = self.count_nodes()
        print("初始状态--------------------------")
        print('I节点个数:{0},比例:{1}%'.format(self.count_I,round(self.count_I/self.N*100,2)))
        print('S节点个数:{0},比例:{1}%'.format(self.count_S,round(self.count_S/self.N*100,2)))

            
        
    def __1_simplex_contagion(self):
        for t in range(self.t0):
            self.i_add = [] #一轮中感染的节点
            self.s_add = []
            positive_edges = pick_edges(self.signedNetwork,{'po_weight':1,'state':'I'})
            selected_edges = random_sample_edges(positive_edges,self.beta_1)
            #selected_edges = random.sample(positive_edges,int(self.beta_1*len(positive_edges)))
            for (u,v) in selected_edges:
                self.i_add.append(v)
            selected_nodes = random_sample_edges(self.I_list,self.mu)
            '''
            if int(len(self.I_list)*self.mu) > 1:
                selected_nodes = random.sample(self.I_list,int(len(self.I_list)*self.mu))
            else:
                selected_nodes = []
                for i_node in self.I_list:
                    if random.random()<=self.mu:
                        selected_nodes.append(i_node)
            '''
            for i_node in selected_nodes:
                self.signedNetwork.nodes[i_node]['state']='S'
                self.s_add.append(i_node)
            '''
            for (u,v) in positive_edges:
                if random.random() <= self.beta_1:
                    self.i_add.append(v)
            
            for i_node in self.I_list:
                if random.random() <= self.mu:
                    self.signedNetwork.nodes[i_node]['state']='S'
                    self.s_add.append(i_node)
            '''
            self.__contagion()
            self.__recovery()
            _,_ = self.count_nodes()
            self.rho_list.append(self.count_I/self.N)
            print('\r'+'t={0}:I={1:.2f}%,S={2:.2f}%'.format(t,round(self.count_I/self.N*100,2),round(self.count_S/self.N*100,2)),end='')
            if self.count_I == 0 or self.count_I == self.N:
                break
    def __2_simplex_contagion(self):
        for t in range(self.t0):
            self.i_add = [] #一轮中感染的节点
            self.s_add = []
            '''
            #1-simplex
            for s_node in self.S_list:
                pi_list = []
                for neighbor in nx.neighbors(self.signedNetwork,s_node):
                    if self.signedNetwork[s_node][neighbor]['po_weight']==1 and self.signedNetwork[neighbor]['state']=='I':
                        pi_list.append(neighbor)
                if len(pi_list)==1:
                #1-simplex
                    if random.random() <= self.beta_1:
                        self.i_add.append(s_node)
                elif len(pi_list) > 1:
                    for i_node1,i_node2 in itertools.combinations(self.i_add,2):
                        try:
                            gamma = self.gamma1*self.signedNetwork[i_node1][i_node2]['po_weight']+self.gamma2*self.signedNetwork[i_node1][i_node2]['ne_weight']
                        except:
                            gamma = 1
                        if np.random.rand(1)[0]<= 2*self.beta_1*gamma:
                                self.i_add.append(s_node)
            '''
            #1-simplex
            positive_edges = pick_edges(self.signedNetwork,{'po_weight':1,'state':'I'})
            selected_edges = random_sample_edges(positive_edges,self.beta_1)
            #selected_edges = random.sample(positive_edges,int(self.beta_1*len(positive_edges)))
            for (u,v) in selected_edges:
                self.i_add.append(v)
            #2-simplex
            triangle_list = []
            for i_node1,i_node2 in itertools.combinations(self.I_list,2):
                if self.signedNetwork.has_edge(i_node1,i_node2):
                #if i_node2 in list(nx.neighbors(self.signedNetwork,i_node1)):
                    gamma = self.gamma1*self.signedNetwork[i_node1][i_node2]['po_weight']+self.gamma3*self.signedNetwork[i_node1][i_node2]['ne_weight']
                #else:
                    #gamma = self.gamma2
                    for s_node in list(nx.common_neighbors(self.signedNetwork,i_node1,i_node2)):
                        if s_node in self.S_list and self.signedNetwork[i_node1][s_node]['po_weight']==1 and self.signedNetwork[i_node2][s_node]['po_weight']==1:
    
                            '''
                            if random.random()<= self.beta_1*gamma:
                                self.i_add.append(s_node)
                            '''
                            triangle_list.append([(i_node1,i_node2,s_node),(gamma*self.beta_2)])
        
            sample_triangle_list = random_sample_triangle(triangle_list)
            try:
                for sample_triangle in sample_triangle_list:
                    self.i_add.append(sample_triangle[2])
            except:
                continue
            selected_nodes = random_sample_edges(self.I_list,self.mu)
            for i_node in selected_nodes:
                self.signedNetwork.nodes[i_node]['state']='S'
                self.s_add.append(i_node)
            self.__contagion()
            self.__recovery()
            _,_ = self.count_nodes()
            self.rho_list.append(self.count_I/self.N)
            print('\r'+'t={0}:I={1:.2f}%,S={2:.2f}%'.format(t,round(self.count_I/self.N*100,2),round(self.count_S/self.N*100,2)),end='')
            if self.count_I == 0 or self.count_I == self.N:
                break
    def __contagion(self):
        for i_node in set(self.i_add):
            self.signedNetwork.nodes[i_node]['state']='I'
            self.I_list.append(i_node)
            self.S_list.remove(i_node)
    def __recovery(self):
        for i_node in set(self.s_add):
            self.signedNetwork.nodes[i_node]['state']='S'
            self.S_list.append(i_node)
            self.I_list.remove(i_node)
            
    def count_nodes(self):
        #节点感染比例的计算
        self.count_I = len(self.I_list)
        self.count_S = len(self.S_list)
        return self.count_I,self.count_S
    def run(self):
        self.__setup()
        if self.simplex == 1 or self.beta_2 == 0 or self.gamma1 == 0:
            self.__1_simplex_contagion()
        else:
            self.__2_simplex_contagion()
        #plt.plot(self.rho_list)
        if self.rho_list[-1]>0 and self.rho_list[-1]<1:
            self.rho_steady_from_sim=np.mean(self.rho_list[-200:])
        else:
            self.rho_steady_from_sim=self.rho_list[-1]
        if self.NetworkType != 'ba' and self.NetworkType != 'real':
            print('\r理论稳态密度为{0:.2f}%,模拟稳态密度为{1:.2f}%'.format(round(self.rho_steady*100,2),round(self.rho_steady_from_sim*100,2)))
        else:
            print('模拟稳态密度为{:.2f}%'.format(round(self.rho_steady_from_sim*100,2)))
