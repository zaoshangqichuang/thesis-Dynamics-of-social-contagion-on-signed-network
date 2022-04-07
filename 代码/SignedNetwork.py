# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:14:46 2021

@author: 201921250004
"""
import numpy as np
import random
import networkx as nx
from pick_edges import pick_edges 
from pick_edges import random_sample_edges
from RealNetwork import RealNetwork
from itertools import combinations
import operator
class SignedNetwork(RealNetwork):
    def __init__(self,Network):
        self.NetworkType = Network['NetworkType']
        if self.NetworkType in ['er','ws','re','ba']:
            #self.NetworkType=Network["NetworkType"]
            self.N=Network["N"]
            self.k=Network["k"]
            self.p_ws=Network.get("p_ws",None) #小世界网络的重连概率
            self.strategy = Network.get("strategy","random")
            self.p_n = Network.get('p_n',None) #随机撒负边
            #self.p = Network['p']
            #self.q=Network.get("q",None)
            if self.strategy == "random":
                if self.NetworkType=="er":#ER network
                    self.nosignNetwork = nx.erdos_renyi_graph(self.N, self.k/(self.N-1))
                if self.NetworkType=="ws":#ER network
                    self.nosignNetwork = nx.watts_strogatz_graph(self.N,self.k,self.p_ws)
                if self.NetworkType=="ba":#BA network
                    self.nosignNetwork = nx.barabasi_albert_graph(self.N,self.k)
                        
                if self.NetworkType=="re":#Re network
                    self.nosignNetwork = nx.random_regular_graph(self.k,self.N)
            else:
                #小世界网络断边重连
                if self.NetworkType=="ws":
                    self.nosignNetwork = nx.random_regular_graph(self.k,self.N)
                    n_edges_data = list(self.nosignNetwork.edges.data())
                    selected_edges = random_sample_edges(self.nosignNetwork.edges,self.p_ws)
                    for u,v,wt in n_edges_data:
                        if (u,v) in selected_edges:
                            self.nosignNetwork.remove_edge(u, v)
                            if random.random()<0.5:
                                new_node = self.relink(u)
                                self.nosignNetwork.add_edge(u,new_node)
                                self.nosignNetwork[u][new_node]['new_link']=1
                            else:
                                new_node = self.relink(v)
                                self.nosignNetwork.add_edge(v,new_node)
                                self.nosignNetwork[v][new_node]['new_link']=1
                        else:
                            self.nosignNetwork[u][v]['new_link']= 0
                elif self.NetworkType=="ba":
                    self.nosignNetwork = nx.barabasi_albert_graph(self.N,self.k)
                    self.edge_betweenness_centrality =nx.centrality.edge_betweenness_centrality(self.nosignNetwork)
                    selected_edges = sorted(self.edge_betweenness_centrality, key=operator.itemgetter(1),reverse=True)
                    selected_edges = selected_edges[:int(len(selected_edges)*self.p_ws)]
                    for u,v,wt in self.nosignNetwork.edges.data():
                        if (u,v) in selected_edges:
                            self.nosignNetwork[u][v]['new_link']=1
                        else:
                            self.nosignNetwork[u][v]['new_link']=0
                        
                    
            #self.p_po = self.cal_po_edge_p()
            #self.p_ne = 1-self.p_po
            if self.p_n != None:
                self.p = self.cal_p()
                self.q = self.cal_q()
        elif self.NetworkType == 'real':
            self.readtxt(filename="dataset\\"+Network['name'])
    def cal_q(self):
        return self.k*self.p_n/((self.N-1)*(1-self.p))
    def cal_p(self):
        return (1-self.p_n)*self.k/(self.N-1)
    def relink(self,u):
        neighbor_list = list(nx.neighbors(self.nosignNetwork,u))
        node_list = list(self.nosignNetwork.nodes)
        for del_node in neighbor_list:
            node_list.remove(del_node)
        node_list.remove(u)
        return random.sample(node_list,1)[0]
    def add_Sign(self,p_n,alpha1=None):
        self.signedNetwork=self.nosignNetwork
        self.p_n=p_n
        self.p = self.cal_p()
        self.q = self.cal_q()
        if self.strategy == 'random':
            selected_edges = random.sample(self.signedNetwork.edges,int(len(self.signedNetwork.edges)*self.p_n))
        else:
            new_link_edges = pick_edges(self.signedNetwork,{'new_link':1})
            old_link_edges = pick_edges(self.signedNetwork,{'new_link':0})
            selected_edges = random_sample_edges(new_link_edges,self.p_n*alpha1/self.p_ws)
            selected_edges += random_sample_edges(old_link_edges,self.p_n*(1-alpha1)/(1-self.p_ws))
            #selected_edges = random.sample(new_link_edges,int(len(new_link_edges)*self.p_n*alpha1/self.p_ws))
            #selected_edges += random.sample(old_link_edges,int(len(old_link_edges)*self.p_n*(1-alpha1)/(1-self.p_ws)))
        for u,v,wt in self.signedNetwork.edges.data():
                if (u,v) in selected_edges:
                    self.signedNetwork[u][v]["ne_weight"]=int(1)
                    self.signedNetwork[u][v]["po_weight"]=int(0)
                else:
                    self.signedNetwork[u][v]["ne_weight"]=int(0)
                    self.signedNetwork[u][v]["po_weight"]=int(1)
            #for u,v,wt in self.signedNetwork.edges.data():
                #一半分布在长程边，一半分布在非长程边
                #alpha1表示负边在长程边上的比例
                '''
                if self.signedNetwork[u][v]['mark'] == 'new_link':
                    if random.random()<self.p_n*alpha1/self.p_ws:
                        self.signedNetwork[u][v]["ne_weight"]=int(1)
                        self.signedNetwork[u][v]["po_weight"]=int(0)
                    else:
                        self.signedNetwork[u][v]["ne_weight"]=int(0)
                        self.signedNetwork[u][v]["po_weight"]=int(1)
                else:
                    if random.random()<self.p_n*(1-alpha1)/(1-self.p_ws):
                        self.signedNetwork[u][v]["ne_weight"]=int(1)
                        self.signedNetwork[u][v]["po_weight"]=int(0)
                    else:
                        self.signedNetwork[u][v]["ne_weight"]=int(0)
                        self.signedNetwork[u][v]["po_weight"]=int(1)
                '''
        return self
    
    def get_degree(self):
        self.po_degree_array=np.array(list(self.signedNetwork.degree(weight="po_weight")))
        self.ne_degree_array=np.array(list(self.signedNetwork.degree(weight="ne_weight")))
        self.po_mean=np.mean(self.po_degree_array[:,1])
        self.ne_mean=np.mean(self.ne_degree_array[:,1])
        print("%.2f"%self.po_mean,"%.2f"%self.ne_mean)
        return self.po_mean,self.ne_mean
    def get_delta_degree(self):
        self.po_degree_array=self.signedNetwork.degree(weight="po_weight")
        self.ne_degree_array=self.signedNetwork.degree(weight="ne_weight")
        self.degree_dict = {}
        for node in self.signedNetwork.nodes():
            self.degree_dict[node] = {}
            self.degree_dict[node]["po_degree"]=self.po_degree_array[node]
            self.degree_dict[node]["ne_degree"]=self.ne_degree_array[node]
            po_neighbor_list = list(filter(lambda x: self.signedNetwork[node][x]['po_weight']==1,self.signedNetwork.neighbors(node)))
            num_po_triangles = 0
            num_ne_triangles = 0
            num_triangles = 0
            for node1,node2 in combinations(po_neighbor_list,2):
                num_triangles += 1
                if self.signedNetwork.has_edge(node1,node2):
                    if self.signedNetwork[node1][node2]['po_weight']==1:
                        num_po_triangles += 1
                    elif self.signedNetwork[node1][node2]['ne_weight']==1:
                        num_ne_triangles += 1
            self.degree_dict[node]["num_triangles"]=num_triangles
            self.degree_dict[node]["num_po_triangles"]=num_po_triangles
            self.degree_dict[node]["num_ne_triangles"]=num_ne_triangles
        return 1
                
            

'''
Network = {
'N':100,
'k':6,
'p':0.01,
'p_n':0.2,
'NetworkType':'ba'
}
for p_n in np.arange(0,1,0.01):
    sn = SignedNetwork(Network)
    sn.add_Sign(p_n)
    sn.get_degree()
'''