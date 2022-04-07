# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:57:46 2022

@author: huangguo
"""
import networkx as nx
import random 
def pick_edges(net,param):
    pick_list = []
    if len(list(param.items()))==1:
        for u,v,wt in net.edges.data(data = list(param.keys())[0]):
            if wt == list(param.values())[0]:
                pick_list.append((u,v))
    elif len(list(param.items()))==2:
        for u,v,wt in net.edges.data(data = list(param.keys())[0]):
            if wt == list(param.values())[0]:
                if net.nodes[u]['state']=='I'and net.nodes[v]['state']=='S':
                    pick_list.append((u,v))
                elif net.nodes[u]['state']=='S' and net.nodes[v]['state']=='I':
                    pick_list.append((v,u))
    return pick_list
def random_sample_edges(edge_list,p):
    sample_num = int(len(edge_list)*p)
    if sample_num > 0:
        return random.sample(edge_list,sample_num)
    else:
        sample_list = []
        for edge in edge_list:
            if random.random() <= p:
                sample_list.append(edge)
        return sample_list
def random_sample_triangle(triangle_list):
    pick_l = {}
    for l,p in triangle_list:
        if pick_l.get(str(round(p,2)),-1)==-1:
            pick_l[str(round(p,2))]=[]
        else:
            pick_l[str(round(p,2))].append(l)
    sample_list = []
    for key in pick_l.keys():
        sample_triangles=random_sample_edges(pick_l[key],eval(key))
        for sample_triangle in sample_triangles:
            sample_list.append(sample_triangle)
        if [] in sample_list:
            sample_list.remove([])
    return sample_list
'''
net= nx.random_regular_graph(6,100)
for u,v,wt in net.edges.data():
    if random.random()<0.5:
        net[u][v]['new_link']=1
    else:
        net[u][v]['new_link']=0
pick_list = pick_edges(net,{'new_link':1})
triangle_list =[ [(1,2,3),0.5],[(1,3,4),0.5],[(2,5,7),0.6],[(2,3,8),0.6],[(3,5,8),0.6]]
'''