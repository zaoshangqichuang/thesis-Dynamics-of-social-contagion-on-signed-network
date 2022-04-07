# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:43:57 2022

@author: 201921250004
"""
from me_analysis import me_analysis
from BA_me_analysis import BA_me_analysis
def base(cl):
    if cl.NetworkType == 'ba':
        return 'BA_me_analysis'
    else:
        return 'me_analysis'
       
       
   
     
