# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 23:43:13 2018

@author: pig84
"""

import pandas as pd

data = pd.read_csv('minitree_4b_2_26.txt', sep=" ", header=0)
data['Target'] = (data['Jet_genjetPt']/data['Jet_pt']).values
data = data.append(data.iloc[0])
print(data.head())
print(data.shape)
data.to_csv('minitree_4b_2_26.csv', sep=',', index = False)