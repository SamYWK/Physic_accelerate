# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 23:43:13 2018

@author: pig84
"""

import pandas as pd
import numpy as np

data = pd.read_csv('minitree_4b_2_26.txt', sep=" ", header=0)
data['Target'] = (data['Jet_genjetPt']/data['Jet_pt']).values
data = data.drop(['Jet_genjetPt'], axis = 1)

print(data.head())
data.to_csv('minitree_4b_2_26.csv', sep=',', float_format = np.float64, index = False)