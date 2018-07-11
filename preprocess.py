# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:18:19 2018

@author: SamKao
"""

import pandas as pd
import numpy as np

df = pd.read_csv('minitree_4b_2_26_modified.csv', header = 0)
index = pd.read_csv('bad_list.csv', header = 0).values.tolist()
index = [item for sublist in index for item in sublist]
print(index)
df = df.drop(df.index[index])
print(df.head())
print(df.values.shape)
df.to_csv('./minitree_4b_2_26_modified_2.csv', index = False)