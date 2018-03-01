# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:46:06 2018

@author: pig84
"""

import pandas as pd

reader = pd.read_table('minitree_4b_2_26.txt', sep = ' ', iterator = True)

df = pd.DataFrame(reader.get_chunk(50000))

print(df.iloc[0])