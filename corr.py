# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:43:14 2018

@author: pig84
"""

import pandas as pd

def main():
#    df = pd.read_table('minitree_4b_2_26.txt', header = 0, sep = ' ')
    df = pd.read_csv('./minitree_4b_2_26_modified.csv', header = 0)
    df['Target'] = (df['Jet_genjetPt']/df['Jet_pt']).values
    df.corr().to_csv('correlation.csv', index = False)

if __name__ == '__main__':
    main()