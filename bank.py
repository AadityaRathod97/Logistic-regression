# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:09:00 2020

@author: DELL
"""

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt


bank_full = pd.read_csv("bank-full.csv")



bank_full.isnull().sum()
bank_full.columns
df = pd.DataFrame(bank_full.corr())


#coverting the string values into numeric 
bank_full['job'] = bank_full['job'].astype('category')
bank_full['job'] = bank_full['job'].cat.codes
bank_full['marital'] = bank_full['marital'].astype('category')
bank_full['marital'] = bank_full['marital'].cat.codes
bank_full['education'] = bank_full['education'].astype('category')
bank_full['education'] = bank_full['education'].cat.codes
bank_full['default'] = bank_full['default'].astype('category')
bank_full['default'] = bank_full['default'].cat.codes
bank_full['housing'] = bank_full['housing'].astype('category')
bank_full['housing'] = bank_full['housing'].cat.codes
bank_full['loan'] = bank_full['loan'].astype('category')
bank_full['loan'] = bank_full['loan'].cat.codes
bank_full['contact'] = bank_full['contact'].astype('category')
bank_full['contact'] = bank_full['contact'].cat.codes
bank_full['month'] = bank_full['month'].astype('category')
bank_full['month'] = bank_full['month'].cat.codes
bank_full['poutcome'] = bank_full['poutcome'].astype('category')
bank_full['poutcome'] = bank_full['poutcome'].cat.codes
bank_full['y'] = bank_full['y'].astype('category')
bank_full['y'] = bank_full['y'].cat.codes

