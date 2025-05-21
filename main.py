# %% imprting packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
# %% read test and train data
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

# %% check the data
df_train.head()

# %% combine the data
df = pd.concat([df_train, df_test], axis=0)

