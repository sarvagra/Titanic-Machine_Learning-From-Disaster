# %% imprting packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


# %% read test and train data
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
if 'Survived' not in df_test.columns :
    df_test['Survived'] = 0

df_test.head(),df_train.head()

# %% PREPROCESSING

# combine both datasets for preprocessing
df = pd.concat([df_train, df_test], axis=0)

# drop unnecessary columns
df=df.drop(columns=['Ticket','Name'],axis=1)

# convert categorical variables to numerical
df['Embarked']=df['Embarked'].map({'C':0,'Q':1,'S':2})
#df['Sex']=df['Sex'].map({'male':0,'female':1})

# fill all missing values
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Cabin']=df['Cabin'].fillna('X000')

# %% use KNNImputer to impute the missing fare and embarked values
impute_list = ['Age','Fare','Embarked']
rest=[ col for col in df.columns if col not in impute_list]

df_impute=df[impute_list] # split the data into impute and non-impute parts
df_rest=df[rest]

imputer=KNNImputer() # create the imputer object
impute_array=imputer.fit_transform(df_impute) # fit the imputer to the data

df_imputed=pd.DataFrame(impute_array,columns=impute_list) # convert the imputed array back to a dataframe

df=pd.concat([df_rest.reset_index(drop=True),df_imputed.reset_index(drop=True)],axis=1) # concatenate the imputed and non-imputed dataframes
df['Embarked']=df['Embarked'].round().astype(int) # round the embarked values to the nearest integer

# %% feature engineering
# split males and females , cabin
df = pd.get_dummies(df, columns = ['Sex'], prefix = 'Sex')

df['Cabin_Number'] = df['Cabin'].str.extract(r'(\d+)', expand=False).astype(float)
df['Cabin_Letter'] = df['Cabin'].str.extract(r'([a-zA-Z]+)', expand=False)
df['cabin_number'] = df['cabin_number'].fillna(0) # fill missing values with 0
   df['cabin_number'] = pd.to_numeric(df['cabin_number']) # convert to numeric
# %%
df.isna().sum() # check for missing values
# %%
