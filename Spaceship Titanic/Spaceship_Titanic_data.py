#!/usr/bin/env python
# coding: utf-8

# Spaceship Titanic
# =================
# 
# https://www.kaggle.com/competitions/spaceship-titanic/code?competitionId=34377

# In[1]:


import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer

import warnings
warnings.filterwarnings('ignore')


# Set random seed from reproductability
# -------------------------------------

# In[2]:


def set_seed(seed):
    'Sets the seed of the entire notebook so results are the same every time we run. This is for REPRODUCIBILITY.'
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(42)


# Read datasets
# -------------

# In[3]:


ROOT_PATH = './input_data/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

train_data = pd.read_csv(ROOT_PATH+TRAIN_FILE)
test_data = pd.read_csv(ROOT_PATH+TEST_FILE)
print(f'(train) Number of rows = {train_data.shape[0]} and Number of cols = {train_data.shape[1]}')
print(f'(test) Number of rows = {test_data.shape[0]} and Number of cols = {test_data.shape[1]}')

train_data


# Analyze train data
# ------------------

# In[4]:


def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summ['null'] = df.isnull().sum()
    summ['unique'] = df.nunique()
    summ['min'] = df.min()
    summ['median'] = df.median()
    summ['max'] = df.max()
    summ['mean'] = df.mean()
    summ['std'] = df.std()
    return summ

summary(train_data)


# Feature engineering
# -------------------

# Create a `FunctionTransformer` to remove Name and Cabin (not significative because too many different values), and split the PassengerId in the group id and person id.

# In[5]:


def feature_engineering(data):
    return (
            data
            .assign(
                PassengerGGG = [x.split('_')[-0] for x in data['PassengerId']],
                PassengerPP = [x.split('_')[-1] for x in data['PassengerId']],
            )
            .drop(columns=['Name', 'Cabin', 'PassengerId'])
        )

fe_eng = FunctionTransformer(feature_engineering)

fe_eng.fit_transform(train_data)


# Create an `Imputer` to fill the missing values with the most frequent value in text columns, and with the mean in numeric ones.

# In[6]:


imputer = ColumnTransformer(
    [
        (
            'label_imputer',
            SimpleImputer(strategy='most_frequent'),
            ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'PassengerGGG', 'PassengerPP']
        ),
        (
            'num_imputer',
            SimpleImputer(strategy='mean'),
            ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        )
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
).set_output(transform='pandas')

imputer.fit_transform(fe_eng.fit_transform(train_data))


# In[7]:


scale_encode = ColumnTransformer(
    [
        (
            'std_scaler',
            StandardScaler(),
            ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        ),
        (
            'minmax_scaler',
            MinMaxScaler(),
            ['PassengerGGG', 'PassengerPP']
        ),
        (
            'one_hot',
            OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'),
            ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
        )
    ],
    verbose_feature_names_out=False,
    remainder='passthrough'
).set_output(transform='pandas')

scale_encode.fit_transform(imputer.fit_transform(fe_eng.fit_transform(train_data)))


# Combine all transformers in a pipeline.

# In[8]:


preproc = Pipeline(
    [
        ('fe_eng', fe_eng),
        ('imputer', imputer),
        ('scale_encode', scale_encode)
    ]
).set_output(transform='pandas')

preproc.fit_transform(train_data)


# Create a separated transformer to drop the target column

# In[10]:


def drp_trg(data):
    if hasattr(data, 'Transported'):
        return data.drop(columns=['Transported'])
    else:
        return data

drop_target = FunctionTransformer(drp_trg)

drop_target.fit_transform(train_data)

