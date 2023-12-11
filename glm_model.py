"""
Description:
The purpose of this file is prepare our model for production by
dumping our final GLM model into a pickle file.
"""

# Load Libraries
import os
import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import pickle
from data_segmentation_processing import training_data_processing

## Local Path Set up
path = os.getcwd()
os.chdir('state-farm-coding-exercise')

## Load Model Train data
raw_train = pd.read_csv('exercise_26_train.csv')

## Data Processing & Feature Engineering
train_imputed_std, val_imputed_std, test_imputed_std = training_data_processing(raw_train)

## Initial Feature Selection
exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
exploratory_results['coefs'] = exploratory_LR.coef_[0]
exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
var_reduced = exploratory_results.nlargest(25,'coefs_squared')


## Preliminary Model
variables = var_reduced['name'].to_list()
train_y = train_imputed_std['y']
train_X = train_imputed_std[variables].copy()
train_X.replace({False:0, True:1}, inplace = True)
logit = sm.Logit(train_y, train_X)

## Fit Model
result = logit.fit()

## Finaliz the Model
train_and_val = pd.concat([train_imputed_std, val_imputed_std])
all_train = pd.concat([train_and_val, test_imputed_std])
variables = var_reduced['name'].to_list()
all_train_y = all_train['y']
all_train_X = all_train[variables].copy()
all_train_X.replace({False:0, True:1}, inplace = True)
final_logit = sm.Logit(all_train_y, all_train_X)

## Final Model Fit
final_result = final_logit.fit()

# Pickle Final Model
with open('glm_model.pickle', 'wb') as f:
    pickle.dump(final_result, f)
