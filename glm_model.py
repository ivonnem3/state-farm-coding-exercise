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

## Local Path Set up
path = os.getcwd()
os.chdir('state-farm-coding-exercise')

## Load Train & Test data
#-------------------------------------------
raw_train = pd.read_csv('exercise_26_train.csv')
raw_test = pd.read_csv('exercise_26_test.csv')


## Feature Engineering
#-------------------------------------------
train_val = raw_train.copy(deep=True)

# Fixing the money and percents#
train_val['x12'] = train_val['x12'].str.replace('$','')
train_val['x12'] = train_val['x12'].str.replace(',','')
train_val['x12'] = train_val['x12'].str.replace(')','')
train_val['x12'] = train_val['x12'].str.replace('(','-')
train_val['x12'] = train_val['x12'].astype(float)
train_val['x63'] = train_val['x63'].str.replace('%','')
train_val['x63'] = train_val['x63'].astype(float)

# Creating the train/val/test set
x_train, x_val, y_train, y_val = train_test_split(train_val.drop(columns=['y']), train_val['y'], test_size=0.1, random_state=13)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

# Smashing sets back together
train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

# With mean imputation from Train set
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
train_imputed = pd.DataFrame(imputer.fit_transform(train.drop(columns=['y', 'x5', 'x31',  'x81' ,'x82'])), columns=train.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
std_scaler = StandardScaler()
train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

# Create dummies
dumb5 = pd.get_dummies(train['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(train['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(train['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(train['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
train_imputed_std = pd.concat([train_imputed_std, dumb82], axis=1, sort=False)
train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)

del dumb5, dumb31, dumb81, dumb82


## Initial Feature Selection
#-------------------------------------------
exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
exploratory_LR.fit(train_imputed_std.drop(columns=['y']), train_imputed_std['y'])
exploratory_results = pd.DataFrame(train_imputed_std.drop(columns=['y']).columns).rename(columns={0:'name'})
exploratory_results['coefs'] = exploratory_LR.coef_[0]
exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
var_reduced = exploratory_results.nlargest(25,'coefs_squared')


## Preliminary Model
#-------------------------------------------
variables = var_reduced['name'].to_list()
train_y = train_imputed_std['y']
train_X = train_imputed_std[variables].copy()
train_X.replace({False:0, True:1}, inplace = True)
logit = sm.Logit(train_y, train_X)

# fit the model
result = logit.fit()

## Prepping the validation set
#-------------------------------------------
val_imputed = pd.DataFrame(imputer.transform(val.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
val_imputed_std = pd.DataFrame(std_scaler.transform(val_imputed), columns=train_imputed.columns)

dumb5 = pd.get_dummies(val['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(val['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(val['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(val['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
val_imputed_std = pd.concat([val_imputed_std, dumb82], axis=1, sort=False)
val_imputed_std = pd.concat([val_imputed_std, val['y']], axis=1, sort=False)


## Prepping the test set
#-------------------------------------------
test_imputed = pd.DataFrame(imputer.transform(test.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=train_imputed.columns)

# 3 create dummies

dumb5 = pd.get_dummies(test['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

dumb31 = pd.get_dummies(test['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

dumb81 = pd.get_dummies(test['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

dumb82 = pd.get_dummies(test['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
test_imputed_std = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)
test_imputed_std = pd.concat([test_imputed_std, test['y']], axis=1, sort=False)


## Finaliz the Model
#-------------------------------------------
train_and_val = pd.concat([train_imputed_std, val_imputed_std])
all_train = pd.concat([train_and_val, test_imputed_std])
variables = var_reduced['name'].to_list()
all_train_y = all_train['y']
all_train_X = all_train[variables].copy()
all_train_X.replace({False:0, True:1}, inplace = True)
final_logit = sm.Logit(all_train_y, all_train_X)

# fit the model
final_result = final_logit.fit()


# Save model
with open('glm_model.pickle', 'wb') as f:
    pickle.dump(final_result, f)
