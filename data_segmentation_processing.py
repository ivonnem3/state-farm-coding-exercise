"""
Description:
This python script is used to processed the training dataset for our GLM Model including, special characted replacement,
one hot encoding, imputing and standarizing our model. 
"""
# Load Libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def fix_special_char(data):
    """
    Description:
            This function is able replaces all the special characters known in our datasets
    Attributes:
            data (pd.DataFrame) : Original Dataframe
    Output:
            data (pd.DataFrame) : Updated Dataframe
    """
    # Fix money and percents
    if 'x12' in data.columns:
        data['x12'] = data['x12'].str.replace('$','')
        data['x12'] = data['x12'].str.replace(',','')
        data['x12'] = data['x12'].str.replace(')','')
        data['x12'] = data['x12'].str.replace('(','-')
        data['x12'] = data['x12'].astype(float)

    if 'x63' in data.columns:
        data['x63'] = data['x63'].str.replace('%','')
        data['x63'] = data['x63'].astype(float)

    return data



def create_dummies(data,data_imputed_std):
    """
    Description:
        This functions create the dummy variables for the knonw categorical values in our dataset and
        appends them to our processed dataset
    Attributes:
        data (pd.Dataframe) : Original dataset
        data_imputed_std (pd.DataFrame): Impute and processed dataframe
    Output:
        data_imputed_std (pd.DataFrame): Processed dataframe with added dummy one-hot encoding
    """
    if 'x5' in data.columns:
        dumb5 = pd.get_dummies(data['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True, dtype=float)
        data_imputed_std = pd.concat([data_imputed_std, dumb5], axis=1, sort=False)

        del dumb5

    if 'x31' in data.columns:
        dumb31 = pd.get_dummies(data['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True, dtype=float)
        data_imputed_std = pd.concat([data_imputed_std, dumb31], axis=1, sort=False)

        del dumb31

    if 'x81' in data.columns:
        dumb81 = pd.get_dummies(data['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True, dtype=float)
        data_imputed_std = pd.concat([data_imputed_std, dumb81], axis=1, sort=False)

        del dumb81

    if 'x82' in data.columns:
        dumb82 = pd.get_dummies(data['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True, dtype=float)
        data_imputed_std = pd.concat([data_imputed_std, dumb82], axis=1, sort=False)

        del dumb82

    return data_imputed_std



def training_data_processing(data):
    """
    Description:
        This function takes care of all the data processing needed for training and tuning our GLM model.

    Input:
        data (pd.DataFrame): Preprocessed dataframe
    Output:
        train_imputed_std (pd.DataFrame): Imputed, standarized, one hot encoded dataset
        val_imputed_std (pd.DataFrame): Imputed, standarized, one hot encoded dataset
        test_imputed_std (pd.DataFrame): Imputed, standarized, one hot encoded dataset
    """
    # Create a deep copy of dataframe
    data = data.copy(deep=True)

    # Fix money and percents
    data = fix_special_char(data)

    # Creating the train/val/test set
    x_train, x_val, y_train, y_val = train_test_split(data.drop(columns=['y']), data['y'], test_size=0.1, random_state=13)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=4000, random_state=13)

    # Concat features and target variables
    train = pd.concat([x_train, y_train], axis=1, sort=False).reset_index(drop=True)
    val = pd.concat([x_val, y_val], axis=1, sort=False).reset_index(drop=True)
    test = pd.concat([x_test, y_test], axis=1, sort=False).reset_index(drop=True)

    # With mean imputation from Train set
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    std_scaler = StandardScaler()

    # Prepare train dataset
    train_imputed = pd.DataFrame(imputer.fit_transform(train.drop(columns=['y', 'x5', 'x31',  'x81' ,'x82'])), columns=train.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    train_imputed_std = pd.DataFrame(std_scaler.fit_transform(train_imputed), columns=train_imputed.columns)

    train_imputed_std = create_dummies(train,train_imputed_std)
    train_imputed_std = pd.concat([train_imputed_std, train['y']], axis=1, sort=False)

    # Prepare validation dataset
    val_imputed = pd.DataFrame(imputer.transform(val.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
    val_imputed_std = pd.DataFrame(std_scaler.transform(val_imputed), columns=train_imputed.columns)

    val_imputed_std = create_dummies(val,val_imputed_std)
    val_imputed_std = pd.concat([val_imputed_std, val['y']], axis=1, sort=False)

    # Prepare Test Set
    test_imputed = pd.DataFrame(imputer.transform(test.drop(columns=['y', 'x5', 'x31', 'x81' ,'x82'])), columns=train.drop(columns=['y','x5', 'x31', 'x81', 'x82']).columns)
    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=train_imputed.columns)

    test_imputed_std = create_dummies(test,test_imputed_std)
    test_imputed_std = pd.concat([test_imputed_std, test['y']], axis=1, sort=False)

    # Pickle Imputer
    with open('imputer.pickle', 'wb') as f:
        pickle.dump(imputer, f)

    # Pickle StandardScaler
    with open('std_scaler.pickle', 'wb') as f:
        pickle.dump(std_scaler, f)

    return train_imputed_std, val_imputed_std, test_imputed_std
