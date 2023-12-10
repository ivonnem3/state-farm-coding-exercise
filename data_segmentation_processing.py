"""
Description:
"""
# Load Libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def fix_special_char(data):
    """
    Description:
    Input: pd.DataFrame
    Output:
    """
    # Fix money and percents
    data['x12'] = data['x12'].str.replace('$','')
    data['x12'] = data['x12'].str.replace(',','')
    data['x12'] = data['x12'].str.replace(')','')
    data['x12'] = data['x12'].str.replace('(','-')
    data['x12'] = data['x12'].astype(float)
    data['x63'] = data['x63'].str.replace('%','')
    data['x63'] = data['x63'].astype(float)

    return data



def create_dummies(data,data_imputed_std):
    """
    Description:
    Input: pd.DataFrame
    Output:
    """
    dumb5 = pd.get_dummies(data['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(data['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(data['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(data['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    data_imputed_std = pd.concat([data_imputed_std, dumb82], axis=1, sort=False)

    del dumb5, dumb31, dumb81, dumb82

    return data_imputed_std



def training_data_processing(data):
    """
    Description:
    Input: pd.DataFrame
    Output:
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

    return train_imputed_std, val_imputed_std, test_imputed_std



def prediction_data_processing(data):
    """
    Description:
    Input: pd.DataFrame
    Output:
    """
    # Create a deep copy of dataframe
    data = data.copy(deep=True)

    # Fix money and percents
    data = fix_special_char(data)

    # Impute data
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data.drop(columns=['x5', 'x31',  'x81' ,'x82'])), columns=data.drop(columns=[ 'x5', 'x31', 'x81', 'x82']).columns)
    std_scaler = StandardScaler()
    data_imputed_std = pd.DataFrame(std_scaler.fit_transform(data_imputed), columns=data_imputed.columns)

    # Create Dummies
    dumb5 = pd.get_dummies(data['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True, dtype=float)
    data_imputed_std = pd.concat([data_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(data['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True, dtype=float)
    data_imputed_std = pd.concat([data_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(data['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True, dtype=float)
    data_imputed_std = pd.concat([data_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(data['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True, dtype=float)
    data_imputed_std = pd.concat([data_imputed_std, dumb82], axis=1, sort=False)

    del dumb5, dumb31, dumb81, dumb82

    # Select variables if they are on the dataframe if not then create a zero-column
    variables = ['x5_saturday','x81_July','x81_December','x31_japan','x81_October',
                 'x5_sunday','x31_asia','x81_February','x91','x81_May','x5_monday','x81_September',
                 'x81_March','x53','x81_November','x44','x81_June','x12','x5_tuesday','x81_August',
                 'x81_January','x62','x31_germany','x58','x56']

    n_rows = data_imputed_std.shape[0]
    processed_data = pd.DataFrame()
    for var in variables:
        if var in data_imputed_std.columns:
            processed_data[var] = data_imputed_std[var]

        else:
            processed_data[var] = [0] * n_rows

    return processed_data
