"""
Description:
"""
import json
import pandas as pd
import numpy as np
import pickle
from data_segmentation_processing import fix_special_char



def data_formating(json_data):
    """
    [Summary]
    Input:
    Output: pd.DataFrame
    """
    # Import imputer and std_scaler from trined model set column names
    imputer = pickle.load(open('imputer.pickle','rb'))
    std_scaler = pickle.load(open('std_scaler.pickle','rb'))

    # Need to edit this out later -- not standard
    train_imputed_columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
       'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
       'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x32',
       'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41', 'x42',
       'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52',
       'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61', 'x62',
       'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71', 'x72',
       'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x80', 'x83', 'x84',
       'x85', 'x86', 'x87', 'x88', 'x89', 'x90', 'x91', 'x92', 'x93', 'x94',
       'x95', 'x96', 'x97', 'x98', 'x99']

    # Replace null with None
    for record in json_data:
        for key, value in record.items():
            if value == "null":
                record[key] = 0

    # Create a DataFrame from the JSON data
    df = pd.DataFrame(json_data)

    # Fix Special Characters
    df = fix_special_char(df)

    # Check if we have all the categorical columns
    categorical_cols = ['x5', 'x31', 'x81', 'x82']
    new_categorical_cols = []
    for category in categorical_cols:
        if category in df.columns:
            new_categorical_cols.append(category)

    # Impute and Scale data using the models from before
    imputed_df =  pd.DataFrame(imputer.transform(df.drop(columns=new_categorical_cols)), columns=df.drop(columns=new_categorical_cols).columns)
    imputed_std_df= pd.DataFrame(std_scaler.transform(imputed_df), columns=train_imputed_columns) # Only works if all columns exist in the datafram, need to fix 

    # Get Dummies
    n_rows = imputed_std_df.shape[0]
    # Assumes the categorical values exist in the dataset
    for cat_name in new_categorical_cols:
        if len(df[cat_name].unique()) > 1:
            dumb = pd.get_dummies(df[cat_name], drop_first=True, prefix= cat_name, prefix_sep='_', dummy_na=True, dtype=float)

        else:
            dumb = pd.DataFrame([1]*n_rows, columns=[f'{cat_name}_{df[cat_name].values[0]}'])

        imputed_std_df = pd.concat([imputed_std_df, dumb], axis=1, sort=False)

    # Select variables
    variables = ['x5_saturday','x81_July','x81_December','x31_japan','x81_October',
                    'x5_sunday','x31_asia','x81_February','x91','x81_May','x5_monday','x81_September',
                    'x81_March','x53','x81_November','x44','x81_June','x12','x5_tuesday','x81_August',
                    'x81_January','x62','x31_germany','x58','x56']

    # Select variable if in data, else generate a zero column for that variable
    processed_data = pd.DataFrame()
    for var in variables:
        # Add feature to our final dataset
        if var in imputed_std_df.columns:
            processed_data[var] = imputed_std_df[var]

        # Generate zero column
        else:
            processed_data[var] = [0] * n_rows

    del imputed_std_df

    return processed_data
