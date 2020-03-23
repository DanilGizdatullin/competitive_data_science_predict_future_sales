import os

import numpy as np
import pandas as pd

PATH_TO_DATA = '../data/'
PATH_TO_VALID = '../data_valid/'
PATH_TO_TEST = '../data_test/'


def final_preparations(df):
    df = df[df.date_block_num > 3]

    def fill_na(df_n):
        for col in df_n.columns:
            if ('_lag_' in col) & (df_n[col].isnull().any()):
                if ('item_cnt' in col):
                    df_n[col].fillna(0, inplace=True)
        return df_n

    df = fill_na(df)
    return df


def split_train_valid_test(df):
    X_train = df[df.date_block_num < 32].drop(['item_cnt_month'], axis=1)
    Y_train = df[df.date_block_num < 32]['item_cnt_month']
    X_valid = df[df.date_block_num == 32].drop(['item_cnt_month'], axis=1)
    Y_valid = df[df.date_block_num == 32]['item_cnt_month']
    X_test = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_test = df[df.date_block_num == 33]['item_cnt_month']
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def split_train_test(df):
    X_train = df[df.date_block_num < 34].drop(['item_cnt_month'], axis=1)
    Y_train = df[df.date_block_num < 34]['item_cnt_month']
    X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    return X_train, Y_train, X_test


full_df_with_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'full_df_with_features.csv'))
full_df_with_features = final_preparations(full_df_with_features)

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = split_train_valid_test(full_df_with_features)
np.save(os.path.join(PATH_TO_VALID, 'X_train.npy'), X_train)
np.save(os.path.join(PATH_TO_VALID, 'Y_train.npy'), Y_train)
np.save(os.path.join(PATH_TO_VALID, 'X_valid.npy'), X_valid)
np.save(os.path.join(PATH_TO_VALID, 'Y_valid.npy'), Y_valid)
np.save(os.path.join(PATH_TO_VALID, 'X_test.npy'), X_test)
np.save(os.path.join(PATH_TO_VALID, 'Y_test.npy'), Y_test)

X_train, Y_train, X_test = split_train_test(full_df_with_features)
np.save(os.path.join(PATH_TO_TEST, 'X_train.npy'), X_train)
np.save(os.path.join(PATH_TO_TEST, 'Y_train.npy'), Y_train)
np.save(os.path.join(PATH_TO_TEST, 'X_test.npy'), X_test)
