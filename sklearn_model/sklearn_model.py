import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

PATH_TO_DATA_TEST = '../data_test/'

np.random.seed(42)

df_train = pd.read_csv(os.path.join(PATH_TO_DATA_TEST, 'train.csv'))
df_test = pd.read_csv(os.path.join(PATH_TO_DATA_TEST, 'test.csv'))

share_of_zero_examples_in_test = ((df_test.shape[0] - df_test['item_cnt_day_lag_1'].fillna(0).nonzero()[0].shape[0]) /
                                  df_test.shape[0])
print('share_of_zero_examples_in_test', share_of_zero_examples_in_test)


def add_random_zero_examples(df, share_of_zero_values):
    number_of_nonzero_lag1_train = df.shape[0] - df['item_cnt_day_lag_1'].fillna(0).nonzero()[0].shape[0]
    additional_zero_rows_number = int(
        (share_of_zero_values * df.shape[0] + - number_of_nonzero_lag1_train) / (1 - share_of_zero_values)
    )
    shops = df['shop_id'].values
    items = df['item_id'].values

    random_shops = np.random.choice(shops, additional_zero_rows_number)
    random_items = np.random.choice(items, additional_zero_rows_number)

    additional_df = pd.DataFrame({
        'shop_id': random_shops,
        'item_id': random_items
    })
    for col in df.columns[2:]:
        additional_df[col] = 0

    return pd.concat([df_train, additional_df])


def prepare_array(df, is_target=True):
    df_cp = df.copy(deep=True)
    df_cp.fillna(0.0, inplace=True)
    # X_train = df_cp[['item_price_lag_1', 'item_cnt_day_lag_1',
    #                  'item_price_lag_2', 'item_cnt_day_lag_2',
    #                  'item_price_lag_3', 'item_cnt_day_lag_3']].values
    X_train = df_cp[['item_cnt_day_lag_1',
                     'item_cnt_day_lag_2',
                     'item_cnt_day_lag_3']].values
    if is_target:
        y_train = df_cp['item_cnt_day'].values
        y_train[y_train > 20] = 20
        y_train[y_train < 0] = 0
        return X_train, y_train
    return X_train

share_of_zero_examples_in_train = ((df_train.shape[0] - df_train['item_cnt_day_lag_1'].fillna(0).nonzero()[0].shape[0]) /
                                   df_train.shape[0])
print('share_of_zero_examples_in_train', share_of_zero_examples_in_train)
df_train = add_random_zero_examples(df_train, share_of_zero_examples_in_test)
share_of_zero_examples_in_train = ((df_train.shape[0] - df_train['item_cnt_day_lag_1'].fillna(0).nonzero()[0].shape[0]) /
                                   df_train.shape[0])
print('share_of_zero_examples_in_train', share_of_zero_examples_in_train)

X_train, y_train = prepare_array(df_train)
X_test = prepare_array(df_test, is_target=False)

# model = LinearRegression(normalize=True)
# model = Ridge(alpha=1.0)
model = GradientBoostingRegressor()

model.fit(X_train, y_train)
# print(model.coef_)
# print(model.intercept_)
y_train_pred = model.predict(X_train)
y_train_pred[y_train_pred > 20] = 20
y_train_pred[y_train_pred < 0] = 0
print('Train RMSE', np.sqrt(mean_squared_error(y_train, y_train_pred)))

y_test_total_pred = model.predict(X_test)
y_test_total_pred[y_test_total_pred > 20] = 20
y_test_total_pred[y_test_total_pred < 0] = 0

df_test['item_cnt_month'] = y_test_total_pred
df_test[['ID', 'item_cnt_month']].to_csv('submission_gbm_model.csv', index=False)

# ======= train1 ======= # ======= train1 ======= # ======= test ======= #
#           GBR          #        1.25791         #       1.11976        #
#         GBR (cnt)      #        1.26301         #       1.11773        #
#     linear regression  #        1.48400         #       1.21457        #
##########################################################################
