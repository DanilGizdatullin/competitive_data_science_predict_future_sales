import os

import numpy as np
import pandas as pd
# from pandas import HDFStore
from tqdm import tqdm

PATH_TO_DATA = '../data'
PATH_TO_SAVE_TRAIN_DATA = '../data_valid'
PATH_TO_SAVE_TEST_DATA = '../data_test'
NUMBER_OF_PREVIOUS_MONTHS = 3


def make_lag_data(
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        current_month: int
) -> pd.DataFrame:
    """Add lag features to test dataframe

    :param test_df:
    :param train_df:
    :param current_month:
    :return:
    """
    total_month_dataset = test_df
    for j in range(1, NUMBER_OF_PREVIOUS_MONTHS + 1):
        train_month = current_month - j
        train_sales = train_df[train_df['date_block_num'] == train_month].copy(deep=True)
        train_sales = train_sales[['shop_id', 'item_id', 'item_price', 'item_cnt_day']]
        train_sales.rename(columns={
            'item_price': f'item_shop_price_lag_{j}',
            'item_cnt_day': f'item_shop_cnt_day_lag_{j}'
        }, inplace=True)
        total_month_dataset = pd.merge(total_month_dataset, train_sales, on=['shop_id', 'item_id'], how='left')

    return total_month_dataset


def add_random_zero_examples(df, share_of_zero_values):
    number_of_zero_lag1_train = df.shape[0] - df['item_shop_cnt_day_lag_1'].fillna(0).nonzero()[0].shape[0]
    additional_zero_rows_number = int(
        (share_of_zero_values * df.shape[0] - number_of_zero_lag1_train) / (1 - share_of_zero_values)
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

    return pd.concat([df, additional_df])


def make_dataset(
        train_df: pd.DataFrame,
        fisrt_month: int,
        last_month: int,
        items: pd.DataFrame
) -> pd.DataFrame:

    total_dataset = None
    for valid_month in tqdm(range(fisrt_month, last_month)):
        valid_sales = train_df[train_df['date_block_num'] == valid_month].copy(deep=True)
        valid_sales = valid_sales[['shop_id', 'item_id', 'item_cnt_day']]
        total_month_dataset = make_lag_data(
            test_df=valid_sales,
            train_df=train_df,
            current_month=valid_month
        )

        if total_dataset is None:
            total_dataset = total_month_dataset
        else:
            total_dataset = pd.concat([total_dataset, total_month_dataset], axis=0)

    print((total_dataset.shape[0] - total_dataset['item_shop_cnt_day_lag_1'].nonzero()[0].shape[0]) / total_dataset.shape[0])
    total_dataset = add_random_zero_examples(total_dataset, 0.866)
    print((total_dataset.shape[0] - total_dataset['item_shop_cnt_day_lag_1'].nonzero()[0].shape[0]) / total_dataset.shape[0])
    total_dataset = pd.merge(total_dataset, items, how='left', on='item_id')

    cols = list(total_dataset.columns.values)
    cols.pop(cols.index('item_cnt_day'))
    cols.append('item_cnt_day')
    total_dataset = total_dataset[cols]

    return total_dataset


if __name__ == '__main__':
    sales_train = pd.read_csv(os.path.join(PATH_TO_DATA, 'sales_train.csv'))
    sales_train = (
        sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg(
            {'item_price': np.mean, 'item_cnt_day': np.sum}).reset_index()
    )
    test = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

    items = pd.read_csv(os.path.join(PATH_TO_DATA, 'items.csv'))
    # item_categories = pd.read_csv(os.path.join(PATH_TO_DATA, 'item_categories.csv')
    items = items[['item_id', 'item_category_id']]

    # VALIDATION
    train_train_sales_df = make_dataset(sales_train, 3, 23, items)
    train_train_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'train.csv'), index=False)
    print('TRAIN is ready!')

    train_valid_sales_df = make_dataset(sales_train, 23, 33, items)
    train_valid_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'valid.csv'), index=False)
    print('VALID is ready!')

    train_test_sales_df = make_dataset(sales_train, 33, 34, items)
    train_test_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'test.csv'), index=False)
    print('TEST is ready!')

    # TEST
    test_train_sales_df = make_dataset(sales_train, 3, 34, items)
    test_train_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TEST_DATA, 'train.csv'), index=False)
    print('FINAL TRAIN is ready!')

    test_test_sales_df = make_lag_data(
        test_df=test,
        train_df=sales_train,
        current_month=34
    )
    test_test_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TEST_DATA, 'test.csv'), index=False)
    print('FINAL TEST is ready!')

    # hdf5
    # hdf = HDFStore(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'storage.h5'))
    # for i in range(14, 23, 1):
    #     if i+3 <= 34:
    #         train_train_sales_df = make_dataset(sales_train, i, i+3)
    #     else:
    #         train_train_sales_df = make_dataset(sales_train, i, 34)
    #     # hdf.put(f'train/p_{i}', train_train_sales_df)
    #     if i > 3:
    #         hdf.put('train_data', train_train_sales_df, format='table', data_columns=True, append=True)
    #     else:
    #         hdf.put('train_data', train_train_sales_df, format='table', data_columns=True)
    # train_train_sales_df.to_csv(os.path.join('../data_temp', f'train_{i}.csv'), index=False)
