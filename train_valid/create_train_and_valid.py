import os

import pandas as pd
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
            'item_price': f'item_price_lag_{j}',
            'item_cnt_day': f'item_cnt_day_lag_{j}'
        }, inplace=True)
        total_month_dataset = pd.merge(total_month_dataset, train_sales, on=['shop_id', 'item_id'], how='left')

        cols = list(total_month_dataset.columns.values)
        cols.pop(cols.index('item_cnt_day'))
        cols.append('item_cnt_day')
        total_month_dataset = total_month_dataset[cols]

    return total_month_dataset


def make_dataset(
        train_df: pd.DataFrame,
        fisrt_month: int,
        last_month: int
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
            total_dataset = pd.concat([total_dataset, total_month_dataset], axis=1)

    return total_dataset


if __name__ == '__main__':
    sales_train = pd.read_csv(os.path.join(PATH_TO_DATA, 'sales_train.csv'))
    test = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

    train_train_sales_df = make_dataset(sales_train, 3, 23)
    train_train_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'train.csv'))
    print('TRAIN is ready!')

    train_valid_sales_df = make_dataset(sales_train, 23, 33)
    train_valid_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'valid.csv'))
    print('VALID is ready!')

    train_test_sales_df = make_dataset(sales_train, 33, 34)
    train_test_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TRAIN_DATA, 'test.csv'))
    print('TEST is ready!')

    # This is the right way
    # test_train_sales_df = make_dataset(sales_train, 3, 34)
    # But we don't want to wait to much time
    test_train_sales_df = pd.concat([train_train_sales_df, train_valid_sales_df, train_test_sales_df])
    test_train_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TEST_DATA, 'train.csv'))
    print('FINAL TRAIN is ready!')

    test_test_sales_df = make_lag_data(
        test_df=test,
        train_df=sales_train,
        current_month=34
    )
    test_test_sales_df.to_csv(os.path.join(PATH_TO_SAVE_TEST_DATA, 'test.csv'))
    print('FINAL TEST is ready!')
