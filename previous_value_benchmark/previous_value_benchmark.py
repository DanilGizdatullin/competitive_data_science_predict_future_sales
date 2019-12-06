import os

import pandas as pd

PATH_TO_DATA = '../data/'

sales_train = pd.read_csv(os.path.join(PATH_TO_DATA, 'sales_train.csv'))
test = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

sales_month_grouped_train = (
    sales_train.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
)

sales_month_grouped_train_last_month = sales_month_grouped_train[sales_month_grouped_train['date_block_num'] == 33]

test_with_benchmark = pd.merge(
    test,
    sales_month_grouped_train_last_month,
    on=['shop_id', 'item_id'],
    how='left'
)[['ID', 'item_cnt_day']]

test_with_benchmark.columns = ['ID', 'item_cnt_month']
test_with_benchmark.fillna(0.0, inplace=True)
test_with_benchmark.loc[test_with_benchmark['item_cnt_month'] > 20, 'item_cnt_month'] = 20
test_with_benchmark.loc[test_with_benchmark['item_cnt_month'] < 0, 'item_cnt_month'] = 0
test_with_benchmark.to_csv('submission_previous_month_benchmark.csv', index=False)
