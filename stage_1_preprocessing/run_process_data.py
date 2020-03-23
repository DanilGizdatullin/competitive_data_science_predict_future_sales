from stage_1_preprocessing.preprocess_train_test_data import train_test_preprocess
from stage_1_preprocessing.preprocess_categories import categories_preprocess
from stage_1_preprocessing.preprocess_items import items_preprocess
from stage_1_preprocessing.preprocess_shops import shops_preprocess

if __name__ == '__main__':
    train_test_preprocess()
    categories_preprocess()
    items_preprocess()
    shops_preprocess()
