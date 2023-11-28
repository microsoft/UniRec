# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import pandas as pd
import shutil
import sys
CUR_DIR = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(CUR_DIR)

from download_split_ml100k import download_zip_from_url, unzip_zip_file, split_train_test_set, k_core_filter, merge_category


def prepare_ml10m():
    # Download MovieLens-10m
    url = "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
    folder_path = os.path.expanduser("~/.unirec/dataset")

    zip_file_path = os.path.join(folder_path, "ml-10m.zip")
    if not os.path.exists(zip_file_path):
        zip_file_path = download_zip_from_url(url, folder_path)
    print(f"Load raw dataset from {zip_file_path}")

    extract_path = folder_path
    data_folder_path = os.path.join(extract_path, "ml-10m")
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path, exist_ok=True)

    if not os.listdir(data_folder_path):
        shutil.rmtree(data_folder_path)

        extract_path = unzip_zip_file(zip_file_path, extract_path)
        print(f"Unzip raw dataset compressed file into {extract_path}")
        os.rename(os.path.join(extract_path, "ml-10M100K"), data_folder_path)

    seed = 2023
    path = os.path.join(data_folder_path, "ratings.dat")
    item_info_path = os.path.join(data_folder_path, "movies.dat")
    # _path = os.path.dirname(__file__)
    # dataset_folder = os.path.abspath(os.path.join(_path, "../../data/"))
    outpath = os.path.join(folder_path, "ml-10m")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    rating_df = pd.read_csv(path, sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], header=None, engine='python')
    print(rating_df.head(3))

    cate_df = pd.read_csv(item_info_path, sep='::', header=None, names=['movieId', 'movieName', 'genre'], engine='python')
    cate_df = cate_df[cate_df['genre'] != '(no genres listed)']
    cate_df['genre'] = cate_df['genre'].apply(lambda x: x.split('|'))

    cates_token = list(set().union(*cate_df['genre']))
    cates_token.sort()
    print(cates_token)
    catetoken2idx = {x[1]: x[0] + 1 for x in enumerate(cates_token)}

    with open(os.path.join(outpath, "cate2id.json"), "w", encoding="utf-8") as jf:
        json.dump(catetoken2idx, jf)

    cate_df['genre'] = cate_df['genre'].apply(lambda x: [catetoken2idx[i] for i in x])
    cate_df = cate_df[['movieId', 'genre']]
    cate_df.columns = ['movieId', 'cateId']
    print(cate_df.head(3))

    # Merge item cate
    rating_df = pd.merge(rating_df, cate_df, how='inner', on=['movieId'])

    cate2idx, item2cate, num_cates = merge_category(rating_df, min_item_in_cate=200)

    user_col_name = 'userId'
    item_col_name='movieId'

    # Filtering
    data = rating_df.sort_values(by=['userId', 'timestamp'], ignore_index=True)
    print('original dataset size: {0}'.format(data.shape))
    data = data[data['rating']>=3].reset_index(drop=True)
    print('filter by rating>=3 dataset size: {0}'.format(data.shape))

    data = data.drop_duplicates(subset=['userId', 'movieId'], keep='last').reset_index(drop=True)
    print('drop_duplicates dataset size: {0}'.format(data.shape))

    data = k_core_filter(data, user_k=10, item_k=10, user_col_name=user_col_name, item_col_name=item_col_name)
    data = data.reset_index(drop=True)
    print('k-core filtered dataset size: {0}'.format(data.shape))

    # Map
    data = data.rename(columns={"userId": "user_id", "movieId": "item_id", "cateId": "cate_id"})

    users, items = data['user_id'].unique(), data['item_id'].unique()
    num_users, num_items = len(users), len(items)
    user_id_map, item_id_map = {id: i+1 for i, id in enumerate(users)}, {id: i+1 for i, id in enumerate(items)}
    data['item_id'], data['user_id'], data['cate_id'] = data['item_id'].apply(lambda x: item_id_map[x]), data['user_id'].apply(lambda x: user_id_map[x]), data['item_id'].apply(lambda x: list(item2cate[x]))
    print(num_users, num_items, num_cates)
    map_info = {"user": {str(k): v for k, v in user_id_map.items()}, 
                "item": {str(k): v for k, v in item_id_map.items()}, 
                "cate": {str(k): v for k, v in cate2idx.items()}}
    itemid2cate = data.set_index('item_id')['cate_id'].to_dict()

    data = data[['user_id', 'item_id', 'cate_id']]
    user_col_name = 'user_id'
    item_col_name='item_id'

    full_user_history = data.groupby(by=user_col_name, as_index=False).agg(list).reset_index(drop=True)
    full_user_history['item_seq'] = full_user_history[item_col_name].apply(lambda x: ",".join(map(str,x)))
    full_user_history = full_user_history[[user_col_name, 'item_seq']]
    full_user_history.to_csv(os.path.join(outpath, 'full_user_history.csv'), index=False, sep='\t')

    df_train0, df_test = split_train_test_set(data, col_name=user_col_name, col_names_2_return=None, seed=seed)
    df_train, df_valid = split_train_test_set(df_train0, col_name=user_col_name, col_names_2_return=None, seed=seed)
    print('size in Train/Valid/Test: {0} / {1} / {2}'.format(df_train.shape, df_valid.shape, df_test.shape))

    user_history = df_train0.groupby(by=user_col_name, as_index=False).agg(list).reset_index(drop=True)
    user_history['item_seq'] = user_history[item_col_name].apply(lambda x: ",".join(map(str,x)))
    user_history = user_history[[user_col_name, 'item_seq']]

    df_train = df_train[[user_col_name, item_col_name]]
    df_valid = df_valid[[user_col_name, item_col_name]]
    df_test = df_test[[user_col_name, item_col_name]]
    df_train.to_csv(os.path.join(outpath, 'train.csv'), index=False, sep='\t')
    df_valid.to_csv(os.path.join(outpath, 'valid.csv'), index=False, sep='\t')
    df_test.to_csv(os.path.join(outpath, 'test.csv'), index=False, sep='\t')
    user_history.to_csv(os.path.join(outpath, 'user_history.csv'), index=False, sep='\t')

    with open(os.path.join(outpath, "map.json"), "w", encoding="utf-8") as jf:
        json.dump(map_info, jf)

    with open(os.path.join(outpath, "item2cate.json"), "w", encoding="utf-8") as jf:
        json.dump(itemid2cate, jf)

    print(f"Processed dataset saved in {os.path.join(outpath)}.")
    return True


if __name__ == "__main__":
    prepare_ml10m()
