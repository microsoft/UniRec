# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import requests
import zipfile
from tqdm import tqdm
import pandas as pd
import numpy as np

"""
# Acknowledge

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM Transactions on Interactive Intelligent
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
DOI=http://dx.doi.org/10.1145/2827872
"""

def download_zip_from_url(url: str, folder_path: str) -> str:
    """
    Download a zip file from a given URL and save it to the specified folder.

    Args:
        url (str): The URL of the zip file to download.
        folder_path (str): The path of the folder where the zip file should be saved.

    Returns:
        str: The path of the downloaded zip file on successful download. Empty string on error.

    Example:
        url = "https://example.com/sample.zip"
        folder_path = "/path/to/save/folder"
        result = download_zip_from_url(url, folder_path)
    """
    # Check if the folder_path exists, create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Extract the filename from the URL
    file_name = os.path.join(folder_path, os.path.basename(url))

    # Send an HTTP GET request to the URL with stream=True to download in chunks
    response = requests.get(url, stream=True, timeout=600)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the total file size in bytes
        file_size = int(response.headers.get('content-length', 0))

        # Create a progress bar using tqdm
        with open(file_name, 'wb') as file, tqdm(
            total=file_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

        # Check if the downloaded file is a valid zip archive
        if zipfile.is_zipfile(file_name):
            # Return the path to the downloaded zip file
            return file_name
        else:
            # Delete the invalid file and return an error message
            os.remove(file_name)
            return ""

    else:
        # Return an empty string if the HTTP request was not successful
        return ""


def unzip_zip_file(zip_file_path: str, extract_path: str = None) -> str:
    """
    Unzip a zip file to the specified destination path or a default path.

    Args:
        zip_file_path (str): The path to the zip file to be extracted.
        extract_path (str, optional): The destination path for extracted files.
            If None, a folder with the same name as the zip file (without .zip) will be created
            in the same directory as the zip file. Defaults to None.

    Returns:
        str: The path where the files were extracted.
    """
    # Get the directory containing the zip file
    zip_dir = os.path.dirname(zip_file_path)

    # Get the base name of the zip file (without .zip extension)
    zip_base_name = os.path.splitext(os.path.basename(zip_file_path))[0]

    # Determine the extraction path
    if extract_path is None:
        # Use the default extraction path in the same directory as the zip file
        extract_path = os.path.join(zip_dir, zip_base_name)

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Return the path where the files were extracted
    return extract_path


def get_valid_ids(df, col_name, k):
    frequency = df.groupby([col_name])[[col_name]].count()
    valid_id = frequency[frequency[col_name]>=k].index
    return valid_id


### leave-one-out split
def split_train_test_set(data: pd.DataFrame, col_name: str, col_names_2_return: list, seed: int = 0):
    if col_names_2_return is None:
        col_names_2_return = data.columns #.to_list()
    df_groupby = data.groupby(by=col_name, as_index=False) 
    df_test = df_groupby.nth(-1)[col_names_2_return]
    df_train = data.iloc[data.index.difference(df_test.index)] 
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True) 



def k_core_filter(df: pd.DataFrame, user_k=10, item_k=10, user_col_name='user_id', item_col_name='item_id'):
    num_users_prev, num_items_prev = len(df[user_col_name].unique()), len(df[item_col_name].unique()) 
    delta = True
    n_iter, max_iter = 0, 5

    while delta and n_iter < max_iter:
        valid_users = get_valid_ids(df, user_col_name, user_k)
        df = df[df[user_col_name].isin(valid_users)]

        valid_items = get_valid_ids(df, item_col_name, item_k)
        df = df[df[item_col_name].isin(valid_items)]

        num_users = len(valid_users)
        num_items = len(valid_items)

        delta = (num_users != num_users_prev) or (num_items != num_items_prev)
        print(f"Ite: {n_iter}, users: {num_users} / {num_users_prev}, items: {num_items} / {num_items_prev}")

        num_users_prev = num_users
        num_items_prev = num_items
        n_iter+=1
    return df


def prepare_ml100k():
    # Download MovieLens-100k
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    folder_path = os.path.expanduser("~/.unirec/dataset")

    zip_file_path = os.path.join(folder_path, "ml-100k.zip")
    if not os.path.exists(zip_file_path):
        zip_file_path = download_zip_from_url(url, folder_path)
    print(f"Load raw dataset from {zip_file_path}")

    extract_path = folder_path
    extract_path = unzip_zip_file(zip_file_path, extract_path)
    print(f"Unzip raw dataset compressed file into {extract_path}")

    seed = 2022
    path = os.path.join(extract_path, "ml-100k", "u.data")
    item_info_path = os.path.join(extract_path, "ml-100k", "u.item")
    # _path = os.path.dirname(__file__)
    # dataset_folder = os.path.abspath(os.path.join(_path, "../../data/"))
    outpath = os.path.join(folder_path, "ml-100k")
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    rating_df = pd.read_csv(path, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    rating_df.head(3)


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
    data = data.rename(columns={"userId": "user_id", "movieId": "item_id"})

    users, items = data['user_id'].unique(), data['item_id'].unique()
    num_users, num_items = len(users), len(items)
    user_id_map, item_id_map = {id: i+1 for i, id in enumerate(users)}, {id: i+1 for i, id in enumerate(items)}
    data['item_id'], data['user_id'] = data['item_id'].apply(lambda x: item_id_map[x]), data['user_id'].apply(lambda x: user_id_map[x])
    print(num_users, num_items)
    map_info = {"user": {str(k): v for k, v in user_id_map.items()}, 
                "item": {str(k): v for k, v in item_id_map.items()}}

    data = data.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
    data = data[['user_id', 'item_id']]
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

    df_train.to_csv(os.path.join(outpath, 'train.csv'), index=False, sep='\t')
    df_valid.to_csv(os.path.join(outpath, 'valid.csv'), index=False, sep='\t')
    df_test.to_csv(os.path.join(outpath, 'test.csv'), index=False, sep='\t')
    user_history.to_csv(os.path.join(outpath, 'user_history.csv'), index=False, sep='\t')

    with open(os.path.join(outpath, "map.json"), "w", encoding="utf-8") as jf:
        json.dump(map_info, jf)


    # fake item price and item categories to obtain item meta file for MoRec
    num_items = data['item_id'].max() + 1   # padding_idx=0
    # num_users = data['user_id'].max() + 1
    price_range = [20, 100]
    item_price = np.random.rand(num_items) * (price_range[1]-price_range[0]) + price_range[0]    #
    item_price[0] = 0.0 # padding item

    num_fair_group = 5
    item_fair_group = np.arange(1, num_fair_group+1)
    item_fair_group = np.concatenate([item_fair_group, np.random.randint(1, num_fair_group+1, (num_items-num_fair_group))])
    np.random.shuffle(item_fair_group)
    item_fair_group[0] = 0

    num_align_group = 5
    item_align_group = np.arange(1, num_align_group+1)
    item_align_group = np.concatenate([item_align_group, np.random.randint(1, num_align_group+1, (num_items-num_align_group))])
    np.random.shuffle(item_align_group)
    item_align_group[0] = 0

    item_meta_morec = pd.DataFrame({
        'item_id': np.arange(num_items), 
        'weight': item_price,
        'fair_group': item_fair_group,
        'align_group': item_align_group
    })
    item_meta_morec.to_csv(os.path.join(outpath, 'item_meta_morec.csv'), index=False, sep=',')

    print(f"Processed dataset saved in {os.path.join(outpath)}.")
    return True

if __name__ == "__main__":
    prepare_ml100k()
