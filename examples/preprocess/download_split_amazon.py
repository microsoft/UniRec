from collections import Counter
from copy import deepcopy
import os
import json
import gzip
import requests
import zipfile
from tqdm import tqdm
import pandas as pd
import numpy as np
from ast import literal_eval

from unirec.facility.morec import MoRecDS

def download_targz_from_url(url: str, folder_path: str) -> str:
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

        return file_name

    else:
        # Return an empty string if the HTTP request was not successful
        return ""


def load_amazon_json_gz(filepath: str) -> pd.DataFrame:
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    i = 0
    df = {}
    for d in parse(filepath):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def load_data(filename: str, filename_csv: str, sep: str) -> pd.DataFrame:
    if os.path.exists(filename_csv):
        df = pd.read_csv(filename_csv, sep=sep)
    else:
        df = load_amazon_json_gz(filename)
        df.to_csv(filename_csv, sep=sep, index=None)
    return df


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



def preprocess_amazon(catgory_name: str):
    # Download MovieLens-100k
    url = {
        "reviews": f"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{catgory_name}.json.gz",
        "meta": f"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{catgory_name}.json.gz"
    }
    folder_path = os.path.expanduser(f"~/.unirec/dataset/{catgory_name}")

    data_frames = {
        "reviews": None,
        "meta": None
    }

    for d, url in url.items():
        zip_file_path = os.path.join(folder_path, f"{d}_{catgory_name}.json.gz")
        if not os.path.exists(zip_file_path):
            zip_file_path = download_targz_from_url(url, folder_path)
        # print(f"Load raw dataset from {zip_file_path}")
        data_frames[d] = load_data(zip_file_path, zip_file_path.replace(".json.gz", ".csv"), sep=",")


    user_col_name = 'reviewerID'
    item_col_name='asin'
    time_col_name='unixReviewTime'
    rating_col_name='overall'
    col_map = {user_col_name: "user_id", item_col_name: "item_id", time_col_name: "timestamp", rating_col_name: "rating"}

    review_df, meta_df = data_frames['reviews'], data_frames['meta']

    review_df = review_df.rename(columns=col_map)
    meta_df = meta_df.rename(columns=col_map)

    # Filtering
    data = review_df.sort_values(by=['user_id', 'timestamp'], ignore_index=True)
    print('original dataset size: {0}'.format(data.shape))
    data = data[data['rating']>3].reset_index(drop=True)
    print('filter by rating>3 dataset size: {0}'.format(data.shape))

    data = data.drop_duplicates(subset=['user_id', 'item_id'], keep='last').reset_index(drop=True)
    print('drop_duplicates dataset size: {0}'.format(data.shape))

    data = k_core_filter(data, user_k=5, item_k=5, user_col_name='user_id', item_col_name='item_id')
    data = data.reset_index(drop=True)
    print('k-core filtered dataset size: {0}'.format(data.shape))
    data = data.sort_values(by=['user_id', 'timestamp'], ignore_index=True)

    # Map
    users, items = data['user_id'].unique(), data['item_id'].unique()
    num_users, num_items = len(users), len(items)
    user_id_map, item_id_map = {id: i+1 for i, id in enumerate(users)}, {id: i+1 for i, id in enumerate(items)}
    data['item_id'], data['user_id'] = data['item_id'].apply(lambda x: item_id_map[x]), data['user_id'].apply(lambda x: user_id_map[x])
    map_info = {"user": {str(k): v for k, v in user_id_map.items()}, "item": {str(k): v for k, v in item_id_map.items()}}
    print(f"#Users: {num_users}, #Items: {num_items}")

    # Split
    user_col_name = 'user_id'
    item_col_name = 'item_id'

    data = data[['user_id', 'item_id']]
    df_train0, df_test = split_train_test_set(data, col_name=user_col_name, col_names_2_return=None)
    df_train, df_valid = split_train_test_set(df_train0, col_name=user_col_name, col_names_2_return=None)
    print('size in Train/Valid/Test: {0} / {1} / {2}'.format(df_train.shape, df_valid.shape, df_test.shape))

    user_history = df_train0.groupby(by=user_col_name, as_index=False).agg(list).reset_index(drop=True)
    user_history['item_seq'] = user_history[item_col_name].apply(lambda x: ",".join(map(str,x)))
    user_history = user_history[[user_col_name, 'item_seq']]

    df_train.to_csv(os.path.join(folder_path, 'train.csv'), index=False, sep='\t')
    df_valid.to_csv(os.path.join(folder_path, 'valid.csv'), index=False, sep='\t')
    df_test.to_csv(os.path.join(folder_path, 'test.csv'), index=False, sep='\t')
    user_history.to_csv(os.path.join(folder_path, 'user_history.csv'), index=False, sep='\t')

    # Revenue-Price
    item2price = deepcopy(meta_df[['item_id', 'price']])

    item2price.fillna(0.0, inplace=True)
    item2price['price'] = item2price['price'].apply(float)

    non_float = len(item2price) - (item2price['price'].apply(lambda x: isinstance(x, float) and x>=0)).sum()
    print(f"{non_float} raws price not float")
    item2price = item2price[item2price['item_id'].isin(item_id_map.keys())]
    print(f"{len(item2price['item_id'].unique())}/{len(item_id_map)} item prices existing in meta df")
    item2price['item_id'] = item2price['item_id'].apply(lambda x: item_id_map[x])
    item2price = item2price.sort_values(by='item_id', ignore_index=True)

    # Fairness-Category
    item2cat = deepcopy(meta_df[['item_id', 'categories']])
    if isinstance(item2cat['categories'][0], str):  # str of list
        item2cat['categories'] = item2cat['categories'].apply(lambda x: literal_eval(x)[0])
    else:   # list
        item2cat['categories'] = item2cat['categories'].apply(lambda x: x[0])

    item2cat = item2cat[item2cat['item_id'].isin(item_id_map.keys())]
    item2cat['item_id'] = item2cat['item_id'].apply(lambda x: item_id_map[x])
    item2cat = item2cat.sort_values('item_id', ignore_index=True)

    item2cat['1stcate'] = item2cat['categories'].apply(lambda x: x[0] if not x[0].lower()=='electronics' else x[1])

    n_groups = 10   # leave 0 as padding idx

    first_cate2count = Counter(item2cat['1stcate'])
    # print(first_cate2count)
    mis_cate = {}
    mis_count = 0
    most_common_cates = [x[0] for x in first_cate2count.most_common(n_groups-1)]
    for cate in first_cate2count:
        if cate not in most_common_cates:
            mis_cate[cate] = "Miscellaneous"
            mis_count += first_cate2count[cate]
        # else:
            # print(f"{cate}: {first_cate2count[cate]}")
    first_cate2count['Miscellaneous'] = mis_count
    # print(f"Miscellaneous: {mis_count}")

    item2cat['main_cate'] = item2cat['1stcate'].apply(lambda x: mis_cate[x] if x in mis_cate else x)
    all_cate = item2cat['main_cate'].unique()
    group_id_map = {cate: i+1 for i, cate in enumerate(all_cate)}

    map_info['group'] = group_id_map

    with open(os.path.join(folder_path, "map.json"), "w", encoding="utf-8") as jf:
        json.dump(map_info, jf)
    
    print(f"Map dict saved in {os.path.join(folder_path, 'map.json')}.")

    item2cat['category'] = item2cat['main_cate'].apply(lambda x: group_id_map[x])
    item2cat = item2cat[['item_id', 'category']]


    # Alignment-item popularity
    item_counts = df_train0['item_id'].value_counts()
    all_item_id = np.arange(1, num_items+1)
    item2pop = np.zeros(num_items + 1)
    item2pop[item_counts.index] = item_counts.to_numpy()
    item2align_group, _ = MoRecDS.group_item_by_attr(item2pop, n_groups, zero_as_group=False)

    item_meta = pd.DataFrame({
        "item_id": all_item_id,
        "weight": item2price.set_index("item_id").loc[all_item_id]['price'].to_numpy(),
        "fair_group": item2cat.set_index("item_id").loc[all_item_id]['category'].to_numpy(),
        "align_group": item2align_group[1:]
    })

    item_meta.to_csv(os.path.join(folder_path, "item_meta_morec.csv"), sep=",", index=None)

    print(f"Item meta file saved in {os.path.join(folder_path, 'item_meta_morec.csv')}.")



if __name__ == "__main__":
    category = "Electronics"
    preprocess_amazon(category)