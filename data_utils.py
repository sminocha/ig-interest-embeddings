import ast
import functools
import collections
import numpy as np
import pandas as pd
import re
import string
import csv
import sys
from googletrans import Translator
import caption_data_utils as caption
import img_data_utils as img

rgx = re.compile('[^' + ''.join(string.printable) + ']')

def parse_hashtags(hashtag_str):
    """Return list of parsed hashtags"""
    return ast.literal_eval(hashtag_str)


def get_user_hashtags(df, username):
    """Aggregate all user hashtags into single set"""
    user_rows = df.loc[df['username'] == username]
    hashtags = user_rows['tags'].tolist()
    all_tags = set()
    for tags in hashtags:
        parsed_tags = parse_hashtags(tags)
        all_tags.update(set(parsed_tags))
    return all_tags


def filter_users(df):
    """Remove users that have no hashtags"""
    df['tags'].replace('', np.nan)
    df.dropna(subset=['username','tags'], inplace=True)
    return df


def clean_username(username):
    """Removes spaces and lowercases characters in username"""
    username =  ''.join([i for i in username.lower().replace(' ', '_') if not i.isdigit()])
    username = rgx.sub('', username)
    return username


def get_user_nodes(df):
    """Return list of (username, tags_set) tuples"""
    df = filter_users(df)
    user_nodes = []
    username_mapping = dict()
    for username in df['username'].unique().tolist():
        user_tags = get_user_hashtags(df, username)

        parsed_username = clean_username(username)
        username_mapping[parsed_username] = username

        user_node = (parsed_username, user_tags)
        user_nodes.append(user_node)

    # Save username mapping
    with open('usernames.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(username_mapping.items())

    return user_nodes


def report_hash_tag_stats(user_nodes):
    """Generates preliminary stats of hashtag data"""
    all_tags = list(map(lambda u: list(u[1]), user_nodes))
    all_tags = list(functools.reduce(lambda u,v : u+v, all_tags))
    # Compute freq for each tag
    counter = collections.Counter(all_tags)
    print("Number of unique tags: {}".format(len(set(all_tags))))
    print("5 most common tag: {}".format(counter.most_common(5)))
    print("Avg freqeency of tags: {}".format('NOT IMPLEMENTED YET'))


# we will select test rows by (post) url
test_url_1 = "https://www.instagram.com/p/BTdRaquBZTD/?taken-by=1misssmeis"
test_url_2 = "https://www.instagram.com/p/BTdS7XgBe4X/?taken-by=1misssmeis"
row_screen = [test_url_1, test_url_2]
def output_test_row(row_screen):
    data_path = 'backbones/dataset.csv'
    new_csv_location = 'backbones/modified_dataset.csv'
    download_path_base = "backbones/ig_downloaded_imgs/"
    desired_cols = ["username", "tags", "url", "urlImage", "mentions", "description", "indiv_img_url", "downloaded_image"]
    cols_to_drop = ["website", "numberPosts", "numberFollowers", "numberFollowing", "filename", "date", "isVideo", "numberLikes"]
    # initialize pandas dataframe
    df = pd.read_csv(data_path)
    # dataframe with just rows selected by row_screen
    df = df.loc[df['url'].isin(row_screen)] # select several rows by their url

    # remove unnecessary cols
    df = df.drop(columns=cols_to_drop)

    # perform manipulations made in ig_preprocessing_main.py
    df['description'] = np.where(caption.isEnglish(caption.remove_unnecessary(df['description'])), caption.remove_unnecessary(df['description']), caption.removeThenTranslate(df['description']))

    # add column that transforms ig links of each post to individual img urls
    extract_post_urls = lambda post_url: img.extract_img_url(post_url)
    df['indiv_img_url'] = df['url'].apply(extract_post_urls)

    # filter out records with "INVALID_URL" indiv_img_urls
    df = df[df['indiv_img_url'] != "INVALID_URL"]

    # add a column that transforms the individual img urls to a download path leading to that image
    # includes actual download + placement of image
    download_and_put_path = lambda indiv_img_url: img.download_img(indiv_img_url, download_path_base)
    df['downloaded_image'] = df['indiv_img_url'].apply(download_and_put_path)

    df['description'] = df[desired_cols].groupby(['username'])['description'].transform(lambda x: ' '.join(x))
    df['downloaded_image'] = df[desired_cols].groupby(['username'])['downloaded_image'].transform(lambda x: ','.join(x))
    # df['urlImage'] = df[desired_cols].groupby(['username'])['urlImage'].transform(lambda x: ','.join(x))
    df['mentions'] = df[desired_cols].groupby(['username'])['mentions'].transform(lambda x: ','.join(x))
    df['tags'] = df[desired_cols].groupby(['username'])['tags'].transform(lambda x: ','.join(x))
    df['indiv_img_url'] = df[desired_cols].groupby(['username'])['indiv_img_url'].transform(lambda x: ','.join(x))

    # df = df[desired_cols].drop_duplicates()

    # df.groupby(['username'])['description'].apply(' '.join).reset_index() # concatanate post descriptions from each user using space
    # df.groupby(['username'])['download_img'].apply(', '.join).reset_index() # comma separate image paths of images from each user using space

    df.to_csv(new_csv_location) # write to csv

output_test_row(row_screen)
