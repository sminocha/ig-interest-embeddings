import ast
import functools
import collections
import numpy as np
import re
import string
import csv

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
