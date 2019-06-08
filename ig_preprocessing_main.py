'''
Preprocessing Architecture:
TODO:

get image data for each post. write crawler / scraper script
parse caption/description data
    - write translator for non english captions / descriptions. (Use googletrans? (https://pypi.org/project/googletrans/) )
    - take out emojis and other non utf? chars
    - Remove mentions and tags (to avoid redundancy and data leakage at the end)
    - Then do filtration (scrap datapoints that still have weird characters etc)

assumptions - for posts that have multiple images, we are just getting the first images (convenience)

result / output - New csv that has csv - processed text data, another column for images (relative paths to images)
'''

'''
TODO:
 - could eventually split this up into multiple files, one that handles caption parsing (translation function, filtering, etc.)
 - and this one that just houses the main function.
 - clean up comments
 - make filtration more robust
 - test at end to see how many records lost
'''

import csv
import json
import sys
from googletrans import Translator
import numpy as np
import pandas as pd
import data_utils as data
import caption_data_utils as caption
import img_data_utils as img
# from collections import deque
# from itertools import map

def dropSelectedCols(df):
    """delete selected columns from pandas df object"""
    cols_to_drop = ["website", "numberPosts", "numberFollowers", "numberFollowing", "filename", "date", "isVideo", "numberLikes"]
    df = df.drop(columns=cols_to_drop)
    return df

def performCaptionMods(df):
    """apply caption mods to df including caption translation + purging of emojis and other chars"""
    # if caption is english, simply remove unnecessary elements (captions/tag/mentions), otherwise translate then remove unnecessary elements
    # replace old captions with new ones in df
    df['description'] = np.where(caption.isEnglish(caption.remove_unnecessary(df['description'])), caption.remove_unnecessary(df['description']), caption.removeThenTranslate(df['description']))
    return df

def filterBadURLS(df):
    """filter out records with post urls that pointed to deleted/bad images"""
    df = df[df['isolated_img_url'] != "INVALID_URL"]

def addIsolatedImgCol(df):
    """add column that transforms links of each full ig post to to urls of just that isolated img"""
    extract_post_urls = lambda post_url: img.extract_img_url(post_url)
    df['isolated_img_url'] = df['url'].apply(extract_post_urls)

def downloadImgAddPath(df, download_path_base):
    """download images located at isolated image urls, add col with local path to that image """
    download_and_put_path = lambda indiv_img_url: img.download_img(indiv_img_url, download_path_base)
    df['downloaded_image'] = df['isolated_img_url'].apply(download_and_put_path)

def groupByKey(df, key, desired_cols):
    print(key)
    """group records by alias/ig handle, outputting df where each row is one user, i.e. descriptions concatenated, download paths comma separated, etc."""
    df['description'] = df[desired_cols].groupby([key])['description'].transform(lambda x: ' '.join(x))
    df['downloaded_image'] = df[desired_cols].groupby([key])['downloaded_image'].transform(lambda x: ','.join(x))
    df['mentions'] = df[desired_cols].groupby([key])['mentions'].transform(lambda x: ','.join(x))
    df['tags'] = df[desired_cols].groupby([key])['tags'].transform(lambda x: ','.join(x))
    df['isolated_img_url'] = df[desired_cols].groupby([key])['isolated_img_url'].transform(lambda x: ','.join(x))
    # TODO: delete duplicate records

def outputCaptionsMilestone(df, key, path, desired_cols):
    """output csv with captions translated and users aggregated, for debugging purposes"""
    df['description'] = df[desired_cols].groupby([key])['description'].transform(lambda x: ' '.join(x))
    df['mentions'] = df[desired_cols].groupby([key])['mentions'].transform(lambda x: ','.join(x))
    df['tags'] = df[desired_cols].groupby([key])['tags'].transform(lambda x: ','.join(x))
    print("printed intermediate dataset, with relevant captions mods...")
    df.to_csv(path)

def main():
    # relevant paths
    data_path = 'backbones/dataset.csv'
    new_csv_location = 'backbones/modified_dataset.csv'
    new_csv_milestone_location = 'backbones/captionsTranslated_dataset.csv'
    download_path_base = "backbones/ig_downloaded_imgs/"
    # other constants
    # numberPosts,website,urlProfile,username,numberFollowing,descriptionProfile,alias,numberFollowers,urlImgProfile,filename,date,urlImage,mentions,multipleImage,isVideo,localization,tags,numberLikes,url,description
    desired_cols_milestone = ["username", "alias", "tags", "url", "urlImage", "mentions", "description"]
    desired_cols = desired_cols_milestone + ["isolated_img_url", "downloaded_image"]

    df = pd.read_csv(data_path) # pandas df to be populated with modified captions and 17 image paths

    ## DEBUGGING
    test_url_1 = "https://www.instagram.com/p/BTdRaquBZTD/?taken-by=1misssmeis"
    test_url_2 = "https://www.instagram.com/p/BTdS7XgBe4X/?taken-by=1misssmeis"
    row_screen = [test_url_1, test_url_2]

    # dataframe with just rows selected by row_screen
    df = df.loc[df['url'].isin(row_screen)] # select several rows by their url
    ## DEBUGGING


    df = dropSelectedCols(df) # drop unnecessary cols

    ## caption modifications
    df = performCaptionMods(df) # translate to english, remove emojis / hashtags / etc
    outputCaptionsMilestone(df, 'alias', new_csv_milestone_location, desired_cols_milestone) # debugging - output csv with captions translated and users aggregated
    # STILL TODO: filtration : scrap datapoints that still have weird characters etc

    ## image path / url modifications
    addIsolatedImgCol(df) # convert col of post urls to isolated image urls
    filterBadURLS(df) # filter out post urls from above that pointed to deleted/bad images


    ## download images, add path col
    downloadImgAddPath(df, download_path_base) # download images located at isolated image urls, add col with local path to that image

    ## group by operations
    groupByKey(df, 'alias', desired_cols) # aggregate records corresponding to the same user

    ## publish
    df.to_csv(new_csv_location) # write to csv




if __name__ == '__main__':
    main()
