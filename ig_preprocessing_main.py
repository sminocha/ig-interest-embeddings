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


# get user nodes, using name
# pandas data frame
# todo for processing captions and images.
# pd.read_csv(pass in path ), then grab column that has all captions
# url image or url image profile, can go run image_data_utils on that link to download that photo.
# process all things for singl euser, then add to new row in new csv. new

def main():
    data_path = 'backbones/dataset.csv'
    new_csv_location = 'backbones/modified_dataset.csv'
    download_path = "backbones/ig_downloaded_imgs/"
    # initialize pandas dataframe
    df = pd.read_csv(data_path) # to be populated with modified captions and 17 image paths
    # new_df = df # to be populated with modified captions and 17 image paths # DELETE?

    ## caption modifications

    # if caption is english, simply remove unnecessary elements (captions/tag/mentions), otherwise translate then remove unnecessary elements
    # replace old captions with new ones in df
    # df['description'] = np.where(caption.isEnglish(df['description']), caption.remove_unnecessary(df['description']), caption.translateOne(caption.remove_unnecessary(df['description'])))
    df['description'] = np.where(caption.isEnglish(caption.remove_unnecessary(df['description'])), caption.remove_unnecessary(df['description']), caption.removeThenTranslate(df['description']))
    # STILL TODO: filtration : scrap datapoints that still have weird characters etc

    ## image / image path modifications

    # add a column that transforms the ig links of each post to a download path leading to that post's image
    transform_post_to_path = lambda post_url: img.create_path(post_url, download_path)
    df['downloaded_image'] = df['url'].apply(transform_post_to_path)

    df.to_csv(new_csv_location) # write to csv




if __name__ == '__main__':
    main()
