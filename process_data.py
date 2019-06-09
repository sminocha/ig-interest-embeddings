import csv
import json
import sys
from googletrans import Translator
import numpy as np
import pandas as pd
import data_utils as data
import caption_data_utils as caption
import img_data_utils as img
from ig_preprocessing_main import dropSelectedCols, performCaptionMods, addIsolatedImgCol, filterBadURLS, downloadImgAddPath

def main():
    # relevant paths
    data_path = 'data/dataset.csv'
    new_csv_location = 'data/modified_dataset.csv'
    new_csv_milestone_location = 'data/captionsTranslated_dataset.csv'
    download_path_base = "data/ig_downloaded_imgs/"
    # other constants
    # numberPosts,website,urlProfile,username,numberFollowing,descriptionProfile,alias,numberFollowers,urlImgProfile,filename,date,urlImage,mentions,multipleImage,isVideo,localization,tags,numberLikes,url,description
    desired_cols_milestone = ["alias", "description"]
    desired_cols = desired_cols_milestone + ["downloaded_image"]

    src_df = pd.read_csv(data_path) # Source dataframe

    # # Save headers to csv file
    # dest_df = pd.DataFrame()
    # # dest_df.columns = desired_cols
    with open(new_csv_location, 'a') as f:
        # Process each user independently
        usernames = src_df['alias'].unique().tolist()
        idx_at = usernames.index('burakkahveci')
        for username in usernames[idx_at+1:]:
            print("Processing {}".format(username))
            # Get the rows corresponding to the user
            user_rows = src_df.loc[src_df['alias'] == username]
            # Process the user's descriptions
            user_rows = performCaptionMods(user_rows)
            user_rows = addIsolatedImgCol(user_rows) # convert col of post urls to isolated image urls
            user_rows = filterBadURLS(user_rows) # filter out post urls from above that pointed to deleted/bad images
            ## download images, add path col
            user_rows = downloadImgAddPath(user_rows, download_path_base) # download images located at isolated image urls, add col with local path to that image
            user_rows.drop(user_rows.columns.difference(desired_cols), 1, inplace=True) # drop unnecessary cols
            grouped_row = user_rows.groupby('alias').agg({'description': lambda c: ' '.join(c),
                                                          'downloaded_image': lambda c: ' '.join(c)}).reset_index()

            # Append new row to csv
            grouped_row.to_csv(f, header=False, index=False)


    #
    # ## caption modifications
    # df = performCaptionMods(df) # translate to english, remove emojis / hashtags / etc
    # outputCaptionsMilestone(df, 'alias', new_csv_milestone_location, desired_cols_milestone) # debugging - output csv with captions translated and users aggregated
    # # STILL TODO: filtration : scrap datapoints that still have weird characters etc
    #
    # ## image path / url modifications
    # addIsolatedImgCol(df) # convert col of post urls to isolated image urls
    # filterBadURLS(df) # filter out post urls from above that pointed to deleted/bad images
    #
    #
    # ## download images, add path col
    # downloadImgAddPath(df, download_path_base) # download images located at isolated image urls, add col with local path to that image
    #
    # ## group by operations
    # groupByKey(df, 'alias', desired_cols) # aggregate records corresponding to the same user
    #
    # ## publish
    # df.to_csv(new_csv_location) # write to csv




if __name__ == '__main__':
    main()
