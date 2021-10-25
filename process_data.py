from datetime import datetime
from types import new_class
import pandas as pd
import numpy as np
import sqlite3
import sys
import re
from pandas.core.reshape.concat import concat
from sklearn.preprocessing import MultiLabelBinarizer

#file_folder="D:/online courses/data science nanodegree/Disaster response/data"

#create disaster_response database for storing the transformed dataset
disaster_response_db=sqlite3.connect('disaster_response.db')
cur=disaster_response_db.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS disaster_response (message text, multilabel text)''')
disaster_response_db.commit()

def extract_all_categories(categories):
    '''extracting all categories of the disaster responses
    
    args:
        categories: str. A string containing all types of categories
    
    returns:
        allcategories_list: list. A list of all categories.
    '''
    allcategories_list=re.findall(r'([a-zA-Z_]+)',categories)
    return allcategories_list

def extract_categories(categories):
    '''extracting the specific categories as labels
    
    args:
        categories: str. A string containing all types of categories
        
    returns:
        categories_list: list. A list of categories of specific disaster response.
    '''
    categories_list=re.findall(r'([a-zA-Z_]+)-[1-2]',categories)
    return categories_list

def add_multilabels_columns(df):
    '''adding labels columns into the original dataframe based on the categories column in original dataframe

    args:
        df: pandas dataframe. Dataframe including a categories column
    
    returns:
        new_df: pandas dataframe. A new dataframe including the multilables as columns
    '''
    all_categories=extract_all_categories(df['categories'].iloc[0])
    mlb=MultiLabelBinarizer(classes=all_categories)
    labels_df=pd.DataFrame(data=mlb.fit_transform(df['categories'].apply(extract_categories).values),columns=all_categories)
    new_df = pd.concat([df,labels_df],axis=1)

    #print(mlb.fit_transform(df['categories'].apply(extract_categories).values))

    return new_df


def main(paths):
    '''read csv files, combine the dataset and load the combined dataset into a database

    args:
        paths: a list of file paths. paths[0]: path to the messages dataset.
               paths[1]: path to the labels dataset.

    returns： 
        disaster_response_db: a sqllite database storing transformed dataset.
    '''
    
    dataframe_list=[pd.read_csv(r'data/'+path) for path in paths]
    
    df_with_multilabels=add_multilabels_columns(dataframe_list[1])
    mes_cat_df=dataframe_list[0].merge(df_with_multilabels,on='id')

    #drop original and genre columns in the combined dataframe
    #mes_cat_df=mes_cat_df.drop(columns=['id','original','genre'])

    mes_cat_df.to_sql('disaster_response',disaster_response_db,if_exists='replace')

if __name__=='__main__':
    paths=sys.argv[1:]
    main(paths)
