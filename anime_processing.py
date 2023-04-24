import pandas as pd
import numpy as np
import re


import warnings
warnings.filterwarnings('ignore')


# function to clean the text from impurities
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    return text

# function to extract genres
def genre_extraction(anime_data):
    return anime_data['genre'].str.split(',').astype(str)

# function to extract an anime index
def anime_index_extraction(anime_data):
    return pd.Series(anime_data.index, index=anime_data['name']).drop_duplicates()

# function to better transform and organize data
def anime_data_processing(anime_data):
    anime_data['name'] = anime_data['name'].apply(text_cleaning)
    anime_data["rating"].replace({-1: np.nan}, inplace=True)
    anime_data = anime_data.dropna(axis = 0, how ='any')
    anime_data['genreClone'] = anime_data.loc[:, 'genre']
    anime_data['genreClone'] = anime_data['genreClone'].fillna('')
    anime_data['genreClone'] = anime_data['genreClone'].astype('str')
    anime_data['genreClone'] = anime_data['genreClone'].str.split(', ')
    genre_columns_temp=anime_data.genreClone.apply(pd.Series).stack().str.get_dummies().sum(level=0)
    anime_data = pd.concat([anime_data,genre_columns_temp],axis=1)
    anime_data = anime_data.drop(['genreClone'], axis=1)
    anime_data.loc[anime_data['rating']> 6 , 'success'] = '1'
    anime_data.loc[anime_data['rating']< 7 , 'success'] = '0'
    #anime_data.drop(anime_data[anime_data['episodes']=='Unknown'].index , inplace=True)
    #anime_data['episodes'] = anime_data['episodes'].astype(str).astype(int)
    del genre_columns_temp
    return anime_data

def genre_table_extraction(anime_data):
    anime_data_clone = anime_data.copy()
    genre_table = anime_data_clone.genre.apply(pd.Series).stack().str.get_dummies().sum(level=0)
    #anime_data = anime_data.drop(['genre'], axis=1)
    genre_table = pd.concat([anime_data['anime_id'],genre_table],axis=1)
    del anime_data_clone
    return genre_table

# function to merge the data between the list of anime and their votes
def anime_pivot_processing(anime_data, rating_data):
    anime_dataclone = anime_data[['anime_id', 'name']].copy()
    anime_fulldata=pd.merge(anime_dataclone,rating_data,on='anime_id')
    del anime_dataclone
    del rating_data
    anime_fulldata["rating"].replace({-1: np.nan}, inplace=True)
    anime_fulldata = anime_fulldata.dropna(axis = 0, how ='any') 
    counts = anime_fulldata['user_id'].value_counts()
    anime_fulldata = anime_fulldata[anime_fulldata['user_id'].isin(counts[counts >= 500].index)]
    anime_pivot=anime_fulldata.pivot_table(index='name',columns='user_id',values='rating').fillna(0)
    del anime_fulldata
    return anime_pivot
