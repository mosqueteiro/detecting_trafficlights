from data_pipeline import QueryDatabase
import numpy as np
import pandas as pd



dataset = 'train2017'
store_dir = '/media/mosqueteiro/TOSHIBA EXT/detecting_trafficlights/'
user = 'mosqueteiro'
host = '/var/run/postgresql'


# train2017.query_database('SELECT * FROM street_annotations')
# annotations = train2017.df_query

def load_binary_train(**kwargs):
    dataset = kwargs.get('dataset', dataset)
    host = kwargs.get('host', host)
    user = kwargs.get('user', user)
    data_dir = kwargs.get('data_dir', store_dir)
    with QueryDatabase(dataset, host, user, data_dir=data_dir) as data:
        data.query_database('SELECT * FROM street_images')
        X = data.df_query.set_index('image_id').sort_index()
        data.query_database('SELECT * FROM street_annotations')
        y = data.df_query
        y = y.groupby('image_id')['category'].apply(
                                lambda x: 1 if 'traffic light' in set(x) else 0)
        y = y.sort_index()
        return (X, y)
