from data_pipeline import QueryDatabase
import numpy as np
import pandas as pd


# train2017.query_database('SELECT * FROM street_annotations')
# annotations = train2017.df_query

def load_binary_train(**kwargs):
    dataset = kwargs.get('dataset', 'train2017')
    host = kwargs.get('host', 'localhost')
    user = kwargs.get('user', 'postgres')
    data_dir = kwargs.get('data_dir', 'data/coco/')
    with QueryDatabase(dataset, host, user, data_dir) as data:
        data.query_database('SELECT * FROM street_images')
        X = data.df_query.set_index('image_id').sort_index()
        data.query_database('SELECT * FROM street_annotations')
        y = data.df_query
        y = y.groupby('image_id')['category'].apply(
                                lambda x: 1 if 'traffic light' in set(x) else 0)
        y = y.sort_index()
        return (X, y)
