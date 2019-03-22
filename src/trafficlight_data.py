from data_pipeline import QueryDatabase
import numpy as np
import pandas as pd


dataset = 'train2017'
store_dir = '/media/mosqueteiro/TOSHIBA EXT/detecting_trafficlights/'
user = 'mosqueteiro'
host = '/var/run/postgresql'


# train2017.query_database('SELECT * FROM street_annotations')
# annotations = train2017.df_query

def load_binary_train():
    with QueryDatabase(dataset, user, host, data_dir=store_dir) as data:
        data.query_database('SELECT * FROM street_images')
        X = data.df_query.set_index('image_id').sort_index()
        data.query_database('SELECT * FROM street_annotations')
        y = data.df_query
        y = y.groupby('image_id')['category'].apply(
                                lambda x: 1 if 'traffic light' in set(x) else 0)
        y = y.sort_index()
        return (X, y)













create_image_tbl = '''
CREATE TABLE street_images AS
    (SELECT
        image_id,
        file_name,
        coco_url,
        local_path,
        CASE WHEN
                'traffic light' IN st_an.category
            THEN
                1
            ELSE
                0
    FROM
        images
    JOIN
        (
        SELECT *
        FROM
            street_annotations
        GROUP BY
            image_id
        ) AS st_an

    );
'''
