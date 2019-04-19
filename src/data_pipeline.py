'''
File:   data_pipeline.py
Author: Nathan James
Date:   03/15/19

Pipeline for filtering and downloading dataset from cocodataset.org
'''

from psycopg2 import connect, sql
from psycopg2.extras import execute_batch, RealDictCursor
from datetime import datetime
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import requests
import os

from tqdm import tqdm

''' Data pipeline class '''
class DataPipeline(object):
    def __init__(self, dataset, user, host, port='5432', data_dir=None):
        self.dataset = dataset
        self.data_dir = data_dir
        if data_dir:
            assert os.path.exists(data_dir), \
                'This directory does not exist.\n' + \
                'Please check the path and try again.'
            self.data_dir = '{}coco/{}/'.format(data_dir, self.dataset)
        self.connect_sql(dbname=dataset, user=user, host=host)

    def __del__(self):
        self.cursor.close()
        # print('Cursor closed.')
        self.connxn.close()
        # print('Connection closed.')
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def connect_sql(self, dbname, user='postgres', host='/tmp', port='5432'):
        print('Connecting to PostgreSQL server....')
        self.connxn = connect(dbname=dbname, user=user, host=host, port=port)
        print('\tConnected.')
        self.cursor = self.connxn.cursor()

    def check_location(self, image_name):
        path = '{}{}'.format(data_dir, image_name)
        return os.path.exists(path)


class BuildDatabase(DataPipeline):
    def __init__(self, dataset, user, host, data_dir=None):
        super().__init__(dataset, user, host)
        self.coco = None
        self.tables = None
        self.data_dir = data_dir


    def build_sql(self, coco_dir):
        print('Building Database...')

        # print('Setting up SQL tables')
        # self.create_tables('coco_dataset.sql')

        self.load_json(coco_dir)

        for table,entries in self.tables.items():
            self.insert_into_table(table, entries, pages=1000)


        # print('Adding relational constraints')
        # self.create_tables('sql_constraints.sql')

        print('Finished building database.')


    def load_json(self, coco_dir):
        print('Loading COCO dataset {} information....'.format(self.dataset))
        path = '{}/annotations/instances_{}.json'.format(coco_dir, self.dataset)
        self.coco = COCO(path)
        self.tables = {
            'categories': self.coco.loadCats(self.coco.getCatIds()),
            'images': self.coco.loadImgs(self.coco.getImgIds()),
            'annotations': self.coco.loadAnns(self.coco.getAnnIds())
        }
        for annotation in self.tables['annotations']:
            annotation['seg_dims'] = np.array(
                [len(seg) for seg in annotation['segmentation']]
            ).tostring()
            annotation['segmentation'] = np.array(
                [pt for pts in annotation['segmentation'] for pt in pts]
            )
            annotation['bbox'] = np.array(annotation['bbox']).tostring()
        if self.data_dir:
            for img in self.tables['images']:
                if self.check_location(img['file_name']):
                    img['local_path'] = '{}{}'.format(data_dir, image_name)


    def create_tables(self, file):
        print('Running {}...'.format(file))
        with open(file,'r') as f:
            query = f.read()
        if not self.connxn:
            print('Connection to database must first be established.')
            return 0
        print('Executing SQL...')
        self.cursor.execute(query)
        print('\tDone.')
        self.connxn.commit()
        print('\tCommited.')

    def insert_into_table(self, table_name, dict_lst, pages=100):
        print('Loading {} into database...'.format(table_name))
        fields = [field for field in dict_lst[0]]
        query = sql.SQL('''
        INSERT INTO {} ({})
        VALUES ({})
        ''').format(
                sql.Identifier(table_name),
                sql.SQL(',').join(map(sql.Identifier, fields)),
                sql.SQL(',').join(map(sql.Placeholder, fields)),
            )
        execute_batch(self.cursor, query, dict_lst, page_size=pages)
        print('\t{} Done.'.format(table_name))
        self.connxn.commit()
        print('\tCommited.')

class QueryDatabase(DataPipeline):
    def __init__(self, dataset, user, host, data_dir=None):
        assert data_dir, \
            'No data directory was specified.\n' + \
            'Please specify the directory where images are stored.'

        assert os.path.exists(data_dir), \
            'This directory does not exist.\n' + \
            'Please check the path and try again.'

        super().__init__(dataset, user, host, db_prefix)
        self.data_dir = data_dir
        self.cursor.close()
        self.cursor = self.connxn.cursor(cursor_factory=RealDictCursor)
        self.df_query = None
        self.path = '{}data/coco/{}/'.format(self.data_dir, self.dataset)

    def query_database(self, query=None):
        if not query:
            query = input('Type out query:\n# ')
            while ';' not in query:
                query += ' ' + input('# ')

        self.cursor.execute(query)
        self.df_query = pd.DataFrame(self.cursor.fetchall())
        self.query = query

    def download(self, image_id, image_name, image_url):
        img_data = requests.get(image_url).content
        # path = '{}data/coco/{}/'.format(self.data_dir, self.dataset)
        path = self.path
        if not os.path.exists(path):
            os.makedirs(path)
        path += image_name
        with open(path, 'wb') as file:
            file.write(img_data)
        self.update_sql('images', image_id, 'local_path', path)

    def update_sql(self, table, id, field, value):
        query = sql.SQL('''
            UPDATE {} SET {} = {} WHERE id = {};
        ''').format(
                sql.Identifier(table),
                sql.Identifier(field),
                sql.Literal(value),
                sql.Literal(id)
             )
        self.cursor.execute(query)
        self.connxn.commit()

    def get_images(self):
        field_req = ['image_id', 'file_name', 'local_path']
        url_req = ['coco_url', 'flickr_url']
        assert all(col in self.df_query.columns for col in field_req), \
            'Data frame must include {}.'.format(', '.join(field_req))

        assert any(col in self.df_query.columns for col in url_req), \
            'Data frame must include at least {} or {}'.format(*url_req)


        print('Checking images...')
        for i,image in tqdm(self.df_query.iterrows()):
            if not image['local_path']:
                if self.check_location(image['file_name']):
                    path = self.path + image['image_name']
                    self.update_sql('images', image.image_id, 'local_path', path)
                    continue
                print('Image not yet downloaded. Downloading a copy.')
                self.download(*image[['image_id', 'file_name', 'coco_url']])
            elif not self.check_location(image['file_name']):
                print("Couldn't find file. Downloading a copy.")
                self.download(*image[['image_id', 'file_name', 'coco_url']])

        self.query_database(self.query)

        return list(self.df_query.local_path)

    def bytestr_to_list(
        self, bytestr, dims=b'\x04\x00\x00\x00\x00\x00\x00\x00', dtype=float
    ):
        it = iter(np.frombuffer(bytestr, dtype=dtype))
        return [[next(it) for _ in range(shape)]
                 for shape in np.frombuffer(dims, dtype=int)]

''' Functions '''
def load_sql(data_dir, dataset, dbname, user='postgres', host='/tmp'):
    '''
    Parameters =====================================================
        data_dir (str)      : path to data directory
        dataset (str)       : specific dataset to load
        dbname (str)        : name of database to connect with
        user (str)          : user name for database (default: "postgres")
    '''
    pass



if __name__ == "__main__":
    coco_dir = '../data/coco'
    dataset = 'train2017'
    store_dir = '/media/mosqueteiro/TOSHIBA EXT/detecting_trafficlights/'
    user = 'mosqueteiro'
    host = '/var/run/postgresql'

    with BuildDatabase(
        dataset='val2016', user='postgres', host='pg_serv')

#     query = '''
# SELECT id as image_id, file_name, coco_url, local_path
# FROM images
# WHERE id IN (309022, 5802, 118113, 483108, 60623)
# LIMIT 5;
#     '''
#
#     with QueryDatabase(
#         dataset=dataset, user=user, host=host, data_dir=store_dir
#     ) as train2017:
#         train2017.query_database(query)
#         print(train2017.df_query)
#         # train2017.df_query.drop([3,4], axis=0, inplace=True)
#         images = train2017.get_images()
#         print(images)
#         print(train2017.df_query)
