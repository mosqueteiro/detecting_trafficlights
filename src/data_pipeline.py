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
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import requests
import os

from tqdm import tqdm

''' Data pipeline class '''
class DataPipeline(object):
    def __init__(self, dataset, user, host, db_prefix='coco_', data_dir=None):
        self.dataset = dataset
        self.data_dir = data_dir
        dbname = db_prefix + dataset
        self.connect_sql(dbname=dbname, user=user, host=host)

    def __del__(self):
        self.cursor.close()
        # print('Cursor closed.')
        self.connxn.close()
        # print('Connection closed.')
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def connect_sql(self, dbname, user='postgres', host='/tmp'):
        print('Connecting to PostgreSQL server....')
        self.connxn = connect(dbname=dbname, user=user, host=host)
        print('\tConnected.')
        self.cursor = self.connxn.cursor()


class BuildDatabase(DataPipeline):
    def __init__(
        self, dataset, user, host, db_prefix='coco_', data_dir=None
    ):
        super().__init__(dataset, user, host, db_prefix, data_dir)
        self.coco = None
        self.tables = None

    def build_sql(self, coco_dir):
        print('Building Database...')

        print('Setting up SQL tables')
        self.create_tables()

        self.load_json(coco_dir)

        for table,entries in self.tables.items():
            self.insert_into_table(table, entries, pages=1000)


        print('Adding relational constraints')
        self.create_tables('sql_constraints.sql')

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
            annotation['segmentation'] = str(annotation['segmentation'])


    def create_tables(self, file='coco_dataset.sql'):
        print('Running {}...'.format(file))
        with open(file,'r') as f:
            query = f.read()
        if self.connxn is None:
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
    def __init__(self, dataset, user, host, db_prefix='coco_', data_dir=None):
        super().__init__(dataset, user, host, db_prefix, data_dir)
        self.cursor.close()
        self.cursor = self.connxn.cursor(cursor_factory=RealDictCursor)
        self.df_query = None

        assert data_dir, \
            'No data directory was specified.\n' + \
            'Please specify the directory where images are stored.'

        assert os.path.exists(data_dir), \
            'This directory does not exist.\n' + \
            'Please check the path and try again.'

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
        path = '{}data/coco/{}/'.format(self.data_dir, self.dataset)
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
                print('Image not yet downloaded. Downloading a copy.')
                self.download(*image[['image_id', 'file_name', 'coco_url']])
            elif not self.check_location(image['file_name']):
                print("Couldn't find file. Downloading a copy.")
                self.download(*image[['image_id', 'file_name', 'coco_url']])

        self.query_database(self.query)

        return list(self.df_query.local_path)

    def check_location(self, image_name):
        path = '{}data/coco/{}/{}'.format(self.data_dir, self.dataset, image_name)
        return os.path.exists(path)

''' Functions '''
def load_sql(data_dir, dataset, dbname, user='postgres', host='/tmp'):
    '''
    Parameters =====================================================
        data_dir (str)      : path to data directory
        dataset (str)       : specific dataset to load
        dbname (str)        : name of database to connect with
        user (str)          : user name for database (default: "postgres")
    '''
    print('Connecting to PostgreSQL server....', end='')
    conn = connect(dbname=dbname, user=user, host=host)
    print('Connected')
    curs = conn.cursor()

    print('Loading COCO dataset {} information:'.format(dataset))
    json_path = '{}/annotations/instances_{}.json'.format(data_dir, dataset)
    coco = COCO(json_path)

    print('Loading image table into SQL.......', end='')
    imgs = coco.getImgIds()
    images = coco.loadImgs(imgs)
    insert_into_table(curs, 'images', images, 1000)
    conn.commit()
    print('Done', end='\n\n')

    print('Loading categories table into SQL......', end='')
    categories = coco.loadCats(coco.getCatIds())
    insert_into_table(curs, 'categories', categories)
    conn.commit()
    print('Done', end='\n\n')

    print('Loading annotations table in SQL......', end='')
    annotations = coco.loadAnns(coco.getAnnIds())
    for annotation in tqdm(annotations):
        annotation['bbox'] = str(annotation['bbox'])
        annotation['segmentation'] = str(annotation['segmentation'])
    insert_into_table(curs, 'annotations', annotations)
    conn.commit()
    print('Done', end='\n\n')

    print('Closing cursor.')
    curs.close()
    print('Closing connection.')
    conn.close()
    print('exit')

def insert_into_table(curs, table_name, dict_lst, pages=100):
    fields = [field for field in dict_lst[0]]
    query = sql.SQL('''
    INSERT INTO {} ({})
    VALUES ({})
    ON CONFLICT (id)
    DO UPDATE
        SET {}
    ''').format(
        sql.Identifier(table_name),
        sql.SQL(',').join(map(sql.Identifier, fields)),
        sql.SQL(',').join(map(sql.Placeholder, fields)),
        sql.SQL(',').join([
            sql.SQL('{0}=EXCLUDED.{0}').format(s)
            for s in map(sql.Identifier, fields)
        ])
    )
    execute_batch(curs, query, dict_lst, page_size=pages)

if __name__ == "__main__":
    coco_dir = '../data/coco'
    dataset = 'train2017'
    store_dir = '/media/mosqueteiro/TOSHIBA EXT/detecting_trafficlights/'
    user = 'mosqueteiro'
    host = '/var/run/postgresql'

    query = '''
SELECT id as image_id, file_name, coco_url, local_path
FROM images
WHERE id IN (309022, 5802, 118113, 483108, 60623)
LIMIT 5;
    '''

    with QueryDatabase(
        dataset=dataset, user=user, host=host, data_dir=store_dir
    ) as train2017:
        train2017.query_database(query)
        print(train2017.df_query)
        # train2017.df_query.drop([3,4], axis=0, inplace=True)
        images = train2017.get_images()
        print(images)
        print(train2017.df_query)
