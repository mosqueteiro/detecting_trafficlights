'''
File:   data_pipeline.py
Author: Nathan James
Date:   03/15/19

Pipeline for filtering and downloading dataset from cocodataset.org
'''

import psycopg2 as pg2
from psycopg2 import connect, sql
from psycopg2.extras import execute_batch, RealDictCursor
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import requests
import os
from io import StringIO, BytesIO


''' Data pipeline class '''
class DataPipeline(object):
    def __init__(self, dataset, host, user='postgres', port='5432', data_dir=None):
        '''
    Description: Base class for connecting to SQL server
    Parameters =====================================================
        dataset (str)      : name of dataset
        host (str)         : PostgreSQL host (container name for docker)
        user (str)         : postgres user, default: postsgres
        port (str)         : port the SQL server has open
        data_dir (str)     : path to directory containing all the data, this
                             should have a '/' at the end of it
        '''
        self.dataset = dataset
        self.data_dir = data_dir
        if data_dir:
            assert os.path.exists(data_dir), \
                'This directory does not exist.\n' + \
                'Please check the path and try again.'
            self.data_dir = '{}{}/'.format(data_dir, self.dataset)
        self.connect_sql(dbname=dataset, user=user, host=host, port=port)

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
        path = '{}{}'.format(self.data_dir, image_name)
        return os.path.exists(path)


class BuildDatabase(DataPipeline):
    def __init__(self, dataset, host, user='postgres', port='5432', data_dir=None):
        '''
    Description: class for building a SQL database from a coco annotations
            instances_*.json
    Parameters =====================================================
        dataset (str)      : name of dataset
        host (str)         : PostgreSQL host (container name for docker)
        user (str)         : postgres user, default: postsgres
        port (str)         : port the SQL server has open
        data_dir (str)     : path to directory containing all the data, this
                             should have a '/' at the end of it
        '''
        super().__init__(dataset, host=host, user=user, data_dir=data_dir)
        self.tables = None

    def build_sql(self):
        print('Building Database...')

        # print('Setting up SQL tables')
        self.create_tables('coco_dataset.sql')

        self.load_json(None)

        for table,entries in self.tables.items():
            self.copy_into_table(table, entries)


        # print('Adding relational constraints')
        # self.create_tables('sql_constraints.sql')

        print('Finished building database.')

    def load_json(self, coco_dir=None):
        print('Loading COCO dataset {} information....'.format(self.dataset))
        if not coco_dir:
            path = '{}annotations/instances_{}.json'.format(self.data_dir,
                                                             self.dataset)
        else:
            path = coco_dir
        assert os.path.exists(path), \
            'File does not exists at this location.\n{}'.format(path)
        with open(path, 'r') as file:
            self.tables = json.load(file)
        self.info = self.tables.pop('info')
        for table in self.tables:
            self.tables[table] = pd.DataFrame(self.tables[table]).set_index('id')
        annotations = self.tables['annotations']
        annotations.bbox = annotations.bbox.apply(
                lambda x: np.array(x, dtype=float)
        )
        annotations.segmentation = annotations.segmentation.apply(json.dumps)
        if self.data_dir:
            images = self.tables['images']
            located = images.file_name.apply(self.check_location)
            images['local_path'] = images.loc[located, 'file_name'].apply(
                        lambda x: self.data_dir + x
            )

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

    def copy_into_table(self, table_name, df):
        print('Loading {} into database...'.format(table_name))
        # buffer = StringIO()
        with StringIO() as buffer:
            df.to_csv(buffer, sep='\t')
            buffer.seek(0)
            fields = buffer.readline().strip('\n').split()
            self.cursor.copy_from(
                buffer, table_name, sep='\t', null='', columns=fields
            )
        print('\t{} COPIED.'.format(table_name))
        self.connxn.commit()
        print('\tCOMMITED.')


class QueryDatabase(DataPipeline):
    def __init__(self, dataset, host, user='postgres', port='5432', data_dir=None):
        '''
    Description: Class for connecting and querying SQL server holding annotation
            data for a COCO dataset
    Parameters =====================================================
        dataset (str)      : name of dataset
        host (str)         : PostgreSQL host (container name for docker)
        user (str)         : postgres user, default: postsgres
        port (str)         : port the SQL server has open
        data_dir (str)     : path to directory containing all the data, this
                             should have a '/' at the end of it
        '''
        assert data_dir, \
            'No data directory was specified.\n' + \
            'Please specify the directory where images are stored.'
        super().__init__(dataset, host=host, user=user, data_dir=data_dir)
        self.cursor.close()
        self.cursor = self.connxn.cursor(cursor_factory=RealDictCursor)
        self.df_query = None

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
        path = self.data_dir
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
        field_req = ('image_id', 'file_name', 'local_path')
        url_req = ('coco_url', 'flickr_url')
        assert all(col in self.df_query.columns for col in field_req), \
            'Data frame must include {}.'.format(', '.join(field_req))

        assert any(col in self.df_query.columns for col in url_req), \
            'Data frame must include at least {} or {}'.format(*url_req)


        print('Checking images...')
        for i,image in self.df_query.iterrows():
            if not image['local_path']:
                if self.check_location(image['file_name']):
                    path = self.data_dir + image['image_name']
                    self.update_sql('images', image.image_id, 'local_path', path)
                    continue
                print('Image not yet downloaded. Downloading a copy.')
                self.download(*image[['image_id', 'file_name', 'coco_url']])
            elif not self.check_location(image['file_name']):
                print("Couldn't find file. Downloading a copy.")
                self.download(*image[['image_id', 'file_name', 'coco_url']])

        self.query_database(self.query)

        return list(self.df_query.local_path)


''' Functions '''
def make_csv(file):
    '''
    Parameters =====================================================
        file (str)      : path to json file containing annotations
    '''
    print('Not currently implemented.'); pass
    print('Opening file: {}'.format(file))
    with open(file, 'r') as raw:
        tables = json.load(raw)
    print('Creating Tables')
    for table in tables:
        df = pd.DataFrame(tables[table])


###################################################
# special code to use PostgreSQL BYTEA and np arrays
def _adapt_array(arr):
    out = BytesIO()
    np.savetxt(out, arr, fmt='%.2f')
    out.seek(0)
    return pg2.Binary(out.read())

def _typecast_array(value, cur):
    if value is None:
        return None
    data = pg2.BINARY(value, cur)
    bdata = BytesIO(data[1:-1])
    bdata.seek(0)
    return np.loadtxt(bdata)

pg2.extensions.register_adapter(np.ndarray, _adapt_array)
t_array = pg2.extensions.new_type(pg2.BINARY.values, 'numpy', _typecast_array)
pg2.extensions.register_type(t_array)
###################################################

if __name__ == "__main__":
    coco_dir = '../data/coco/'
    dataset = 'train2017'
    user = 'postgres'
    host = 'pg_serv'

    with BuildDatabase(
        dataset='val2017', user=user, host=host, data_dir=coco_dir
    ) as buildDB:
        buildDB.build_sql()

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
