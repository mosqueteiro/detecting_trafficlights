'''
File:   data_pipeline.py
Author: Nathan James
Date:   03/15/19

Pipeline for filtering and downloading dataset from cocodataset.org
'''

from psycopg2 import connect, sql
from datetime import datetime
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect

from tqdm import tqdm

''' Data pipeline class '''
class DataPipeline(object):
    pass


''' Functions '''

def show_image(coco, imgId, catIds=[], local=False, bbox=False):
    '''
    Parameters ===============================================
        coco (obj)          : COCO object from pycocotools
        imgId (int)         : image ID
        catIds (int, list)  : category ID(s) to pull annotations from
        local (bool)        : is image stored locally
        bbox (bool)         : show bboxes for category annotations
    '''
    img = coco.loadImgs(ids=imgId)[0]
    if local:
        img = io.imread(img['local_dir'])
    else:
        img = io.imread(img['coco_url'])
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    if bbox:
        bx_prop = {'fill': False,
                   'edgecolor': 'red',
                   'linewidth': 1.25}
        annotes = coco.getAnnIds(imgId, catIds=catIds)
        annotes = coco.loadAnns(annotes)
        boxes = [ann['bbox'] for ann in annotes]
        patches = [rect((x,y),w,h, **bx_prop) for x,y,w,h in boxes]
        for patch in patches:
            ax.add_patch(patch)

    fig.show()

def load_sql(data_dir, dataset, dbname, user='postgres', host='/tmp'):
    '''
    Parameters =====================================================
        data_dir (str)      : path to data directory
        dataset (str)       : specific dataset to load
        dbname (str)        : name of database to connect with
        user (str)          : user name for database (default: "postgres")
    '''
    conn = psycopg2.connect(dbname=dbname, user=user, host=host)
    curs = conn.cursor()
    json_path = '{}/annotations/instances_{}.json'.format(data_dir, dataset)
    coco = COCO(json_path)

    def insert_into_table(table_name, dict_lst, pages=100):
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

    imgs = coco.getImgIds()
    images = coco.loadImgs(imgs)
    insert_into_table('images', images, 1000)
    conn.commit()

    categories = coco.loadCats(coco.getCatIds())
    insert_into_table('categories', categories)
    conn.commit()
    
    annotations = coco.loadAnns(coco.getAnnIds())
    insert_into_table('annotations', annotations)
    conn.commit()


if __name__ == "__main__":
    data_dir = './data/coco'
    dataset = 'train2017'
    # anno_file = '{}/annotations/instances_{}.json'.format(data_dir, dataset)
    #
    # coco=COCO(anno_file)
    #
    # categs = pd.DataFrame([cat for cat in coco.cats.values()])

    load_sql(data_dir, dataset, coco_trainval2017,
             'mosqueteiro', 'var/run/postgresql')
