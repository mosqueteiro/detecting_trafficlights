'''
File:   data_pipeline.py
Author: Nathan James
Date:   03/15/19

Pipeline for filtering and downloading dataset from cocodataset.org
'''

from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect

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


if __name__ == "__main__":
    data_dir = './data/coco'
    dataset = 'train2017'
    anno_file = '{}/annotations/instances_{}.json'.format(data_dir, dataset)

    coco=COCO(anno_file)

    categs = pd.DataFrame([cat for cat in coco.cats.values()])

    
