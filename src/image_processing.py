import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imshow, imshow_collection
import matplotlib.pyplot as plt

''' Classes '''
class ImageProcessor(object):
    def __init__(self):
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
        img = imread(img['local_dir'])
    else:
        img = imread(img['coco_url'])
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
    from data_pipeline import QueryDatabase

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

    images = []
    with QueryDatabase(
        dataset=dataset, user=user, host=host, data_dir=store_dir
    ) as train2017:
        train2017.query_database(query)

        images = train2017.get_images()

    imgs = [imread(img) for img in images]
    # fig = imshow_collection(imgs, 'matplotlib')
    # fig.show()
    imgs_g = [rgb2gray(img) for img in imgs]
    # fig = imshow_collection(imgs_g)
    # fig.show()
    # r_sz_op = {'anti_aliasing': True, 'mode':'constant',
    #            'gridspec_kw':{'width_ratios':[[3,]]}}

    fig, axes = plt.subplots(2,5, figsize=(8,15))
    for ax,img in zip([axs for sub in axes for axs in sub], imgs+imgs_g):
        imshow(img, ax=ax)
    fig.show()

    imgs_sm = [
        resize(img,(100,100,3), anti_aliasing=True, mode='constant')
        for img in imgs
    ]
    imgs_sm_g = [rgb2gray(img) for img in imgs_sm]
    fig, axes = plt.subplots(2,5, figsize=(8,15))
    for ax,img in zip([axs for sub in axes for axs in sub], imgs_sm+imgs_sm_g):
        imshow(img, ax=ax)
    fig.show()
