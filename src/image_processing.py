import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imshow, imshow_collection


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

    imshow_collection(images)
