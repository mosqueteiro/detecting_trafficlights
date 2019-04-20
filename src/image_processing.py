import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread, imshow, imshow_collection
import matplotlib.pyplot as plt

''' Classes '''
class ImageProcessor(object):
    def __init__(self, img_paths):
        self.paths = img_paths
        self.images = None
        self.loaded = False

    def load_imgs(self):
        # self.images = (imread(img) for img in self.paths)
        self.images = self.paths.apply(imread)
        self.loaded = True

    def resize_imgs(self, shape, **kwargs):
        opts = {
            'anti_aliasing': True,
            'mode': 'constant',
        }
        if not self.loaded:
            self.load_imgs()
        # self.images = (resize(img, **kwargs) for img in self.images)
        self.images = self.images.apply(lambda img: resize(img, shape, **opts))

    def to_gray(self):
        if not self.loaded:
            self.load_imgs
        # self.images = (rgb2gray(img) for img in self.images)
        self.images = self.images.apply(rgb2gray)


''' Functions '''
def show_images(images, resz_shape=None, bbox=None):
    '''
    Parameters ===============================================
        images (list)       : list of paths to images to display
        catIds (int, list)  : category ID(s) to pull annotations from
        bbox (list)        : list of bbox annotations for each picture
    '''
    imgs = [imread(img) for img in images]

    if resz_shape:
        imgs = [resize(img, resz_shape, anti_aliasing=True, mode='constant')
                for img in imgs]

    if bbox: # needs to be refactored
        bx_prop = {'fill': False,
                   'edgecolor': 'red',
                   'linewidth': 1.25}
        annotes = coco.getAnnIds(imgId, catIds=catIds)
        annotes = coco.loadAnns(annotes)
        boxes = [ann['bbox'] for ann in annotes]
        patches = [rect((x,y),w,h, **bx_prop) for x,y,w,h in boxes]
        for patch in patches:
            ax.add_patch(patch)

    fig = imshow_collection(imgs)
    fig.show()


if __name__ == "__main__":
    from trafficlight_data import load_binary_train

    X, y = load_binary_train()
    images = X['local_path']

    mask = y == 1

    n = 6
    imgs_1 = np.random.choice(images[mask], size=6, replace=False)
    imgs_0 = np.random.choice(images[~mask], size=6, replace=False)

    show_images(imgs_1)
    show_images(imgs_0)


    # imgs = [imread(img) for img in images]
    # # fig = imshow_collection(imgs, 'matplotlib')
    # # fig.show()
    # imgs_g = [rgb2gray(img) for img in imgs]
    # # fig = imshow_collection(imgs_g)
    # # fig.show()
    # # r_sz_op = {'anti_aliasing': True, 'mode':'constant',
    # #            'gridspec_kw':{'width_ratios':[[3,]]}}
    #
    # fig, axes = plt.subplots(2,5, figsize=(10,10))
    # for ax,img in zip([axs for sub in axes for axs in sub], imgs+imgs_g):
    #     imshow(img, ax=ax)
    # fig.show()
    #
    # imgs_sm = [
    #     resize(img,(100,100,3), anti_aliasing=True, mode='constant')
    #     for img in imgs
    # ]
    # imgs_sm_g = [rgb2gray(img) for img in imgs_sm]
    # fig, axes = plt.subplots(2,5, figsize=(10,10))
    # for ax,img in zip([axs for sub in axes for axs in sub], imgs_sm+imgs_sm_g):
    #     imshow(img, ax=ax)
    # fig.show()
