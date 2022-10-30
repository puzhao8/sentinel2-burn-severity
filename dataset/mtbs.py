import os
from glob import glob
from pathlib import Path
from typing import Any, List
from xmlrpc.client import boolean

import numpy as np
import tifffile as tiff 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


''' Data Augments '''
import torch
import torchvision.transforms as T

ALL_BANDS = {
    'ALOS': ['ND', 'VH', 'VV'],
    'S1': ['ND', 'VH', 'VV'],
    'S2': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
}   


def apply_transform(imgs: List, transforms: Any=None):
    '''
    Apply transforms to a list of images by stacking them along the channel dimension. 
    Then unstack when return.

    Parameter
    ----------
    * imgs: a list of images, torch tensor
    * transforms: torchvision transforms

    '''
    inputs = torch.cat(imgs, dim=0)
    channels_list = [im.shape[0] for im in imgs]
    idxs = [np.sum(np.array(channels_list[:i+1])) for i in range(0,len(channels_list))] # channel start index for each image
    idxs = [0] + idxs

    input_aug = transforms(inputs)
    outputs = [input_aug[idxs[i]:idxs[i+1]] for i in range(len(imgs))]
    return outputs


def get_band_index_dict(input_bands: dict[List[str]]):
    '''
    Get the input band index from all bands for satellite 'ALOS', 'S1' and 'S2'

    Parameters
    ------------
    * input_bands: specifies input bands for each satellite data
    '''

    def get_band_index(sat):
        list_all = ALL_BANDS[sat]
        list_input = list(input_bands[sat])
        return [list_all.index(band) for band in list_input]

    band_index_dict = {}
    for sat in ['ALOS', 'S1', 'S2']:
        band_index_dict[sat] = get_band_index(sat)
    
    return band_index_dict

def normalize_sar(img):
    return (np.clip(img, -30, 0) + 30) / 30

class MTBS(BaseDataset):
    """ MTBS Dataset. Read images, apply transform and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        transform (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['unburn', 'low', 'moderate', 'high'] # 'greener', 'cloud'
    
    def __init__(
            self, 
            filenames: List[str],
            data_dir:Path, 
            satellites: List[str]=['S2'],
            input_bands: dict[list[str]]=None,
            prepost: List[str]=['post'],
            stacking: boolean=True,
            mask_src: str='mtbs',
            classes: List[str]=None, 
            transform=None, 
            transform_loc=None,
            **kwargs
    ):
        '''
        Parameter
        ----------
        * filenames: a list of filename of tif image
        * data_dir: the absolute dir for train/val/test data 
        * satellites: list of satellite data for training
        * input_bands: specifies input bands for each satellite data
        * prepost: a subset of ['pre', 'post'], which controls the use of pre images
        * stacking: if stack pre post images channel wise
        * mask_src: the source of mask, e.g, 'mtbs'
        * classes: list of burn severity
        * transform: non-location related transforms
        * transform_loc: location related transforms
        '''
        self.filenames = filenames
        self.data_dir = data_dir
        self.satellites = satellites
        self.use_pre = 'pre' in prepost
        self.stacking = stacking
        self.mask_dir = data_dir / 'mask' / mask_src
        self.is_mtbs = mask_src == 'mtbs'

        self.band_index_dict = get_band_index_dict(input_bands)
        print(self.band_index_dict)
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(clss.lower()) for clss in classes]
        self.transform = transform
        self.transform_loc = transform_loc
    
    
    def __getitem__(self, i):
        
        ''' read data '''
        image_list = []
        # for sat in sorted(self.fps_dict.keys()):
        for sat in self.satellites: # modified on Jan-9

            # read post image from sat
            post_fp = self.data_dir / sat / 'post' / self.filenames[i]
            image_post = tiff.imread(post_fp) # C*H*W
            image_post = np.nan_to_num(image_post, 0)
            if sat in ['S1','ALOS']: image_post = self.normalize_sar(image_post)
            image_post = image_post[self.band_index_dict[sat],] # select bands

            # read pre image from sat
            if self.use_pre:
                pre_fp = self.data_dir / sat / 'pre' / self.filenames[i]
                image_pre = tiff.imread(pre_fp)
                image_pre = np.nan_to_num(image_pre, 0)
                if sat in ['S1','ALOS']: image_pre = normalize_sar(image_pre)
                image_pre = image_pre[self.band_index_dict[sat],] # select bands
                
                if self.stacking: # if stacking bi-temporal data
                    stacked = np.concatenate((image_pre, image_post), axis=0) 
                    image_list.append(stacked) #[x1, x2]
                else:
                    image_list += [image_pre, image_post] #[t1, t2]
            else:
                image_list.append(image_post) #[x1_t2, x2_t2]

        ''' read mask '''
        mask = tiff.imread(self.mask_dir / self.filenames[i])

        if not self.is_mtbs:
            mask = (mask > 0).astype('float32')

        if self.is_mtbs:
            mask[mask==0] = 1 # both 0 and 1 are unburned
            mask[mask>=5] = 0 # ignore 'greener' (5), 'cloud' (6)
            mask = mask - 1 # 4 classes in total: 0, 1, 2, 3
        else:
            masks = [(mask == v) for v in self.class_values] # 1~6
            mask = np.stack(masks, axis=0).astype('float32')

        image_list.append(mask[np.newaxis,])
        imgs = [torch.from_numpy(img) for img in image_list]

        if self.transform_loc is not None: # location related transform
            imgs = apply_transform(imgs, self.transforms_loc)  #TODO: apply some other transforms for only input images 

        if self.transform is not None:
            imgs[:-1] = apply_transform(imgs[:-1], self.transform)

        return (tuple(imgs[:-1]), imgs[-1]) # x, y
        
        
    def __len__(self):
        return len(self.filenames)

    @classmethod
    def train_test_val_split(cls, data_dir: str, test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 0) -> List[str]:
        '''
        Split the data paths with given test_ratio and val_ratio (to all images)
        Return filenames

        Parameters
        -----------
        * data_dir: the folder containing all samples
        * test_ratio: 0-1, default 0.2
        * val_ratio: 0-1, default 0.1
        * seed: default 0, random train and val data
        '''
        # as pre and post images are one-to-one match, here we use pre paths for spliting
        # and then replace 'pre' by 'post' we get all the train/val/test post iamges
        
        all_images = sorted(glob(f"{data_dir}/S2/pre/*.tif"))  # sorted for reproducibility
        all_images = np.array([os.path.basename(path) for path in all_images])
        n = len(all_images)
        train_paths, val_paths, test_paths = [], [], []
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        
        # get test image paths
        np.random.seed(100)
        indices = np.random.permutation(n)
        test_paths.append(all_images[indices[:n_test]])
        train_val_paths = all_images[indices[n_test:]]
        
        # get train & val image paths
        np.random.seed(seed)
        indices = np.random.permutation(train_val_paths.size)
        val_paths.append(train_val_paths[indices[:n_val]])
        train_paths.append(train_val_paths[indices[n_val:]])
        return np.concatenate(train_paths), np.concatenate(val_paths), np.concatenate(test_paths)



