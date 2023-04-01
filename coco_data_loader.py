import torch
import os
import cv2
from PIL import Image
import json
import numpy as np
from torchvision import transforms


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, images_base_path, clip_preprocess, start_index=0, count=0, filter_json_path=None):
        if filter_json_path is not None: # coco karpathy val split
            with open(filter_json_path, 'r') as filter_json_file:
                filter_json_data = json.load(filter_json_file)
            self.images = sorted(['COCO_val2014_' + '{:012d}'.format(x["cocoid"]) + '.jpg' for x in filter_json_data['images'] if x['split'] == 'test'])
            assert len(self.images) == 5000
            self.images = self.images[start_index:(start_index+count)]
            assert len(self.images) == count
        else:
            self.images = [x for x in sorted(os.listdir(images_base_path)) if x.endswith(".jpg")]
        self.clip_preprocess = clip_preprocess
        self.clip_preprocess_debug = transforms.Compose(self.clip_preprocess.transforms[:-1])
        self.images_base_path = images_base_path
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = self.images_base_path + image_name
        
        image = Image.open(image_path)
        images_debug = torch.cat([self.clip_preprocess_debug(image).unsqueeze(0)])
        images = torch.cat([self.clip_preprocess(image).unsqueeze(0)])
        ids = [image_name]
        data_dict = {'images':images, 'ids':ids, 'images_debug':images_debug}

        return data_dict
