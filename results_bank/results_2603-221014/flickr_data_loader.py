from torch.utils.data import Dataset
import CLIP.clip as clip
from PIL import Image
from pycocotools.coco import COCO
import json
import torch

class FlickrDataset(Dataset):
    def __init__(self, root_dir, transform, split='test'):
        self.root_dir = root_dir
        self.images_path = f'{self.root_dir}flickr30k_images/flickr30k_images/'
        ann_path = f'{self.root_dir}/karpathy_dataset_flickr30k.json'
        with open(ann_path, 'r') as f:
            data = json.load(f)['images']
        data = [x for x in data if x['split']==split]
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        
        img_name = ann['filename']
        img_path = f'{self.images_path}{img_name}'
        
        images = torch.cat([self.transform(Image.open(img_path)).unsqueeze(0)])

        ids = [ann['imgid']]
        data = {'images':images, 'ids':ids}

        return data