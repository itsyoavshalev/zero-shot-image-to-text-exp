import json
import os
from torch.utils.data import Dataset


def load_COCOGB_V1(dataset_path, split, file_path):
    if split is not None:
        available_splits = ['train', 'val', 'restval', 'test']
        assert split in available_splits
        if split == 'train' or split == 'restval':
            splits = ['train', 'restval'] 
        else: 
            splits = [split]

    COCOGB_V1_path = f'{dataset_path}{file_path}'

    with open(COCOGB_V1_path, 'r') as f:
        COCOGB_V1 = json.load(f)
    
    if split is not None:
        COCOGB_V1_images = [x for x in COCOGB_V1['images'] if x['split'] in splits]
    else:
        COCOGB_V1_images = [x for x in COCOGB_V1['images']]

    for im_md in COCOGB_V1_images:
        img_split_folder = im_md['filepath']
        img_file_name = im_md['filename']
        img_path = f'{dataset_path}{img_split_folder}/{img_file_name}'
        assert os.path.isfile(img_path)
        im_md['local_path'] = img_path

    COCOGB_V1_images = sorted(COCOGB_V1_images, key=lambda x: x['local_path'], reverse=True)
    return COCOGB_V1_images


class COCO_GB_V1_dataset(Dataset):
    def __init__(self, dataset_path, split, file_path):
        self.dataset_path = dataset_path

        raw_db_records = load_COCOGB_V1(dataset_path, split, file_path)

        # 0 is woman
        # 1 is man
        # 2 is neutral
        # 3 no gender
        # raw_db_records = [x for x in raw_db_records if x['gender'] in [0, 1]]
        self.male_words = []
        

        # self.m_t_w_words_map = {'his':'her', 'man':'woman', 'men':'women', 'boy':'girl', 'boys':'girls'}
        # self.m_t_n_words_map = {'her':'his', 'woman':'man', 'women':'men', 'girl':'boy', 'girls':'boys'}
        # self.w_t_m_words_map = {'her':'his', 'woman':'man', 'women':'men', 'girl':'boy', 'girls':'boys'}
        # self.w_t_n_words_map = {'her':'his', 'woman':'man', 'women':'men', 'girl':'boy', 'girls':'boys'}
        # self.n_t_m_words_map = {'person':'man', 'people':'men', 'human':'man'}
        # self.n_t_w_words_map = {'person':'woman', 'people':'women', 'human':'woman'}

        self.db_records = []
        for raw_db_record in raw_db_records:
            current_gender = raw_db_record['gender']
            self.db_records.append({'id':raw_db_record['filename'], 'gender':current_gender})

    def __len__(self):
        return len(self.db_records)

    def __getitem__(self, idx):
        return self.db_records[idx]