import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
from tqdm import tqdm
import numpy as np
import collections
import os
import json
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
import clip

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()

candidates_json = '/mnt/sb/zero-shot-image-to-text-exp/results_1502-230824/results.json'
print(candidates_json)

image_dir = '/mnt/sb/datasets/COCO/val2014/'
references_json = '/mnt/sb/datasets/COCO/dataset_coco_karpathy.json' 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, transform = clip.load('ViT-B/32', device=device, download_root='./clip_checkpoints', jit=False)
convert_models_to_fp32(model)

model.to(device)
model.eval()

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='Image of '):
        self.data = data

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(c_data).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images, transform),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    candidates = extract_all_captions(candidates, model, device)

    images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
    candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():
    with open(candidates_json) as f:
        tmp_candidates = json.load(f)
        
    if references_json:
        with open(references_json) as f:
            tmp_references = json.load(f)

    ref_dict = {}
    for item in tmp_references['images']:
        if item['filename'] not in ref_dict:
            ref_dict[item['filename']] = []
        t_s = item['sentences'][:5]
        assert len(t_s) == 5
        for s in t_s:
            ref_dict[item['filename']].append(s['raw'].strip())

    candidates = []
    image_paths = []
    image_ids = []
    references = []
    for c in tmp_candidates:
        file_suffix = c['id']
        image_paths.append(os.path.join(image_dir, file_suffix))
        image_ids.append(int(c['id'].split('_')[-1].split('.')[0]))
        candidates.append(c['best_clip_res'].strip().capitalize())

        if references_json:
            references.append(ref_dict[file_suffix])

    print(len(candidates))
    
    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    if references_json:
        # get text-text clipscore
        _, per_instance_text_text = get_refonlyclipscore(
            model, references, candidate_feats, device)
        
        # F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                  for image_id, clipscore, refclipscore in
                  zip(image_ids, per_instance_image_text, refclipscores)}

    else:
        scores = {image_id: {'CLIPScore': float(clipscore)}
                  for image_id, clipscore in
                  zip(image_ids, per_instance_image_text)}
        print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))

    if references_json:
        print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))
        print('RefCLIPScore: {:.4f}'.format(np.mean([s['RefCLIPScore'] for s in scores.values()])))


if __name__ == '__main__':
    main()