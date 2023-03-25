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
from test_ret import generate_heatmap
from CLIP import clip as exp_clip

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()

is_flickr=False

if not is_flickr:
    candidates_json = '/mnt/sb/zero-shot-image-to-text-exp/results_2002-222914/results.json'
    image_dir = '/mnt/sb/datasets/COCO/val2014/'
    references_json = '/mnt/sb/datasets/COCO/dataset_coco_karpathy.json' 
else:
    candidates_json = '/mnt/sb/zero-shot-image-to-text-exp/results_2602-211100_flickr_ours/results.json'
    image_dir = '/mnt/sb/datasets/flickr30k_entities/flickr30k_images/flickr30k_images/'
    references_json = '/mnt/sb/datasets/flickr30k_entities/karpathy_dataset_flickr30k.json' 

print(candidates_json)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
use_masks = True

if not use_masks:
    model, transform = clip.load('ViT-B/32', device=device, download_root='./clip_checkpoints', jit=False)
else:
    model, transform = exp_clip.load("ViT-B/32", device=device, jit=False)
convert_models_to_fp32(model)

model.to(device)
model.eval()

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='Image of '):
        self.data = data

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(c_data[:330]).squeeze()
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
    # with torch.no_grad():
    for b in data:
        b = b['caption'].to(device)
        all_text_features.append(model.encode_text(b))
    all_text_features = torch.cat(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8, heatmaps=None):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images, transform),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    # with torch.no_grad():
    for i, b in enumerate(data):
        b = b['image'].to(device)
        if heatmaps is not None:
            b = b * (heatmaps[i:(i+1)].to(b.device))
        all_image_features.append(model.encode_image(b))
    all_image_features = torch.cat(all_image_features)
    return all_image_features


def get_refonlyclipscore(model, references, candidates, device):
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    with torch.no_grad():
        flattened_refs = extract_all_captions(flattened_refs, model, device)

    flattened_refs = flattened_refs / (flattened_refs**2).sum(axis=1, keepdims=True).sqrt()
    flattened_refs = flattened_refs.cpu().numpy()

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in enumerate(candidates):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():
    w=2.5
    num_cands = 1598
    
    with open(candidates_json) as f:
        tmp_candidates = json.load(f)
        tmp_candidates = tmp_candidates[:num_cands]
        
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
        if is_flickr:
            img_id = c['id']
            file_suffix = [x['filename'] for x in tmp_references['images'] if x['imgid']==img_id][0]
            image_ids.append(img_id)
        else:
            file_suffix = c['id']
            image_ids.append(int(c['id'].split('_')[-1].split('.')[0]))
        
        image_paths.append(os.path.join(image_dir, file_suffix))   
        candidates.append(c['best_clip_res'].strip().capitalize())

        if references_json:
            references.append(ref_dict[file_suffix])

    print(len(candidates))

    image_feats = []
    candidate_feats = []
    logits_per_text = []
    for y in tqdm(range(len(candidates))):
        tmp_image_feats = extract_all_images(image_paths[y:(y+1)], model, device, batch_size=64, num_workers=8)
        tmp_image_feats = tmp_image_feats / tmp_image_feats.norm(dim=-1, keepdim=True)
        
        tmp_candidate_feats = extract_all_captions(candidates[y:(y+1)], model, device)
        tmp_candidate_feats = tmp_candidate_feats / tmp_candidate_feats.norm(dim=-1, keepdim=True)
        candidate_feats.append(tmp_candidate_feats.detach().cpu())

        tmp_logits_per_text = (model.logit_scale.exp() * tmp_image_feats @ tmp_candidate_feats.t()).t()
        
        if use_masks:
            tmp_norm_image_relevance = generate_heatmap(tmp_logits_per_text, 1, model, device, True)
            heatmap = tmp_norm_image_relevance.detach().cpu()
            heatmap = heatmap.expand(heatmap.shape[0],3, *heatmap.shape[2:])
            tmp_image_feats = extract_all_images(image_paths[y:(y+1)], model, device, batch_size=64, num_workers=8, heatmaps=heatmap)
            tmp_image_feats = tmp_image_feats / tmp_image_feats.norm(dim=-1, keepdim=True)
            tmp_logits_per_text = (model.logit_scale.exp() * tmp_image_feats @ tmp_candidate_feats.t()).t()
        
        image_feats.append(tmp_image_feats.detach().cpu())
        logits_per_text.append(tmp_logits_per_text.detach().cpu())

    image_feats = torch.cat(image_feats)
    candidate_feats = torch.cat(candidate_feats)
    logits_per_text = torch.cat(logits_per_text)
    
    per_instance_image_text = w*np.clip(np.sum(image_feats.cpu().numpy() * candidate_feats.cpu().numpy(), axis=1), 0, None)
        
    # get text-text clipscore
    _, per_instance_text_text = get_refonlyclipscore(
        model, references, candidate_feats.cpu().numpy(), device)
    
    # F-score
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

    scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                for image_id, clipscore, refclipscore in
                zip(image_ids, per_instance_image_text, refclipscores)}
    
    print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))
    print('RefCLIPScore: {:.4f}'.format(np.mean([s['RefCLIPScore'] for s in scores.values()])))


if __name__ == '__main__':
    main()