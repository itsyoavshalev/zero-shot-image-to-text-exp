from lavis.models import model_zoo
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import json
import numpy as np
from tqdm import tqdm

print(model_zoo)
num_cands = 298
# num_cands = 1598
is_flickr = False

if is_flickr:
    references_json = '/mnt/sb/datasets/flickr30k_entities/karpathy_dataset_flickr30k.json' 
    with open(references_json) as f:
        tmp_references = json.load(f)
    with open('/mnt/sb/zero-shot-image-to-text-exp/results_2702-192814_flickr_zero/results.json', 'r') as f:
        candidates = json.load(f)
        candidates = candidates[:num_cands]
else:
    with open('/mnt/sb/zero-shot-image-to-text-exp/results_1502-230824/results.json', 'r') as f:
        candidates = json.load(f)
        candidates = candidates[:num_cands]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=True, device=device)

results = []
for item in tqdm(candidates):
    if is_flickr:
        img_id = item['id']
        file_suffix = [x['filename'] for x in tmp_references['images'] if x['imgid']==img_id][0]
        raw_image = Image.open("/mnt/sb/datasets/flickr30k_entities/flickr30k_images/flickr30k_images/" + file_suffix).convert("RGB")
    else:
        raw_image = Image.open("/mnt/sb/datasets/COCO/val2014/" + item['id']).convert("RGB")
    caption = item['best_clip_res'].strip()

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)
    sample = {"image": image, "text_input": [text_input]}

    features_multimodal = model.extract_features(sample)
    features_image = model.extract_features(sample, mode="image")
    features_text = model.extract_features(sample, mode="text")
    similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
    results.append(similarity.item())    

print(np.array(results).mean())