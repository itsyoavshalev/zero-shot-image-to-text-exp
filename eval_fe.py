from lavis.models import model_zoo
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import json
import numpy as np
from tqdm import tqdm

print(model_zoo)

num_cands = 1598

with open('/mnt/sb/zero-shot-image-to-text-exp/results_1502-230824/results.json', 'r') as f:
    candidates = json.load(f)
    candidates = candidates[:num_cands]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor", model_type="base", is_eval=True, device=device)

results = []
for item in tqdm(candidates):
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