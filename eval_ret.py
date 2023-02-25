from lavis.models import model_zoo
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import json
import numpy as np
from tqdm import tqdm

print(model_zoo)

num_cands = 1598

with open('/mnt/sb/zero-shot-image-to-text-exp/results_2002-222914/results.json', 'r') as f:
    candidates = json.load(f)
    candidates = candidates[:num_cands]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "base", device=device, is_eval=True)


results = []
for item in tqdm(candidates):
    raw_image = Image.open("/mnt/sb/datasets/COCO/val2014/" + item['id']).convert("RGB")
    caption = item['best_clip_res'].strip()
    # caption = "merlion in Singapore"
    # raw_image = Image.open("./merlion.png").convert("RGB")

    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)

    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    # print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
    # itc_score = model({"image": img, "text_input": txt}, match_head='itc')
    # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

    results.append(itm_scores[:, 1].item())    

    print(np.array(results).mean())