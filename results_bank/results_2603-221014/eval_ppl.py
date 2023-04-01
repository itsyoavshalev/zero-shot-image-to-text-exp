from tqdm import tqdm
import argparse
import torch
import sys
import numpy as np 
from transformers import BertTokenizer,BertForMaskedLM
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, default='/mnt/sb/zero-shot-image-to-text-exp/results_2002-222914_coco_zero/results.json') # 518.346154616991, 51.229835882731415 (image of a)
    # parser.add_argument("--prediction_path", type=str, default='/mnt/sb/zero-shot-image-to-text-exp/results_1502-230824_coco_ours/results.json') # 835.64989393985, 57.9534711420098
    # parser.add_argument("--prediction_path", type=str, default='/mnt/sb/zero-shot-image-to-text-exp/results_2503-090350/results.json') # 617.1615894011353, 53.57757333423984
    # parser.add_argument("--prediction_path", type=str, default='/mnt/sb/zero-shot-image-to-text-exp/results_2503-174221/results.json') # 74.334
    # parser.add_argument("--prediction_path", type=str, default='/mnt/sb/zero-shot-image-to-text-exp/results_2503-212601/results.json') # 19.5

    # parser.add_argument("--prediction_path", type=str, default='/mnt/sb/zero-shot-image-to-text-exp/results_2702-192814_flickr_zero/results.json')
    return parser

if __name__ == '__main__':
    num_cands = 223

    cli_args = get_parser().parse_args()

    with open(cli_args.prediction_path, 'r') as f:
        predictions = json.load(f)
        predictions = predictions[:num_cands]

    print(len(predictions))

    device = "cpu"
    model_id = "bert-large-cased"
    
    model = BertForMaskedLM.from_pretrained(model_id).to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(model_id)
    # PAIRS:
    # Image of a - 67.53804610182385
    # A - 121.61028312818746
    # none - 587.914284249568
    ppls = []
    for tmp_prediction in tqdm(predictions):
        prediction = 'Image of a ' + tmp_prediction['best_clip_res'].strip().lower()
        # prediction = prediction[1:].strip().capitalize()
        pred_tokens = tokenizer.tokenize(prediction)
        pred_tokens = ["[CLS]"]+pred_tokens+["[SEP]"]
        pred_ids = torch.tensor([tokenizer.convert_tokens_to_ids(pred_tokens)]).to(device)
        labels = pred_ids.clone()
        with torch.no_grad():
            loss = model(pred_ids, labels=labels)[0]
        ppl = np.exp(loss.item())
        ppls.append(ppl)
    print(np.array(ppls).mean())
    print(cli_args.prediction_path)