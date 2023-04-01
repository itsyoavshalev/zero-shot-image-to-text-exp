import torch
from tqdm import tqdm
from COCO_GB_V1_dataset import COCO_GB_V1_dataset
from datasets import load_dataset
import os
import json
from gender_utils import identify_gender_words

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluate(device='cuda'):
    num_samples = 800
    w_corrects = 0
    m_corrects = 0
    n_corrects = 0
    w_total = 0
    m_total = 0
    n_total = 0

    with open('/mnt/sb/zero-shot-image-to-text-exp/results_2002-222914_coco_zero/results.json', 'r') as f:
        predictions = json.load(f)[:num_samples]
    
    karpathy_path = '/mnt/sb/datasets/COCO/dataset_coco_karpathy.json'
    with open(karpathy_path, 'r') as f:
        karpathy_data = json.load(f)

    test_data = [x for x in karpathy_data['images'] if x['split'] == 'test']

    print(len(predictions))
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_data)):
            pred = [x for x in predictions if x['id']==batch_data['filename']]
            if pred != []:
                pred_sent = pred[0]['best_clip_res'].strip().lower().capitalize()
                pred_tag = identify_gender_words(pred_sent)
                gt_tags = []
                for gt_sent in batch_data['sentences']:
                    gt_raw_sent = gt_sent['raw'].strip().lower().capitalize()
                    tmp_gt_tag = identify_gender_words(gt_raw_sent)
                    gt_tags.append(tmp_gt_tag)

                if 0 in gt_tags and 1 not in gt_tags:
                    gt_tag = 0
                elif 1 in gt_tags and 0 not in gt_tags:
                    gt_tag = 1
                else:
                    gt_tag = 2

                if gt_tag == 0:
                    w_total += 1
                    if pred_tag == gt_tag:
                        w_corrects += 1
                elif gt_tag == 1:
                    m_total += 1
                    if pred_tag == gt_tag:
                        m_corrects += 1
                else: 
                    n_total += 1
                    if pred_tag == gt_tag:
                        n_corrects += 1
            
    m_acc = m_corrects / m_total
    w_acc = w_corrects / w_total
    n_acc = n_corrects / n_total
    total_samples = m_total + w_total + n_total
    print(total_samples)
    print(f'woman: {w_acc}, man: {m_acc}, neutral: {n_acc}')

if __name__ == '__main__':
    # dataset = COCO_GB_V1_dataset(dataset_path='/mnt/sb/datasets/COCO/', split='test', file_path='COCOGB_V1/Ksplit_gender_category.json')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12)
    
    evaluate()