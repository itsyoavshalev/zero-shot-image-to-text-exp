import torch
from tqdm import tqdm
from COCO_GB_V1_dataset import COCO_GB_V1_dataset
from datasets import load_dataset
import os
import json


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluate(dataloader, device='cuda']):
    w_corrects = 0
    m_corrects = 0
    w_total = 0
    m_total = 0
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader)):
            current_correct = int(probs[0] > probs[1])
            if batch_data['orig_gender'][0] == 0:
                w_total += 1
                w_corrects += current_correct
            elif batch_data['orig_gender'][0] == 1:
                m_total += 1
                m_corrects += current_correct
            else: 
                raise('invalid gender')
            
    m_acc = m_corrects / m_total
    w_acc = w_corrects / w_total
    print(f'woman: {w_acc}, man: {m_acc}')

if __name__ == '__main__':
    dataset = COCO_GB_V1_dataset(None, dataset_path='/mnt/sb/datasets/COCO/', split='test', file_path='COCOGB_V1/Ksplit_gender_category.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers=12)
    
    evaluate(dataloader)