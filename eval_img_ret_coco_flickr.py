import numpy as np
import torch
from CLIP import clip
from PIL import Image
from tqdm import tqdm
import json
import cv2
import os
from test_ret import generate_heatmap, show_image_relevance
import sys
from torchvision import transforms

int_mode = 'bilinear'

def apply_adaptive_thresholding(heatmaps, threshold_ratio=0):
    thresholds = torch.max(heatmaps.view(heatmaps.size(0), -1), dim=1, keepdim=True)[0] * threshold_ratio

    # Apply the threshold to the batch of heatmaps (in-place operation)
    heatmaps *= (heatmaps >= thresholds.view(-1, 1, 1, 1))

    return heatmaps


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad != None:
            p.grad.data = p.grad.data.float() 

def evaluate(similarity: np.ndarray):
    npt = similarity.shape[0]
    ranks = np.zeros(npt)

    for i in range(npt):
        inds = np.argsort(similarity[i])[::-1]
        ranks[i] = np.where(inds == i)[0][0]

    # recall
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    return (r1, r5, r10)

def run_test(model_path, mask_image, device):
    test_size = 5000
    chunck_size = 250
    dump_hm = True
    is_flickr=True
    
    if is_flickr:
        karpathy_path = '/mnt/sb/datasets/flickr30k_entities/karpathy_dataset_flickr30k.json'
    else:
        karpathy_path = '/mnt/sb/datasets/COCO/dataset_coco_karpathy.json'
    with open(karpathy_path, 'r') as f:
        karpathy_data = json.load(f)

    test_data = [x for x in karpathy_data['images'] if x['split'] == 'test'][:test_size]
    # assert len(test_data) == 5000
    
    if is_flickr:
        images_path = '/mnt/sb/datasets/flickr30k_entities/flickr30k_images/flickr30k_images/'
    else:
        images_path = '/mnt/sb/datasets/COCO/val2014/'
    clip_raw, transform = clip.load("ViT-B/32", device=device, jit=False)
    transform_raw = transforms.Compose(transform.transforms[:-1])
    # convert_models_to_fp32(clip_raw)
    clip_raw.to(device)
    clip_ft = clip_raw
    
    if model_path is not None:
        clip_ft, transform = clip.load("ViT-B/32", device=device, jit=False)
        checkpoint = torch.load(model_path, map_location=device)
        clip_ft.load_state_dict(checkpoint['model_state_dict'])
        clip_ft.zero_grad()
        checkpoint = None
        torch.cuda.empty_cache()
        # convert_models_to_fp32(clip_ft)
        clip_ft.to(device)
    
    logit_scale = clip_raw.logit_scale.exp()
    
    iterations = test_size // chunck_size
    print(iterations)
    all_text_sims = []
    for chunk_index_text in tqdm(range(iterations)):
        start_index_text = chunk_index_text * chunck_size
        end_index_text = start_index_text + chunck_size
        chunk_data_text = test_data[start_index_text:end_index_text]
        captions = []
        for x in chunk_data_text:
            i_sent = 0
            caption = x['sentences'][i_sent]['raw'].strip()
            captions.append(clip.tokenize(caption))

        assert len(captions) == chunck_size

        text_input = torch.zeros(len(captions), clip_raw.context_length, dtype=torch.long)
        for i, caption in enumerate(captions):
            text_input[i, :len(caption[0])] = caption[0]
    
        text_input = text_input.to(device)
        if mask_image:
            text_features = clip_ft.encode_text(text_input)#.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                text_features = clip_raw.encode_text(text_input)#.float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
        bs = text_features.shape[0]
        curr_text_sims = []
        for chunk_index_images in range(iterations):
            start_index_images = chunk_index_images * chunck_size
            end_index_images = start_index_images + chunck_size
            chunk_data_images = test_data[start_index_images:end_index_images]
            images = []
            raw_images = []

            for x in chunk_data_images:
                img_path = images_path + x['filename']
                image = Image.open(img_path).convert('RGB')
                images.append(transform(image))
                if dump_hm:
                    raw_images.append(transform_raw(image))
            
            assert len(images) == chunck_size
            
            images = torch.stack(images)
            images = images.to(device)
            if dump_hm:
                raw_images = torch.stack(raw_images)
                raw_images = raw_images.to(device)
            with torch.no_grad():
                raw_img_features = clip_raw.encode_image(images)#.float()
                raw_img_features = raw_img_features / raw_img_features.norm(dim=-1, keepdim=True)
            
            if mask_image:
                image_features = clip_ft.encode_image(images)#.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits_per_text = logit_scale * text_features @ image_features.T
            else:
                with torch.no_grad():
                    image_features = clip_raw.encode_image(images)#.float()
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits_per_text = logit_scale * text_features @ image_features.T
            
            if mask_image:
                cur_images_sims = []
                norm_image_relevance = []
                for y in range(bs):
                    tmp_norm_image_relevance = generate_heatmap(logits_per_text[y:(y+1)], images.shape[0], clip_ft, device, True)
                    with torch.no_grad():
                        # tmp_norm_image_relevance = apply_adaptive_thresholding(tmp_norm_image_relevance.detach())
                        norm_image_relevance.append(tmp_norm_image_relevance)
                    tmp_norm_image_relevance = None
                for y in range(bs):
                    with torch.no_grad():
                        s = norm_image_relevance[y].shape
                        if dump_hm:
                            hm0_=norm_image_relevance[y].expand(s[0],3,*s[2:])[y].permute(1,2,0).cpu().numpy()
                            hm1_=norm_image_relevance[y].expand(s[0],3,*s[2:])[y+1].permute(1,2,0).cpu().numpy()
                            cv2.imwrite('_hm0.jpg',(hm0_*255).astype(np.uint8))
                            cv2.imwrite('_hm1.jpg',(hm1_*255).astype(np.uint8))
                            cv2.imwrite('_im0.jpg',(raw_images[y].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                            cv2.imwrite('_im1.jpg',(raw_images[y+1].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                            cv2.imwrite('_im_cr0.jpg',(raw_images[y].permute(1,2,0).cpu().numpy()*255*hm0_).astype(np.uint8))
                            cv2.imwrite('_im_cr1.jpg',(raw_images[y+1].permute(1,2,0).cpu().numpy()*255*hm1_).astype(np.uint8))
                            print(chunk_data_text[y]['sentences'][0]['raw'])
                        
                        heatmap_features = clip_raw.encode_image(norm_image_relevance[y].expand(s[0],3,*s[2:])*images)#.float()
                        heatmap_features = heatmap_features / heatmap_features.norm(dim=-1, keepdim=True)
                        sim1 = logit_scale * text_features[y:(y+1)] @ heatmap_features.T
                        sim2 = logit_scale * text_features[y:(y+1)] @ raw_img_features.T
                        assert sim1.shape == sim2.shape
                        assert sim1.shape == (1, chunck_size)
                        sim = 0.5*sim1 + sim2
                        cur_images_sims.append(sim.detach().cpu())
                        
                        heatmap_features = None
                        sim1 = None
                        sim2 = None
                        sim = None
                cur_images_sims = torch.cat(cur_images_sims)
                        # sim = sim.t()
            else:
                cur_images_sims = logits_per_text.detach().cpu()
                logits_per_text = None
            
            cur_images_sims = cur_images_sims.squeeze()
            curr_text_sims.append(cur_images_sims)
            
            images = None
            image_features = None
            logits_per_text = None
            norm_image_relevance = None
            
        curr_text_sims = torch.stack(curr_text_sims).permute(1,0,2).reshape(chunck_size, -1)
        all_text_sims.append(curr_text_sims)
        
        text_input = None
        text_features = None
        torch.cuda.empty_cache()
       
    all_text_sims = torch.stack(all_text_sims).view(test_size, test_size) 
    print(all_text_sims.shape)
    recall = evaluate(all_text_sims.numpy())
    
    print("Recall: ", recall)
    print(model_path)
    print(mask_image)
    
    return recall


if __name__ == '__main__':
    # model_path = sys.argv[1]
    # use_mask = sys.argv[2]
    
    # if model_path == 'None':
    #     model_path = None
    # if use_mask == 'False':
    #     use_mask = False
    # elif use_mask == 'True':
    #     use_mask = True
    # else:
    #     raise('not supported')
    
    models_paths = [('/mnt/sb/fairness/log 01_11_2023 07:22:49/model_19_429.pt', True)] # Recall:  (29.58, 54.44, 65.46)
    # models_paths = [('/mnt/sb/fairness/log 01_11_2023 07:22:49/model_19_429.pt', False)] # not relevant, not zero shot (39.54, 66.4, 77.02)
    # models_paths = [(None, False)] # (29.36, 54.32, 65.06)
    # models_paths = [(None, True)]
    
    print(models_paths)

    device = "cuda" # tbd
    print(device)
    
    # Recall:  (39.94, 67.42, 77.2)
    # '/mnt/sb/fairness/log 01_11_2023 07:22:49/model_19_429.pt'

    print('########################')

    for (model_path, mask_image) in models_paths:
        run_test(model_path, mask_image, device)
        torch.cuda.empty_cache()

