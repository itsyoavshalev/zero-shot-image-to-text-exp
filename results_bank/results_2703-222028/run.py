import argparse
import torch
import clip
from model.ZeroCLIP import CLIPTextGenerator
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
import os
import shutil
from coco_data_loader import ImagesDataset
from flickr_data_loader import FlickrDataset
from datetime import datetime
from tqdm import tqdm
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/sb/datasets/COCO/val2014/')
    parser.add_argument("--filter_json_path", type=str, default='/mnt/sb/datasets/COCO/dataset_coco_karpathy.json', help="json to filter db items, e.g karpathy split")
    parser.add_argument("--db_start_idx", type=int, default=0)
    parser.add_argument("--db_num_images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    # parser.add_argument("--reset_context_delta", action="store_true", help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--multi_gpu", action="store_true")

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_path", type=str, default='example_images/captions/COCO_val2014_000000008775.jpg',
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])

    args = parser.parse_args()

    return args

def run(args, img_path):
    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))

    # dataset = FlickrDataset('/mnt/sb/datasets/flickr30k_entities/', text_generator.clip_preprocess)
    dataset = ImagesDataset(args.data_path, text_generator.clip_preprocess, start_index=args.db_start_idx, count=args.db_num_images, filter_json_path=args.filter_json_path)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    results = []
    results_base_path = './results_' + datetime.now().strftime("%d%m-%H%M%S") + '/'
    results_inputs_base_path = results_base_path + 'inputs/'
    results_path = results_base_path + 'results.json'
    if os.path.isdir(results_base_path):
        shutil.rmtree(results_base_path)
    os.makedirs(results_base_path, exist_ok=True)
    os.makedirs(results_inputs_base_path, exist_ok=True)
    [shutil.copyfile(x, results_base_path + x) for x in os.listdir('./') if '.py' in x]

    for idx, current_data in tqdm(enumerate(dataloader)):
        clip_images_prep = current_data['prep_images'].to(text_generator.device)[0]
        clip_images_raw = current_data['raw_images'].to(text_generator.device)[0]
        db_item_ids = current_data['ids']
        with torch.no_grad():
            image_features_prep = text_generator.get_img_feature(text_generator.clip_raw, None, clip_imgs=clip_images_prep)
        image_features_exp = text_generator.get_img_feature(text_generator.clip_exp, None, clip_imgs=clip_images_prep)
        captions = text_generator.run(image_features_prep, image_features_exp, clip_images_raw, args.cond_text, beam_size=args.beam_size)

        encoded_captions = [text_generator.clip_raw.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
        encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (torch.cat(encoded_captions) @ image_features_prep.t()).squeeze().argmax().item()

        print(captions)
        print('best clip:', args.cond_text + captions[best_clip_idx])

        all_captions = ' # '.join(captions)
        results.append({"id": db_item_ids[0][0],
                        "best_clip_res": captions[best_clip_idx],
                        "captions": all_captions})
        with open(results_path, 'w') as f:
            json.dump(results, f)

def run_arithmetic(args, imgs_path, img_weights):
    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])

if __name__ == "__main__":
    args = get_args()

    if args.run_type == 'caption':
        run(args, img_path=args.caption_img_path)
    elif args.run_type == 'arithmetics':
        args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
        run_arithmetic(args, imgs_path=args.arithmetics_imgs, img_weights=args.arithmetics_weights)
    else:
        raise Exception('run_type must be caption or arithmetics!')