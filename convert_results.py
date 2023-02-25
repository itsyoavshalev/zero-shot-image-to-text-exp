import json

results_c = []
out_file = './results_1502-230824/results_c.json'

with open('./results_1502-230824/results.json', 'r') as jsonfile:
    data = json.load(jsonfile)

with open('/mnt/sb/datasets/annotations/captions_val2014.json', 'r') as jsonfile:
    anns = json.load(jsonfile)

ids_cache = []
img_cache = []

for x in data:
    img_name = x['id']
    image_id = int(img_name.split('_')[-1].split('.')[0])
    id = [x['id'] for x in anns['images'] if x['file_name'] == img_name][0]

    if image_id in img_cache:
        continue
    img_cache.append(image_id)

    if id in ids_cache:
        continue
    ids_cache.append(id)

    tmp_item = {}
    tmp_item['id'] = id
    tmp_item['image_id'] = image_id
    tmp_item['caption'] = x['best_clip_res']
    tmp_item['captions'] = x['captions']
    results_c.append(tmp_item)

print(len(results_c))
with open(out_file, 'w') as f:
    json.dump(results_c, f)

print('done')