import os
import json
import shutil

images_path = '/mnt/sb/datasets/COCO/val2014/'
methods = [{'name':'nft_exp_coco_05_1', 'path':'./results_1502-230824_coco_ours/'}, {'name':'zero_coco', 'path':'./results_2002-222914_coco_zero/'}, {'name':'ft_exp_coco_05_1', 'path':'./results_2503-090350/'}, {'name':'nft_exp_coco_05_05', 'path':'./results_2503-174221/'}]

output_path = './html/'
output_images_path = output_path + 'images/'
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)
os.mkdir(output_images_path)
output_html_file_path = output_path + 'main.html'

ids = []
for method in methods:
    tmp_ids = []
    with open(method['path'] + 'results.json', 'r') as method_json_file:
        method_json_data = json.load(method_json_file)
    method['json_data'] = method_json_data

    for item in method_json_data:
        tmp_ids.append(item['id'])

    tmp_ids = list(set(tmp_ids))
    ids.append(tmp_ids)

intersection = set(ids[0])

for sub_list in ids[1:]:
    intersection.intersection_update(sub_list)

ids = list(intersection)

print(len(ids))

html_lines = []
html_lines.append('<!DOCTYPE html>')
html_lines.append('<html>')
html_lines.append('<body>')
html_lines.append('<table>')

html_dict = {}

for sample_id in ids:
    methods_sents = {}
    for method in methods:
        method_sent = [x['best_clip_res'] for x in method['json_data'] if x['id'] == sample_id][0].strip().capitalize()
        methods_sents[method['name']] = method_sent
    if len(methods_sents) != len(methods):
        continue

    src_img_path = images_path + sample_id
    target_img_path = output_images_path + sample_id
    shutil.copyfile(src_img_path, target_img_path)
    html_lines.append(f'<tr><td><img width=\"224\" height=\"224\" src=\"./images/{sample_id}\"/></td></tr>')
    html_lines.append(f'<tr><td><h4>{sample_id}</h4><br\>')
    for k,v in methods_sents.items():
        html_lines.append(f'<h4>{k}: {v}</h4><br\>')
    html_lines.append(f'</td></tr>')

html_lines.append('</table>')
html_lines.append('</body>')
html_lines.append('</html>')

with open(output_html_file_path, "w") as html_file:
    html_file.writelines(html_lines)

        

