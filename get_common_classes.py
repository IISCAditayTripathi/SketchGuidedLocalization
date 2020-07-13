import numpy
import json
import os
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO

dataDir = 'data/ms_coco_data'
dataType = 'train2014'

annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())

nms = [cat['name'] for cat in cats]
draw_classes = pickle.load(open('/scratche/home/<user>/imagenet/label2idx_draw.pkl', 'rb'))
draw_classes = list(draw_classes.keys())
common_classes = set(draw_classes) - set(nms)
common_classes_v2 = list(set(draw_classes)- common_classes)

class2label = {}
label2class = {}
for cl in common_classes_v2:
    class2label[cl] = len(class2label)
    label2class[len(label2class)] = cl

print(class2label)
print(label2class)
pickle.dump(class2label, open('class2label_common_classes.pkl', 'wb'))
pickle.dump(label2class, open('label2class_common_classes.pkl', 'wb'))
print(len(common_classes_v2))

draw_data_paths = '/scratche/home/<user>/imagenet/processed_quick_draw_paths.pkl'
paths = pickle.load(open(draw_data_paths, 'rb'))

train_paths = paths['train_x']
valid_paths = paths['valid_x']
test_paths = paths['test_x']

new_train_paths = []
new_valid_paths = []
new_test_paths = []
for path in tqdm(train_paths):
    for cl in common_classes_v2:
        label = path.split('/')[-2]
        if label == cl:
        #if cl in path:
            new_train_paths.append(path)

for path in tqdm(valid_paths):
    for cl in common_classes_v2:
        label = path.split('/')[-2]
        if label == cl:
        #if cl in path:
            new_valid_paths.append(path)
for path in tqdm(test_paths):
    for cl in common_classes_v2:
        label = path.split('/')[-2]
        if label == cl:
        #if cl in path:
            new_test_paths.append(path)
print(len(new_test_paths))

print(len(new_train_paths))
print(len(new_valid_paths))


new_paths = {'train_x': new_train_paths, 'valid_x': new_valid_paths, 'test_x': new_test_paths}
pickle.dump(new_paths, open('processed_quick_draw_paths_common_classes.pkl', 'wb'))
