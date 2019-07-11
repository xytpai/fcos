import os, json
import numpy as np
from pycocotools.coco import COCO

root = input('coco root: ')
set_name = input('train2017/val2017: ')
f = open('coco_'+set_name+'.txt', 'w')
coco = COCO(os.path.join(root, 
            'annotations', 'instances_' + set_name + '.json'))

image_ids = coco.getImgIds()
categories = coco.loadCats(coco.getCatIds())
categories.sort(key=lambda x: x['id'])
classes             = {'background':0}
coco_labels         = {}
coco_labels_inverse = {}
LABEL_NAMES = []
for c in categories:
    coco_labels[len(classes)] = c['id']
    coco_labels_inverse[c['id']] = len(classes)
    classes[c['name']] = len(classes)
    labels = {}
    for key, value in classes.items():
        labels[value] = key
for key in labels:
    LABEL_NAMES.append(labels[key])
valid_image_ids=[]
for idx in range(len(image_ids)):
    image_info = coco.loadImgs(image_ids[idx])[0]
    image = image_info['file_name']
    annotations_ids = coco.getAnnIds(imgIds=image_ids[idx], iscrowd=False)
    coco_annotations = coco.loadAnns(annotations_ids)
    boxes = []
    labels = []
    for _idx, a in enumerate(coco_annotations):
        if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            continue
        boxes.append(np.array(a['bbox']))
        labels.append(np.array(coco_labels_inverse[a['category_id']]))
    if len(boxes) <= 0:
        continue
        # boxes = [np.array([0,0,0,0])]
        # labels = [np.array([0])]
    valid_image_ids.append(image_ids[idx])
    boxes = np.stack(boxes)
    labels = np.stack(labels)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    boxes = np.stack([boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2]], axis=1)
    boxes = boxes.astype(np.int32)
    labels = labels.astype(np.int32).reshape((len(labels),1))
    info = np.concatenate((boxes, labels), axis=1).reshape(-1)
    line = image + ' ' + " ".join([str(x) for x in info]) + '\n'
    f.writelines(line)
    print('  {}/{}'.format(idx, len(image_ids)), end='\r')
f.close()
if set_name == 'val2017':
    f = open('coco_name.txt', 'w')
    for i in range(len(LABEL_NAMES)):
        line = LABEL_NAMES[i] + '\n'
        f.writelines(line)
    f.close()
    coco_table = {
        'coco_labels':coco_labels, 
        'coco_labels_inverse':coco_labels_inverse, 
        'val_image_ids':valid_image_ids
    }
    json_str = json.dumps(coco_table, indent=4)
    with open('coco_table.json', 'w') as json_file:
        json_file.write(json_str)
