import os
import numpy as np
import xml.etree.ElementTree as ET

root = input('voc root: ')
split = input('train/val/trainval/test: ') # train val trainval
test = False
if split == 'test':
    print('voc test, exclude difficult')
    test = True
id_list_file = os.path.join(root, 'ImageSets/Main/{0}.txt'.format(split))
ids = [id_.strip() for id_ in open(id_list_file)]
f = open('voc_'+split+'.txt', 'w')

VOC_LABEL_NAMES = (
    'background',#0
    'aeroplane',#1
    'bicycle',#2
    'bird',#3
    'boat',#4
    'bottle',#5
    'bus',#6
    'car',#7
    'cat',#8
    'chair',#9
    'cow',#10
    'diningtable',#11
    'dog',#12
    'horse',#13
    'motorbike',#14
    'person',#15
    'pottedplant',#16
    'sheep',#17
    'sofa',#18
    'train',#19
    'tvmonitor'#20
)

for i in range(len(ids)):
    id_ = ids[i]
    anno = ET.parse(os.path.join(root, 'Annotations', id_ + '.xml'))
    boxes, labels = list(), list()
    for obj in anno.findall('object'):
        if test and (int(obj.find('difficult').text) == 1):
            continue
        bndbox_anno = obj.find('bndbox')
        boxes.append([
            int(float(bndbox_anno.find(tag).text)) - 1
            for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        name = obj.find('name').text.lower().strip()
        labels.append(VOC_LABEL_NAMES.index(name))
    boxes = np.stack(boxes).astype(np.int32)
    labels = np.stack(labels).astype(np.int32).reshape((len(labels),1))
    info = np.concatenate((boxes, labels), axis=1).reshape(-1)
    image = id_ + '.jpg '
    line = image + " ".join([str(x) for x in info]) + '\n'
    f.writelines(line)
    print('  {}/{}'.format(i, len(ids)), end='\r')
f.close()
f = open('voc_name.txt', 'w')
for i in range(len(VOC_LABEL_NAMES)):
    line = VOC_LABEL_NAMES[i] + '\n'
    f.writelines(line)
f.close()
