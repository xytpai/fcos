import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os, math, random
from PIL import Image, ImageDraw
import torch.utils.data as data
import torchvision.transforms as transforms



class Dataset_CSV(data.Dataset):
    def __init__(self, root, list_file, name_file, 
                    size=1025, train=True, normalize=True, boxarea_th=35,
                    img_scale_min=0.8, augmentation=None):
        ''''
        Provide:
        self.fnames:      [fname1, fname2, fname3, ...] # image filename
        self.boxes:       [FloatTensor(N1,4), FloatTensor(N2,4), ...]
        self.labels:      [LongTensor(N1), LongTensor(N2), ...]
        self.LABEL_NAMES: ['background', 'person', 'bicycle', ...] in name_file

        Note:
        - root: folder for jpg images
        - list_file: img_name.jpg ymin1 xmin1 ymax1 xmax1 label1 ... /n
        - name_file: background /n class_name1 /n class_name2 /n ...
        - if not have object -> xxx.jpg 0 0 0 0 0
        - remove box when area < boxarea_th
        - label == 0 indecates background 
        '''
        self.root = root
        self.size = size
        self.train = train
        self.normalize = normalize
        self.boxarea_th = boxarea_th
        self.img_scale_min = img_scale_min
        self.augmentation = augmentation
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.LABEL_NAMES = []
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)
            for line in lines:
                splited = line.strip().split()
                self.fnames.append(splited[0])
                num_boxes = (len(splited) - 1) // 5
                box = []
                label = []
                for i in range(num_boxes):
                    ymin = splited[1+5*i]
                    xmin = splited[2+5*i]
                    ymax = splited[3+5*i]
                    xmax = splited[4+5*i]
                    c = splited[5+5*i]
                    box.append([float(ymin),float(xmin),float(ymax),float(xmax)])
                    label.append(int(c))
                self.boxes.append(torch.FloatTensor(box))
                self.labels.append(torch.LongTensor(label))
        with open(name_file) as f:
            lines = f.readlines()
            for line in lines:
                self.LABEL_NAMES.append(line.strip())
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    

    def __len__(self):
        return self.num_samples
    

    def __getitem__(self, idx):
        '''
        Return:
        img:    FloatTensor(3, size, size)
        boxes:  FloatTensor(box_num, 4)
        labels: LongTensor(box_num)
        loc:    FloatTensor(4)
        scale:  float scalar
        '''
        img = Image.open(os.path.join(self.root, self.fnames[idx]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = self.boxes[idx].clone()
        boxes[:, :2].clamp_(min=1)
        boxes[:, 2].clamp_(max=float(img.size[1])-1)
        boxes[:, 3].clamp_(max=float(img.size[0])-1)
        labels = self.labels[idx].clone()
        if self.train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            # TODO: other augmentation (img, boxes)
            if self.augmentation is not None:
                img, boxes = self.augmentation(img, boxes)
            # standard procedure
            if random.random() < 0.5:
                img, boxes, loc, scale = random_resize_fix(img, boxes, self.size, self.img_scale_min)
            else:
                img, boxes, loc, scale = center_fix(img, boxes, self.size)
        else:
            img, boxes, loc, scale = center_fix(img, boxes, self.size)
        hw = boxes[:, 2:] - boxes[:, :2] # [N,2]
        area = hw[:, 0] * hw[:, 1]       # [N]
        mask = area >= self.boxarea_th
        boxes = boxes[mask]
        labels = labels[mask]
        img = transforms.ToTensor()(img)
        if self.normalize:
            img = self.normalizer(img)
        return img, boxes, labels, loc, scale


    def collate_fn(self, data):
        '''
        Return:
        img     FloatTensor(batch_num, 3, size, size)
        boxes   FloatTensor(batch_num, N_max, 4)
        Labels  LongTensor(batch_num, N_max)
        loc     FloatTensor(batch_num, 4)
        scale   FloatTensor(batch_num)
        '''
        img, boxes, labels, loc, scale = zip(*data)
        img = torch.stack(img)
        batch_num = len(boxes)
        N_max = 0
        for b in range(batch_num):
            n = boxes[b].shape[0]
            if n > N_max: N_max = n
        boxes_t = torch.zeros(batch_num, N_max, 4)
        labels_t = torch.zeros(batch_num, N_max).long()
        for b in range(batch_num):
            boxes_t[b, 0:boxes[b].shape[0]] = boxes[b]
            labels_t[b, 0:boxes[b].shape[0]] = labels[b]
        loc = torch.stack(loc)
        scale_t = torch.FloatTensor(scale)
        return img, boxes_t, labels_t, loc, scale_t



def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,3]
        xmax = w - boxes[:,1]
        boxes[:,1] = xmin
        boxes[:,3] = xmax
    return img, boxes



def center_fix(img, boxes, size):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    sw = sh = float(size) / size_max
    ow = int(w * sw + 0.5)
    oh = int(h * sh + 0.5)
    ofst_w = round((size - ow) / 2.0)
    ofst_h = round((size - oh) / 2.0)
    img = img.resize((ow,oh), Image.BILINEAR)
    img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
    if boxes.shape[0] != 0:
        boxes = boxes*torch.FloatTensor([sh,sw,sh,sw])
        boxes += torch.FloatTensor([ofst_h, ofst_w, ofst_h, ofst_w])
    loc = torch.FloatTensor([ofst_h, ofst_w, ofst_h+oh, ofst_w+ow])
    return img, boxes, loc, sw



def random_resize_fix(img, boxes, size, img_scale_min):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    scale_rate = float(size) / size_max
    scale_rate *= random.uniform(img_scale_min, 1.0)
    ow, oh = int(w * scale_rate + 0.5), int(h * scale_rate + 0.5)
    img = img.resize((ow,oh), Image.BILINEAR)
    if boxes.shape[0] != 0:
        boxes = boxes*torch.FloatTensor([scale_rate, scale_rate, scale_rate, scale_rate])
    max_ofst_h = size - oh
    max_ofst_w = size - ow
    ofst_h = random.randint(0, max_ofst_h)
    ofst_w = random.randint(0, max_ofst_w)
    img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
    if boxes.shape[0] != 0:
        boxes += torch.FloatTensor([ofst_h, ofst_w, ofst_h, ofst_w])
    loc = torch.FloatTensor([ofst_h, ofst_w, ofst_h+oh, ofst_w+ow])
    return img, boxes, loc, scale_rate



def show_bbox(img, boxes, labels, NAME_TAB, file_name=None, matplotlib=False):
    '''
    img:      FloatTensor(3, H, W)
    boxes:    FloatTensor(N, 4)
    labels:   LongTensor(N)
    NAME_TAB: ['background', 'class_1', 'class_2', ...]
    file_name: 'out.bmp' or None
    '''
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    drawObj = ImageDraw.Draw(img)
    for box_id in range(boxes.shape[0]):
        lb = int(labels[box_id])
        if lb > 0:
            box = boxes[box_id]
            if NAME_TAB is not None:
                strlen = len(NAME_TAB[lb])
                drawObj.rectangle((box[1],box[0],box[1]+strlen*6,box[0]+12), fill='blue')
                drawObj.text((box[1],box[0]), NAME_TAB[lb])
            drawObj.rectangle((box[1],box[0],box[3],box[2]), outline='red')
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()



if __name__ == '__main__':

    import augment
    def aug_func_demo(img, boxes):
        if random.random() < 0.9:
            img, boxes = augment.colorJitter(img, boxes, 
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        if random.random() < 0.9:
            img, boxes = augment.random_rotation(img, boxes, degree=5)
        if random.random() < 0.9:
            img, boxes = augment.random_crop_resize(img, boxes, size=512, 
                            crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.1, 
                            attempt_max=10)
        return img, boxes

    #TODO: parameters
    train = True
    size = 1025
    boxarea_th = 32
    img_scale_min = 0.8
    augmentation = None
    batch_size = 8
    csv_root  = 'D:\\dataset\\coco17\\images'
    csv_list  = '../data/coco_val2017.txt'
    csv_name  = '../data/coco_name.txt'

    
    dataset = Dataset_CSV(csv_root, csv_list, csv_name, 
        size=size, train=train, normalize=False, boxarea_th=boxarea_th,
        img_scale_min=img_scale_min, augmentation=augmentation)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    for imgs, boxes, labels, locs, scales in dataloader:
        print(imgs.shape)
        print(boxes.shape)
        print(labels.shape)
        print(locs.shape)
        print(scales.shape)
        for i in range(len(boxes)):
            print(i, ': ', boxes[i].shape, labels[i].shape, locs[i], scales[i])
        # idx = int(input('idx:'))
        idx = 3
        print(labels[idx])
        print(boxes[idx][labels[idx]>0])
        print('avg px:', int(torch.min(locs[:, 2:] - locs[:, :2], dim=1)[0].mean()))
        show_bbox(imgs[idx], boxes[idx], labels[idx], dataset.LABEL_NAMES)
        # show_bbox(imgs[idx], boxes[idx], labels[idx], None)
        break
