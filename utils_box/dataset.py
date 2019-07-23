import torch
import numpy as np 
import matplotlib.pyplot as plt 
import os, math, random
from PIL import Image, ImageDraw
import torch.utils.data as data
import torchvision.transforms as transforms



class Dataset_CSV(data.Dataset):
    def __init__(self, root, list_file, name_file, size=641, 
                    train=True, transform=None, boxarea_th=25,
                    img_scale_min=0.2, crop_scale_min=0.1, aspect_ratio=(3./4, 4./3), remain_min=0.8):
        ''''
        Provide:
        self.fnames:      [fname1, fname2, fname3, ...] # image filename
        self.boxes:       [FloatTensor(N1,4), FloatTensor(N2,4), ...]
        self.labels:      [LongTensor(N1), LongTensor(N2), ...]
        self.LABEL_NAMES: ['background', 'person', 'bicycle', ...] in name_file

        Note:
        - root: folder of jpg images
        - list_file: img_name.jpg ymin1, xmin1, ymax1, xmax1, label1, ... /n
                     ...
        - name_file: background /n class_name1 /n class_name2 /n ...
        - if not have object -> xxx.jpg 0 0 0 0 0
        - if self.train == True: random_flip, random_resize_fix
          else: corner_fix
        - remove box when area <= boxarea_th
        - label == 0 indecates background 
        - box-4 indecates ymin, xmin, ymax, xmax
        '''
        self.root = root
        self.size = size
        self.train = train
        self.transform = transform
        self.boxarea_th = boxarea_th
        self.img_scale_min = img_scale_min
        self.crop_scale_min = crop_scale_min
        self.aspect_ratio = aspect_ratio
        self.remain_min = remain_min
        self.fnames = []
        self.boxes = []
        self.labels = []
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
        self.LABEL_NAMES = []
        for line in lines:
            self.LABEL_NAMES.append(line.strip())
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        '''
        Return:
        img:    FloatTensor(3, size, size) or PILImage(if self.transform==None)
        boxes:  FloatTensor(box_num, 4)
        labels: LongTensor(box_num)
        scale:  float scalar

        Note:
        - box_num can be zero
        - if self.train == True: random_flip, random_resize_fix
          else: corner_fix
        - remove box when area <= boxarea_th
        '''
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        size = self.size
        if self.train:
            img, boxes = random_flip(img, boxes)
            # img, boxes = random_rotation(img, boxes)
            img, boxes, scale = random_resize_fix(img, boxes, size,
                self.img_scale_min, self.crop_scale_min, self.aspect_ratio, self.remain_min)
            # img = transforms.ColorJitter(
            #         brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            #     )(img)
        else:
            img, boxes, scale = corner_fix(img, boxes, size)   
        hw = boxes[:, 2:] - boxes[:, :2] # [N,2]
        area = hw[:, 0] * hw[:, 1]       # [N]
        mask = area > self.boxarea_th
        boxes = boxes[mask]
        labels = labels[mask]
        if self.transform is not None:
            img = self.transform(img)
        return img, boxes, labels, scale

    def collate_fn(self, data):
        '''
        Return:
        img     FloatTensor(batch_num, 3, size, size),
        boxes   (FloatTensor(N1,4), FloatTensor(N2,4), ...)
        Labels  (LongTensor(N1), LongTensor(N2), ...),
        scale   (s1, s2, ...)

        Note:
        - Ni can be zero
        '''
        img, boxes, labels, scale = zip(*data)
        img = torch.stack(img, dim=0)
        return img, boxes, labels, scale



def corner_fix(img, boxes, size):
    w, h = img.size
    size_min = min(w,h)
    size_max = max(w,h)
    sw = sh = float(size) / size_max
    ow = int(w * sw + 0.5)
    oh = int(h * sh + 0.5)
    img = img.resize((ow,oh), Image.BILINEAR)
    img = img.crop((0, 0, size, size))
    if boxes.shape[0] != 0:
        boxes = boxes*torch.Tensor([sh,sw,sh,sw])
    return img, boxes, sw



def random_rotation(img, boxes, degree=6):
    d = random.uniform(-degree, degree)
    w, h = img.size
    rx0, ry0 = w/2.0, h/2.0
    img = img.rotate(d)
    a = -d / 180.0 * math.pi
    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = boxes[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        z = torch.FloatTensor([[y0,x0],[y1,x1],[y2,x2],[y3,x3]])
        tp = torch.zeros_like(z)
        tp[:,1] = (z[:,1] - rx0)*math.cos(a) - (z[:,0] - ry0)*math.sin(a) + rx0
        tp[:,0] = (z[:,1] - rx0)*math.sin(a) + (z[:,0] - ry0)*math.cos(a) + ry0
        ymax, xmax = torch.max(tp, dim=0)[0]
        ymin, xmin = torch.min(tp, dim=0)[0]
        boxes[i] = torch.stack([ymin,xmin,ymax,xmax])
    boxes[:,1::2].clamp_(min=0, max=w-1)
    boxes[:,0::2].clamp_(min=0, max=h-1)
    return img, boxes



def _box_inter(box1, box2, eps=1e-10):
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl+eps).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    return inter



def random_resize_fix(img, boxes, size, 
    img_scale_min=0.2, crop_scale_min=0.5, aspect_ratio=(3./4, 4./3), remain_min=0.9):
    while True:
        method = ['random_resize_fix', 'random_resize_crop',
                    'corner_fix']
        method = random.choice(method)
        if method == 'random_resize_fix':
            w, h = img.size
            size_min = min(w,h)
            size_max = max(w,h)
            scale_rate = float(size) / size_max
            scale_rate *= random.uniform(img_scale_min, 1.0)
            ow, oh = int(w * scale_rate + 0.5), int(h * scale_rate + 0.5)
            img = img.resize((ow,oh), Image.BILINEAR)
            if boxes.shape[0] != 0:
                boxes = boxes*torch.Tensor([scale_rate, scale_rate, scale_rate, scale_rate])
            max_ofst_h = size - oh
            max_ofst_w = size - ow
            ofst_h = random.randint(0, max_ofst_h)
            ofst_w = random.randint(0, max_ofst_w)
            img = img.crop((-ofst_w, -ofst_h, size-ofst_w, size-ofst_h))
            if boxes.shape[0] != 0:
                boxes += torch.FloatTensor([ofst_h, ofst_w, ofst_h, ofst_w])
            return img, boxes, 10
        elif method == 'corner_fix':
            return corner_fix(img, boxes, size)
        elif method == 'random_resize_crop':
            if boxes.shape[0] == 0:
                return corner_fix(img, boxes, size)
            success = False
            for attempt in range(5):
                area = img.size[0] * img.size[1]
                target_area = random.uniform(crop_scale_min, 1.0) * area
                aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
                w = int(round(math.sqrt(target_area * aspect_ratio_)))
                h = int(round(math.sqrt(target_area / aspect_ratio_)))
                if random.random() < 0.5:
                    w, h = h, w
                if w <= img.size[0] and h <= img.size[1]:
                    x = random.randint(0, img.size[0] - w)
                    y = random.randint(0, img.size[1] - h)
                    # TODO:check
                    crop_box = torch.FloatTensor([[y, x, y+h, x+w]])
                    inter = _box_inter(crop_box, boxes) # [1,N]
                    bh = boxes[:, 2] - boxes[:, 0]
                    bw = boxes[:, 3] - boxes[:, 1]
                    box_area = bh*bw # [N]
                    inter = inter.view(-1) # [N]
                    mask = inter>0.0001
                    inter = inter[mask] # [S]
                    box_area = box_area[mask] # [S]
                    box_remain = inter / box_area # [S]
                    if box_remain.shape[0] != 0:
                        if bool(torch.min(box_remain > remain_min)):
                            success = True
                            break
            if success:
                img = img.crop((x, y, x+w, y+h))
                boxes -= torch.Tensor([y,x,y,x])
                boxes[:,1::2].clamp_(min=0, max=w-1)
                boxes[:,0::2].clamp_(min=0, max=h-1)
                ow, oh = (size, size)
                sw = float(ow) / img.size[0]
                sh = float(oh) / img.size[1]
                img = img.resize((ow,oh), Image.BILINEAR)
                boxes *= torch.FloatTensor([sh,sw,sh,sw])
                # scale = max(img.shape[0], img.shape[1]) / float(size)
                return img, boxes, -10



def random_flip(img, boxes):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT) 
        w = img.width
        if boxes.shape[0] != 0:
            xmin = w - boxes[:,3]
            xmax = w - boxes[:,1]
            boxes[:,1] = xmin
            boxes[:,3] = xmax
    return img, boxes



def show_bbox(img, boxes, labels, NAME_TAB, file_name=None, matplotlib=False):
    img = transforms.ToPILImage()(img)
    drawObj = ImageDraw.Draw(img)
    i = 0
    for box in boxes:
        strlen = len(NAME_TAB[int(labels[i])])
        drawObj.rectangle((box[1],box[0],box[1]+strlen*6,box[0]+12), fill='blue')
        drawObj.text((box[1],box[0]), NAME_TAB[int(labels[i])])
        drawObj.rectangle((box[1],box[0],box[3],box[2]), outline='red')
        i += 1
    if file_name is not None:
        img.save(file_name)
    else:
        if matplotlib:
            plt.imshow(img, aspect='equal')
            plt.show()
        else: img.show()



if __name__ == '__main__':

    #TODO: parameters
    train = True
    size = 641
    area_th = 25
    batch_size = 8
    csv_root  = 'D:\\dataset\\VOC0712_trainval\\JPEGImages'
    csv_list  = '../data/voc_trainval.txt'
    csv_name  = '../data/voc_name.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    import json
    with open('../train.json', 'r') as load_f:
        cfg = json.load(load_f)
    
    dataset = Dataset_CSV(csv_root, csv_list, csv_name, 
        size=size, train=train, transform=transform, 
        boxarea_th = cfg['boxarea_th'],
        img_scale_min = cfg['img_scale_min'], 
        crop_scale_min = cfg['crop_scale_min'], 
        aspect_ratio = cfg['aspect_ratio'], 
        remain_min = cfg['remain_min'])
    dataloader = data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    for imgs, boxes, labels, scales in dataloader:
        p10=0
        n10=0
        print(imgs.shape)
        for i in range(len(boxes)):
            print(i, ': ', boxes[i].shape, labels[i].shape, scales[i])
            if int(scales[i]) == 10:
                p10 += 1
            if int(scales[i]) == -10:
                n10 += 1
        # print('+10/-10:', float(p10/n10))
        # idx = int(input('idx:'))
        idx = 0
        show_bbox(imgs[idx], boxes[idx], labels[idx], dataset.LABEL_NAMES)
        break
