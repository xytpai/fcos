import torch
import math, random
from PIL import Image
import torchvision.transforms as transforms



def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    img = transforms.ColorJitter(brightness=brightness, 
                contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, boxes



def random_rotation(img, boxes, degree=5):
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



def _box_inter(box1, box2):
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    return inter



def random_crop_resize(img, boxes, size, 
        crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.8, attempt_max=10):
    success = False
    for attempt in range(attempt_max):
        # choose crop size
        area = img.size[0] * img.size[1]
        target_area = random.uniform(crop_scale_min, 1.0) * area
        aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
        w = int(round(math.sqrt(target_area * aspect_ratio_)))
        h = int(round(math.sqrt(target_area / aspect_ratio_)))
        if random.random() < 0.5:
            w, h = h, w
        # if size is right then random crop
        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            # check
            crop_box = torch.FloatTensor([[y, x, y+h, x+w]])
            inter = _box_inter(crop_box, boxes) # [1,N] N can be zero
            box_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) # [N]
            mask = inter>0.0001 # [1,N] N can be zero
            inter = inter[mask] # [1,S] S can be zero
            box_area = box_area[mask.view(-1)] # [S]
            box_remain = inter.view(-1) / box_area # [S]
            if box_remain.shape[0] != 0:
                if bool(torch.min(box_remain > remain_min)):
                    success = True
                    break
            else:
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
    return img, boxes
