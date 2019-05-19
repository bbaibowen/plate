import numpy as np
import torch
import cv2
from Recognition_network import build_lprnet

model = build_lprnet(phase='test')
model.load_state_dict(torch.load('./checkpoints/Final_LPRNet_model.pth', 'cpu'))
IMG_SIZE = (94,24)
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

def util_im(path = True,im = None):


    img = cv2.imread(im) if path else im
    img = cv2.resize(img,IMG_SIZE).astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img,(2,0,1))
    img = np.expand_dims(img,axis=0)
    return torch.Tensor(img)

def PR(im,path = False):
    prebs = model(util_im(path=path,im = im))
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    lbs = []
    for i in range(prebs.shape[0]):
        preb = prebs[i,:,:]
        preb_label = []
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:,j],axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    for i ,label in enumerate(preb_labels):
        lb = ''
        for i in label:
            lb += CHARS[i]
        lbs.append(lb)
    return lbs



if __name__ == '__main__':
    lbs = PR('./2.jpg',path=True)
    print(lbs)