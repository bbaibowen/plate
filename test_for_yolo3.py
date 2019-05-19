'''
author:baibowen
'''

from yolo3_bulid import YOLO
import cv2
import numpy as np
import os


model = YOLO()
DATA_DIR = './text_img/'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
OUT_DIR = './result/'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
OUT_TEXT = './result/out.txt'
if os.path.exists(OUT_TEXT):
    os.remove(OUT_TEXT)


def get_img_dir():
    img_dir = os.listdir(DATA_DIR)
    imgs_dir = [DATA_DIR + i for i in img_dir]
    return imgs_dir

def check():
    for i,j in enumerate(get_img_dir()):
        im,text = model.detect_image(j)
        cv2.imwrite(OUT_DIR + 'out{}.jpg'.format(str(i)),im)
        with open(OUT_TEXT,'a+',encoding='utf-8') as f:
            f.write(j + '：' + text + '\n')
    print('done')

if __name__ == '__main__':
    check()