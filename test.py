import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from torchvision import transforms
import cv2

if __name__ == '__main__':
    data_path = './input/'
    save_path = './result_map/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name in os.listdir(data_path):
        _images = os.path.join(data_path, name)
        _save = os.path.join(save_path, name.replace('.jpg', '.png'))
        img = cv2.imread(_images)
        img_size = img.shape
        binary = cv2.imread(_images, cv2.IMREAD_GRAYSCALE)
        binary = cv2.Canny(binary, 0, 250)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area = 0
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if area < w * h:
                max_x, max_y, max_w, max_h, area = x, y, w, h, w * h
        cv2.rectangle(img, (max_x, max_y), (max_x + max_w, max_y + max_h), (153, 153, 0), 5)
        cv2.imwrite(_save, img)

