import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
# display plots in this notebook
import cv2
import json

import os

inputjson = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/new_adj.json'))
for i,item in enumerate(inputjson):
    img_id = item['img_id']
    file_name = img_id.zfill(12)
    im_file = '/mnt/data/linkaiyi/mscoco/val2014/COCO_val2014_' + file_name + '.jpg'
    im = cv2.imread(im_file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)

    bboxes = item['box']
    area_idx = item['predict_idx'][0]
    an = item['answer']
    predict_an = item['predict_an'][1:-1]

    bbox = bboxes[area_idx]
    if bbox[0] == 0:
        bbox[0] = 1
    if bbox[1] == 0:
        bbox[1] = 1

    plt.gca().add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linestyle="--", linewidth=2.5, alpha=0.9)
            )
    plt.gca().text(bbox[0], bbox[1]-7,
                   '%s' % (predict_an),
                   bbox=dict(facecolor='blue', alpha=0.6),
                   fontsize=13, color='white')


    plt.axis('off')
    plt.savefig('/mnt/data/xiaojinhui/wangtan_MM/paper/areafind_t/image' + str(i) + '.jpg')
    plt.close()
