import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
# display plots in this notebook
import cv2
import json

import os

def set_alpha(al):
    al_new = []
    scale = 1./al[0][1]
    for idxx, items in enumerate(al):
        alp = items[1]*scale
        if alp<0.4:
            alp = 0.4
        al_new.append([items[0], alp])

    return  al_new

def gongshi(x):
    return -2*x**2 + 3*x

inputjson = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/new_adj_t.json'))
for i,item in enumerate(inputjson):
    img_id = item['img_id']
    file_name = img_id.zfill(12)
    im_file = '/mnt/data/linkaiyi/mscoco/train2014/COCO_train2014_' + file_name + '.jpg'
    im = cv2.imread(im_file)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)

    bboxes = item['box']
    area_idx = item['predict_idx'][0]
    an = item['answer']
    predict_an = item['predict_an'][1:-1]
    adj = item['adj_diag']

    adj_line = {}
    # for i in range(len(adj)):
    #     for j in range(len(adj[i])):
    #         if adj[i][j] == 0:
    #             continue
    #         else:

    for w in range(len(adj)):
        if adj[w] != 0:
            adj_line[w] = adj[w]
        else:
            continue

    sort_adj = sorted(adj_line.items(), key=lambda m: m[1], reverse=True)
    sort_adj = sort_adj[:12]
    scale_adj = set_alpha(sort_adj)
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

    core_dot1 = (bbox[0] + bbox[2]) / 2
    core_dot2 = (bbox[1] + bbox[3]) / 2
    plt.scatter(core_dot1, core_dot2, color='r', s=25)

    for idx,item in enumerate(scale_adj):
        bbox = bboxes[item[0]]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        if item[0] != area_idx:
            dot1 = (bbox[0] + bbox[2]) / 2
            dot2 = (bbox[1] + bbox[3]) / 2
            plt.scatter(dot1, dot2, color='r', s=5)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linestyle="-", linewidth=0.8, alpha=0.8)
                )
            plt.plot([core_dot1, dot1], [core_dot2, dot2], color='w', alpha=item[1], linewidth=2.2)
        else:
            continue

    plt.axis('off')
    plt.savefig('/mnt/data/xiaojinhui/wangtan_MM/paper/image' + str(i) + '.jpg')
    plt.close()