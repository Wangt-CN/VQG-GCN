import numpy as np
import matplotlib.pyplot as plt
import pylab
from skimage import transform
# display plots in this notebook
import cv2

import os

# im_file = 'D:\neurltalk\VQG\base\new\COCO_train2014_000000076753.jpg'
im_file = '/mnt/data/linkaiyi/mscoco/train2014/COCO_train2014_000000076753.jpg'
im = cv2.imread(im_file)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)

boxes = np.array([[ 89.37634  ,  40.58811  , 220.58765  , 283.49945  ],
       [ 91.87245  ,  25.129236 , 475.9419   , 190.44545  ],
       [384.35458  , 182.72554  , 574.40564  , 342.79068  ],
       [365.25253  ,  40.655567 , 512.4547   , 319.925    ],
       [542.3482   ,  82.875755 , 639.2      , 385.38962  ],
       [465.04355  , 303.55182  , 622.4613   , 414.38876  ],
                  [351.86823, 69.503494, 639.2, 244.44368]])

cls = ['black television', 'brick wall', 'navy jeans', 'sitting man', " \
       "'blue shirt', 'black pants', 'brown couch']

boxes1 = np.array([[ 89.37634  ,  40.58811  , 220.58765  , 283.49945  ],
       [ 91.87245  ,  25.129236 , 475.9419   , 190.44545  ],
       [384.35458  , 182.72554  , 574.40564  , 342.79068  ],
       [365.25253  ,  40.655567 , 512.4547   , 319.925    ],
       [446.42032  ,  54.250793 , 616.0967   , 361.80762  ],
       [542.3482   ,  82.875755 , 639.2      , 385.38962  ],
       [  0.       , 278.84027  , 563.8551   , 479.2      ],
       [465.04355  , 303.55182  , 622.4613   , 414.38876  ],
       [442.64618  ,  94.40326  , 548.37384  , 203.95058  ],
       [412.05176  , 127.12758  , 442.7997   , 163.08128  ],
       [351.86823  ,  69.503494 , 639.2      , 244.44368  ],
       [443.7914   ,  63.79183  , 477.33905  ,  94.16733  ],
       [ 19.307196 ,   1.0354493, 160.10628  , 338.39197  ],
       [366.5342   , 297.4958   , 437.77325  , 347.4647   ],
       [  2.1270263,  22.507502 , 109.678055 , 114.09558  ],
       [197.25516  , 101.77095  , 544.91876  , 263.85217  ],
       [252.40878  , 209.29446  , 328.4393   , 255.8986   ],
       [178.6242   ,   8.848282 , 421.14316  ,  78.65724  ],
       [140.56015  ,  34.564087 , 600.03906  , 456.90234  ],
       [467.4383   ,   0.       , 572.70557  ,  96.97469  ],
       [353.8976   , 248.58926  , 404.88214  , 280.09448  ],
       [612.6061   , 393.34482  , 639.2      , 442.84344  ],
       [ 35.75791  ,  34.13137  , 171.6495   , 363.92682  ],
       [207.00517  ,  96.29607  , 338.05533  , 191.04037  ],
       [295.11276  , 233.86295  , 639.2      , 479.2      ],
       [397.94342  , 242.11026  , 639.2      , 479.2      ],
       [406.03217  , 134.35097  , 639.2      , 415.1749   ],
       [131.21977  ,  95.06123  , 209.3153   , 217.91447  ],
       [526.6397   , 154.51059  , 639.2      , 324.3034   ],
       [203.79349  , 187.96458  , 291.56488  , 266.08096  ],
       [ 26.460821 , 201.95566  , 138.70972  , 470.16534  ],
       [ 15.76853  ,  79.6505   , 411.1451   , 216.99951  ],
       [591.12115  , 104.3245   , 639.2      , 167.7816   ],
       [608.3323   , 405.59445  , 638.25006  , 445.32367  ],
       [312.1006   ,   0.       , 639.2      , 208.32718  ],
       [445.48203  ,  59.696167 , 484.8502   ,  97.001755 ]])

cls1 = ['black television', 'brick wall', 'blue jeans', 'sitting man', 'man', " \
       "'blue shirt', 'brown floor', 'black pants', 'blue shirt', 'hand', 'brown couch', " \
       "'glasses', 'brick wall', 'brown shoe', 'silver sink', 'brick wall', 'blue bag', " \
       "'white wall', 'room', 'brick wall', 'black shoe', 'white sock', 'black fireplace', " \
       "'brick wall', 'tan carpet', 'man', 'man', 'television', 'blue shirt', 'white cords', " \
       "'gray stairs', 'brick wall', 'brown hair', 'white shoe', 'brown wall', 'glasses']

for i in range(7):
    bbox = boxes[i]
    if bbox[0] == 0:
        bbox[0] = 1
    if bbox[1] == 0:
        bbox[1] = 1

    plt.gca().add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=2, alpha=0.5)
            )
    plt.gca().text(bbox[0], bbox[1] - 2,
                '%s' % (cls[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
plt.axis('off')
plt.savefig('/mnt/data/xiaojinhui/wangtan_MM/paper/intro.jpg')