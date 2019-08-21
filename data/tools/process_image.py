import csv
import base64
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h',
              'num_boxes', 'boxes', 'features', 'csl']
infile = '/mnt/data/xiaojinhui/wangtan/v2_train_new.tsv.0'
in_data = {}

with open(infile, "r") as tsv_in_file:
    reader = csv.DictReader(
        tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for item in reader:
        item['image_id'] = str(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['csl'] = str(item['csl'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'features']:
            encoded_str = base64.decodestring(
                item[field].encode('utf-8'))
            item[field] = np.frombuffer(encoded_str,
                                        dtype=np.float32).reshape((item['num_boxes'], -1))
        in_data[item['image_id']] = item