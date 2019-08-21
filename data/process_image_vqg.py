import csv
import base64
import numpy as np
import sys
import h5py
import pandas as pd
from tqdm import tqdm
import base64
import argparse
import json
import os

csv.field_size_limit(sys.maxsize)

def feature2hdf5(phase):
    FIELDNAMES = ['image_id', 'image_w', 'image_h',
                  'num_boxes', 'boxes', 'features', 'cls']

    infile = '/mnt/data/xiaojinhui/wangtan/vqa2_train_new.tsv'
    in_data = {}


    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(
            tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = str(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['cls'] = str(item['cls'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                encoded_str = base64.decodestring(
                    item[field].encode('utf-8'))
                item[field] = np.frombuffer(encoded_str,
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            if item['image_id'] == '76753':
                print(item)
            in_data[item['image_id']] = item

    # convert dict to pandas dataframe
    train = pd.DataFrame.from_dict(in_data)
    train = train.transpose()

    # create image sizes csv
    print('Writing image sizes csv...')
    d = train.to_dict()
    dw = d['image_w']
    dh = d['image_h']
    d = [dw, dh]
    dwh = {}
    for k in dw.keys():
        dwh[k] = np.array([d0[k] for d0 in d])
    image_sizes = pd.DataFrame(dwh)
    image_sizes.to_csv(phase + '_image_size.csv')

    # select bounding box coordinates and fill hdf5
    h = h5py.File(phase + '_box.hdf5', mode='w')
    t = train['boxes']
    d = t.to_dict()
    print('Creating bounding box file...')
    for k, v in tqdm(d.items()):
        h.create_dataset(str(k), data=v)
    if h:
        h.close()




    # select features and fill hdf5
    h = h5py.File(phase + '_feature.hdf5', mode='w')
    t = train['features']
    d = t.to_dict()
    print('Creating image features file...')
    for k, v in tqdm(d.items()):
        h.create_dataset(str(k), data=v)
    if h:
        h.close()



    # select classes and fill hdf5
    h = h5py.File(phase + '_cls.hdf5', mode='w')
    t = train['cls']
    d = t.to_dict()
    print('Creating class file...')
    for k, v in tqdm(d.items()):
        h.create_dataset(str(k), data=v)
    if h:
        h.close()


def drawarea():
    inputjson = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/draw_findarea.json'))

    FIELDNAMES = ['image_id', 'image_w', 'image_h',
                  'num_boxes', 'boxes', 'features', 'cls']

    infile = '/mnt/data/xiaojinhui/wangtan/vqa2_train_new.tsv'
    in_data = {}
    new = []

    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(
            tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = str(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['cls'] = str(item['cls'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                encoded_str = base64.decodestring(
                    item[field].encode('utf-8'))
                item[field] = np.frombuffer(encoded_str,
                                            dtype=np.float32).reshape((item['num_boxes'], -1))

            in_data[item['image_id']] = item

    for item in inputjson:
        item['info'] = in_data[item['img_id']]
        new.append(item)
    json.dump(new, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/draw/drawarea_final.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Preprocessing for VQA v2 image data')
    parser.add_argument('-d', '--data', nargs='+', help='trainval, and/or test, list of data phases to be processed', required=True)
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    phase_list = args.data

    for phase in phase_list:
        # First download and extract
        #feature2hdf5(phase)
        drawarea()

    print('Done')

