import csv
import sys

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls']

test = ['/mnt/data/xiaojinhui/wangtan/v2_val_new.tsv.%d' % i for i in range(2)]

outfile = '/mnt/data/xiaojinhui/wangtan/vqa2_val_new.tsv'

with open(outfile, 'ab') as tsvfile:
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)

    for infile in test:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                try:
                    writer.writerow(item)
                except Exception as e:
                    print e
