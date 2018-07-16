import os
import csv
import gzip

data_dir = './'  # direction to data
source_files = os.listdir(data_dir)
train_file = "./data/AmazonRev_train.csv"
test_file = "./data/AmazonRev_test.csv"
train_writer = csv.writer(open(train_file, "a", newline="", encoding='utf8'), delimiter=",")
test_writer = csv.writer(open(test_file, "a", newline="", encoding='utf8'), delimiter=",")
header = ('Label', 'Text', 'TfIDf', '1D-Embedding')
items = []
count = 0

train_writer.writerow(header)
test_writer.writerow(header)

for file in source_files:
    with gzip.open(data_dir + file) as f:
        label = file.replace('revies_Amazon_', '')
        label = label.replace('.json.gz', '')
        for l in f:
            item = eval(l)
            write_item = (label, item['reviewText'])
            if count % 3 == 0:
                test_writer.writerow(write_item)
            else:
                train_writer.writerow(write_item)
            count += 1
