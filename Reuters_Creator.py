from nltk.corpus import reuters
import csv

train_file = "./data/Reuters_train.csv"
test_file = "./data/Reuters_test.csv"
train_writer = csv.writer(open(train_file, "w", newline=""), delimiter=",")
test_writer = csv.writer(open(test_file, "w", newline=""), delimiter=",")
categories = [cat for cat in reuters.categories()]
header = ('Label', 'Text')
train_writer.writerow(header)
test_writer.writerow(header)
count = 0
for cat in categories:
    for doc in reuters.fileids(cat):
        if doc.startswith("train"):
            item = (cat, reuters.raw(doc))
            train_writer.writerow(item)
        if doc.startswith("test"):
            item = (cat, reuters.raw(doc))
            test_writer.writerow(item)