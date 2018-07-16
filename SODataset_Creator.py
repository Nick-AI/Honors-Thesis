import csv

text_file = './data/title_StackOverflow.txt'
label_file = './data/label_StackOverflow.txt'
train_file = "./data/SOData_train.csv"
test_file = "./data/SOData_test.csv"
train_writer = csv.writer(open(train_file, "w", newline="", encoding='utf8'), delimiter=",")
test_writer = csv.writer(open(test_file, "w", newline="", encoding='utf8'), delimiter=",")
header = ('Label', 'Text')
train_writer.writerow(header)
test_writer.writerow(header)

with open(text_file, encoding="utf8") as t:
    text = [x.strip('\n') for x in t.readlines()]
    #text = [x.replace('ï»¿', '') for x in t.readlines()]

with open(label_file, encoding="utf8") as l:
    labels = [x.strip('\n') for x in l.readlines()]
    #labels = [x.replace('ï»¿', '') for x in l.readlines()]

for i in range(len(text)):
    item = (labels[i], text[i])
    if i%3==0:
        test_writer.writerow(item)
    else:
        train_writer.writerow(item)

