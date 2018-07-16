import gensim
import pickle
import re
import time
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

count = 0
overall_length = 0
mapping = {}
t = []
embeddings = gensim.models.KeyedVectors.load_word2vec_format('./data/W2V300_GNews.bin', binary=True)
embeddings = embeddings.wv


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    return " ".join(text.split()).lower()


def get_embedding(text):
    start = time.time()

    text = clean_text(text)
    key = text
    text = text.split(' ')
    text_embedding = []
    factor = 10
    if len(text) < 10:
        factor = len(text)
    len_substring = int(len(text) / factor)
    for j in range(factor):
        substring_embedding = []
        for i in range(len_substring):
            for word in text:
                if word in embeddings:
                    substring_embedding.append(embeddings[word])
                elif word not in embeddings and word.isalpha():
                    if word == 'a':
                        substring_embedding.append(embeddings['A'])
                    else:
                        word.replace('a', 'A')
                        word_split = list(word)
                        word_split_vec = [embeddings[elem] for elem in word_split
                                          if elem in embeddings]
                        word_vec = [sum(i) for i in zip(*word_split_vec)]
                        word_vec = np.array((np.array(word_vec) / len(word_vec)).tolist())
                        substring_embedding.append(word_vec)
                else:  # if numbers or other symbols
                    substring_embedding.append([float(0)] * 300)
        text_embedding.append(np.average(np.array(substring_embedding), axis=0))
        text = text[len_substring:]
    while len(text_embedding) < factor:
        text_embedding.append([float(0)] * 300)

    global count
    count += 1
    if count % 10 == 0:
        t.append(time.time()-start)
        print(count/overall_length)
        print(count)
        if count > 300:
            print('Overall:', np.mean(t), '\n\n\n\n')

    mapping[key] = text_embedding


def pickle_save(data, destination_dir):
    handle = open(destination_dir, 'wb')
    pickle.dump(data, handle)
    return


def create_dict(dataset):
    CHUNKSIZE = 1000000
    test_file = './data/' + dataset + '_test.csv'
    train_file = './data/' +dataset + '_train.csv'
    train_pool = ThreadPool(10000)
    for train_dump in pd.read_csv(train_file, encoding="ISO-8859-1", chunksize=CHUNKSIZE):
        train_dump = train_dump['Text'].as_matrix()

        global overall_length
        global count
        overall_length = len(train_dump)

        train_pool.map(get_embedding, train_dump)
    train_pool.close()
    train_pool.join()
    test_dump = pd.read_csv(test_file, encoding="ISO-8859-1")
    test_dump = test_dump['Text'].as_matrix()
    overall_length = len(test_dump)
    count = 0
    test_pool = ThreadPool(10000)
    test_pool.map(get_embedding, test_dump)
    test_pool.close()
    test_pool.join()
    pickle_save(mapping, './data/' + dataset + '_embeddings.pickle')


if __name__ == '__main__':
    for item in ['AmazonRev']:
        create_dict(item)
