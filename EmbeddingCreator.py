import gensim
import pickle
import re
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

count = 0
overall_length = 0
mapping = {}
embeddings = gensim.models.KeyedVectors.load_word2vec_format('./data/W2V300_GNews.bin', binary=True)
emebeddings = embeddings.wv


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    return " ".join(text.split()).lower()

def get_embedding(text):
    try:
        text = clean_text(text)
        text_embedding = []
        for word in text.split(' '):
            if word in embeddings:
                text_embedding.append(embeddings[word])
            elif word not in emebeddings and word.isalpha():
                if word == 'a':
                    pass
                else:
                    word_split = list(word)
                    word_split_vec = [embeddings[elem] for elem in word_split
                                      if elem in embeddings]
                    word_vec = [sum(i) for i in zip(*word_split_vec)]
                    word_vec = np.array((np.array(word_vec) / len(word_vec)).tolist())
                    text_embedding.append(word_vec)
            else:  # if numbers or other symbols
                text_embedding.append([float(0)] * 300)
        embedding = np.average(np.array(text_embedding), axis=0)
        global count
        count += 1
        if count % 100 == 0:
            print(count/overall_length)
        mapping[text] = embedding
    except:
        print(text)
        pass

def pickle_save(data, destination_dir):
    handle = open(destination_dir, 'wb')
    pickle.dump(data, handle)
    return

def create_dict(dataset):
    CHUNKSIZE = 100000
    global overall_length
    global count
    test_file = './data/' + dataset + '_test.csv'
    train_file = './data/' +dataset + '_train.csv'
    for train_dump in pd.read_csv(train_file, encoding="ISO-8859-1", chunksize=CHUNKSIZE):
        train_pool = ThreadPool(10000)
        train_dump = train_dump['Text'].as_matrix()
        overall_length = len(train_dump)
        train_pool.map(get_embedding, train_dump)
        count = 0
        train_pool.close()
        train_pool.join()
    for test_dump in pd.read_csv(test_file, encoding="ISO-8859-1", chunksize=CHUNKSIZE):
        test_pool = ThreadPool(10000)
        test_dump = test_dump['Text'].as_matrix()
        overall_length = len(test_dump)
        test_pool.map(get_embedding, test_dump)
        count = 0
        test_pool.close()
        test_pool.join()
    pickle_save(mapping, './data/' + dataset + '_embeddings.pickle')


if __name__ == '__main__':
    for item in ['AmazonRev']:
        create_dict(item)
