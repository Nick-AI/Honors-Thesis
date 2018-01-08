import gensim
import pickle
import re
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


test_file = './data/Reuters_test.csv'
train_file = './data/Reuters_train.csv'
mapping = {}
embeddings = gensim.models.KeyedVectors.load_word2vec_format('./data/W2V300_GNews.bin', binary=True)
emebeddings = embeddings.wv


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    return " ".join(text.split()).lower()

def get_embedding(text):
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
    mapping[text] = embedding

def pickle_save(data, destination_dir):
    handle = open(destination_dir, 'wb')
    pickle.dump(data, handle)
    return

def create_dict():
    train_pool = ThreadPool(4)
    test_pool = ThreadPool(2)
    train_dump = pd.read_csv(train_file, encoding="ISO-8859-1")
    train_dump = train_dump['Text'].as_matrix()
    test_dump = pd.read_csv(test_file, encoding="ISO-8859-1")
    test_dump = test_dump['Text'].as_matrix()
    train_pool.map(get_embedding, train_dump)
    test_pool.map(get_embedding, test_dump)
    test_pool.close()
    train_pool.close()
    test_pool.join()
    train_pool.join()
    pickle_save(mapping, './data/Reuters_embeddings.pickle')

if __name__ == '__main__':
    create_dict()