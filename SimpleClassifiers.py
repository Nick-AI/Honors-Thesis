import pickle
import re
import time
import json
import pandas as pd
import numpy as np
from os import listdir
from os.path import join
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


class TfIDfClassifier:

    uses_ngrams = False
    n_gram_range = 1
    t_max_df = 0.3
    nb_features = 1024
    dataset = ""
    directory = "./"
    data_dir = join(directory, "data/")
    vector_file = data_dir + "features_" + dataset + ".pkl"

    def __init__(self, dataset, with_n_grams, n_gram_range, t_max_df, nb_features):
        self.dataset = dataset
        self.uses_ngrams = with_n_grams
        self.n_gram_range = n_gram_range
        self.t_max_df = t_max_df
        self.nb_features = nb_features

    def create_vectors(self, train_data):
        """Creates tfidf matrix from training data
        :param train_data: string
        """
        if self.uses_ngrams:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.t_max_df, ngram_range=self.n_gram_range,
                                         max_features=self.nb_features)
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.t_max_df, analyzer='word',
                                         max_features=self.nb_features)
        vectorizer.fit_transform(train_data['Text'].tolist())
        pickle.dump(vectorizer.vocabulary_, open(self.vector_file, "wb"))
        return

    def get_vectors(self, test_data):
        """Creates vectors for given query
        :param test_data: string
        """
        if self.uses_ngrams:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.t_max_df, ngram_range=self.n_gram_range,
                                     vocabulary=pickle.load(open(self.vector_file, "rb")))
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.t_max_df, analyzer='word',
                                         vocabulary=pickle.load(open(self.vector_file, "rb")))
        transformer = TfidfTransformer()
        vectors = transformer.fit_transform(vectorizer.fit_transform(test_data['Text'].tolist()))
        return vectors

class Classifiers:
    directory = "./"
    data_dir = join(directory, "data/")
    train_data = ""
    test_data = ""


    def __init__(self, dataset):
        self.train_data = join(self.data_dir, dataset+'_train.csv')
        self.test_data = join(self.data_dir, dataset+'_test.csv')

    def file_exists(self, filename, directory=data_dir):
        return any(filename == (self.data_dir+name) for name in listdir(directory))

    def label_creator(self):
        label_file = self.train_data.replace("_train.csv", "_labels.pickle")
        if not self.file_exists(label_file):
            self.create_label_onehots(self.train_data)
        return label_file

    def create_label_onehots(self, source_dir):
        labels = pd.read_csv(source_dir, encoding="ISO-8859-1")['Label'].unique()
        label_vectors = {}

        for i in range(len(labels)):
            vector = np.zeros(len(labels))
            vector[i] = 1.
            label_vectors[labels[i]] = vector
        self.pickle_save(label_vectors, source_dir.replace("_train.csv", "") + "_labels.pickle")
        return

    @staticmethod
    def pickle_save(data, destination_dir):
        handle = open(destination_dir, 'wb')
        pickle.dump(data, handle)
        return

    def simple_test(self, batch_size):
        nb_classifier = MultinomialNB()
        svm = SGDClassifier(max_iter=None)
        tfidf = TfIDfClassifier(self.train_data, with_n_grams=True, n_gram_range=(1,3), t_max_df=0.35, nb_features=4096)

        # training data
        train_dump = pd.read_csv(self.train_data, encoding="ISO-8859-1")
        train_labels = np.array(train_dump['Label'].as_matrix())
        tfidf.create_vectors(train_dump)
        train_vectors = np.array(tfidf.get_vectors(train_dump).toarray())

        # testing_data
        test_dump = pd.read_csv(self.test_data, encoding="ISO-8859-1")
        test_labels = np.array(test_dump['Label'].as_matrix())
        test_vectors = np.array(tfidf.get_vectors(test_dump).toarray())

        #training in batches
        num_splits = len(train_vectors) / batch_size
        rand_split = np.arange(len(train_vectors))
        np.random.shuffle(rand_split)
        arr_splits = np.array_split(rand_split, num_splits)

        # trainig
        for batch_no in arr_splits:
            sample_batch = train_vectors[batch_no]
            label_batch = train_labels[batch_no]
            nb_classifier.partial_fit(sample_batch, label_batch, classes=np.unique(train_labels))
            svm.partial_fit(sample_batch, label_batch, classes=np.unique(train_labels))

        #testing
        nb_correct_pred = 0
        svm_correct_pred = 0
        for idx, text in enumerate(test_vectors):
            text = np.expand_dims(text, axis=0)
            nb_pred_label = nb_classifier.predict(text)
            svm_pred_label = svm.predict(text)
            if nb_pred_label == test_labels[idx]:
                nb_correct_pred += 1
            if svm_pred_label == test_labels[idx]:
                svm_correct_pred += 1
        nb_acc = nb_correct_pred / len(test_vectors)
        svm_acc = svm_correct_pred / len(test_vectors)
        return nb_acc, svm_acc

if __name__ == '__main__':
    simple_classifiers = Classifiers('Reuters')
    nb_acc, svm_acc = simple_classifiers.simple_test(1024)
    print('Naive Bayes:', nb_acc, '\nSupport Vector Machine (SGD):', svm_acc)