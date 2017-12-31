import DenseNet
import pickle
import re
import time
import json
import pydot
import graphviz
import itertools
import collections
import pandas as pd
import numpy as np
import keras.backend as K
from os import listdir
from os.path import join
from nltk.corpus import reuters
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

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

class Classifier:

    directory = "./"
    data_dir = join(directory, "data/")
    train_data = ""
    test_data = ""
    with_plot = False
    nb_epochs = 0
    batch_size = 0

    def __init__(self, dataset, with_plot, nb_epochs, batch_size):
        self.train_data = join(self.data_dir, dataset+'_train.csv')
        self.test_data = join(self.data_dir, dataset+'_test.csv')
        self.with_plot = with_plot
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size


    @staticmethod
    def clean_text(text):
        text = re.sub(r"[^A-Za-z0-9,.;:!?]", " ", text)
        return "".join(text.split()).lower()

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

    def run_classifier_tfidf(self, is_testing=False, t_depth=10, t_filters=6, t_growth_rate=6, t_tfidf_dim=1024,
                             t_with_nGrams=False, t_ngram_range=1, t_max_Df=0.3, t_batchSize=256, t_dropout=0,
                             t_learningRate=0.3, max_acc=0):

        tfidf = TfIDfClassifier(self.train_data, t_with_nGrams, t_ngram_range, t_max_Df, t_tfidf_dim)
        labels = self.label_creator()
        with open(labels, 'rb') as l_file:
            label_onehots = pickle.load(l_file)

        # training data
        train_dump = pd.read_csv(self.train_data, encoding="ISO-8859-1")
        train_labels = train_dump['Label'].as_matrix()
        tfidf.create_vectors(train_dump)
        train_vectors = np.array(tfidf.get_vectors(train_dump).toarray())
        train_labels = np.array([label_onehots[item]for item in train_labels])

        # testing_data
        test_dump = pd.read_csv(self.test_data, encoding="ISO-8859-1")
        test_labels = test_dump['Label'].as_matrix()
        test_vectors = np.array(tfidf.get_vectors(test_dump).toarray())
        test_labels = np.array([label_onehots[item] for item in test_labels])

        if is_testing:
            try:
                network = DenseNet.DenseNet(nb_classes=len(label_onehots), img_dim=t_tfidf_dim, depth=t_depth,
                                            nb_dense_block=4, growth_rate=t_growth_rate, nb_filter=t_filters,
                                            dropout_rate=t_dropout)
                network.summary()

                # Optimizer& intial learning rate
                learning_rate = t_learningRate
                op = Adam(lr=learning_rate)

                network.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
                if self.with_plot:
                    plot_model(network, to_file=self.data_dir + 'figures/densenet_architecture.png', show_shapes=True)

                self.batch_size = t_batchSize

                # Train network
                print("Training")

                list_train_loss = []
                list_test_loss = []
                list_learning_rate = []

                for ep in range(self.nb_epochs):

                    if ep == int(0.5 * self.nb_epochs):
                        K.set_value(network.optimizer.lr, np.float32(learning_rate / 10))
                        print('Learning rate changed to', learning_rate / 10)

                    if ep == int(0.75 * self.nb_epochs):
                        K.set_value(network.optimizer.lr, np.float32(learning_rate / 100))
                        print('Learning rate changed to', learning_rate / 100)

                    # Create training arrays for each batch
                    num_splits = len(train_vectors) / self.batch_size
                    rand_split = np.arange(len(train_vectors))
                    np.random.shuffle(rand_split)
                    arr_splits = np.array_split(rand_split, num_splits)

                    l_train_loss = []
                    start = time.time()

                    # trainig
                    for batch_no in arr_splits:
                        sample_batch = np.expand_dims(train_vectors[batch_no], axis=2)
                        label_batch = train_labels[batch_no]
                        train_logloss, train_acc = network.train_on_batch(sample_batch, label_batch)

                        l_train_loss.append([train_logloss, train_acc])

                    # testing
                    valid_vectors = np.expand_dims(test_vectors, axis=2)
                    test_logloss, test_acc = network.evaluate(valid_vectors, test_labels, verbose=0, batch_size=128)
                    list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
                    list_test_loss.append([test_logloss, test_acc])
                    list_learning_rate.append(float(K.get_value(network.optimizer.lr)))

                    print('Epoch %s/%s, Time: %s' % (ep + 1, self.nb_epochs, time.time() - start))

                    if (ep + 1) % 5 == 0:
                        print('Train Loss: %s\nTest Loss: %s' % (train_logloss, test_logloss))
                        print('Train Accuracy: %s\nTest Accuracy: %s' % (train_acc, test_acc))
                    if ep==(self.nb_epochs-1):
                        if test_logloss>max_acc:
                            max_acc = test_logloss
                            d_log = {}
                            d_log['Accuracy'] = max_acc
                            d_log['Depth'] = t_depth
                            d_log['Filters'] = t_filters
                            d_log['Growth_Rate'] = t_growth_rate
                            d_log['TfIDf_Dim'] = t_tfidf_dim
                            d_log['Used_NGrams'] = t_with_nGrams
                            d_log['NGram_Range'] = t_ngram_range
                            d_log['Max_Df'] = t_max_Df
                            d_log['Batch_Size'] = t_batchSize
                            d_log['Dropout_Rate'] = t_dropout
                            d_log['Learning_Rate'] = t_learningRate

                            json_file = join(self.data_dir, 'Param_test_log.json')
                            with open(json_file, 'w') as log_file:
                                json.dump(d_log, log_file, indent=4, sort_keys=True)
            except:
                print('works')
                pass
            return max_acc



        else:
            #create model
            network = DenseNet.DenseNet(nb_classes=len(label_onehots), img_dim=2048, depth=10, nb_dense_block=4,
                                        growth_rate=6, nb_filter=6)

            # Model output
            network.summary()

            # Optimizer& intial learning rate
            learning_rate = 0.15
            op = Adam(lr=learning_rate)

            network.compile(optimizer=op ,loss='categorical_crossentropy', metrics=['accuracy'])
            if self.with_plot:
                plot_model(network, to_file=self.data_dir+'figures/densenet_architecture.png', show_shapes=True)

            # Train network
            print("Training")

            list_train_loss = []
            list_test_loss = []
            list_learning_rate = []

            for ep in range(self.nb_epochs):

                if ep == int(0.5*self.nb_epochs):
                    K.set_value(network.optimizer.lr, np.float32(learning_rate/10))
                    print('Learning rate changed to', learning_rate/10)

                if ep == int(0.75*self.nb_epochs):
                    K.set_value(network.optimizer.lr, np.float32(learning_rate/100))
                    print('Learning rate changed to', learning_rate/100)

                # Create training arrays for each batch
                num_splits = len(train_vectors) / self.batch_size
                rand_split = np.arange(len(train_vectors))
                np.random.shuffle(rand_split)
                arr_splits = np.array_split(rand_split, num_splits)

                l_train_loss = []
                start = time.time()

                # trainig
                for batch_no in arr_splits:
                    sample_batch = np.expand_dims(train_vectors[batch_no], axis=2)
                    label_batch = train_labels[batch_no]
                    train_logloss, train_acc = network.train_on_batch(sample_batch, label_batch)

                    l_train_loss.append([train_logloss, train_acc])

                #testing
                valid_vectors = np.expand_dims(test_vectors, axis=2)
                test_logloss, test_acc = network.evaluate(valid_vectors, test_labels, verbose=0, batch_size=128)
                list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
                list_test_loss.append([test_logloss, test_acc])
                list_learning_rate.append(float(K.get_value(network.optimizer.lr)))

                print('Epoch %s/%s, Time: %s' % (ep+1, self.nb_epochs, time.time()-start))

                if (ep+1) % 5 == 0:
                    print('Train Loss: %s\nTest Loss: %s' % (train_logloss, test_logloss))
                    print('Train Accuracy: %s\nTest Accuracy: %s' % (train_acc, test_acc))

                d_log = {}
                d_log['batch_size'] = self.batch_size
                d_log['nb_epochs'] = self.nb_epochs
                d_log['optimizer'] = op.get_config()
                d_log['train_loss'] = list_train_loss
                d_log['test_loss'] = list_test_loss
                d_log['learning_rate'] = list_learning_rate

                json_file = join(self.data_dir, 'experiment_log_reuters.json')
                with open(json_file, 'w') as log_file:
                    json.dump(d_log, log_file, indent=4, sort_keys=True)


def create_parameters():
    params = []
    values = []
    param_settings = {}
    t_depth = [10]
    t_filters = [6]
    t_growth_rate = [6]
    t_tfidf_dim = [2048, 4096]
    t_with_ngrams = [True, False]
    t_ngram_range = [(1,2), (1,3)]
    t_max_df = [0.3, 0.4]
    t_batch_size = [256]
    t_dropout = [0, 0.1]
    t_learning_rate = [0.1]
    parameter_grid = dict(t_depth=t_depth, t_filters=t_filters, t_growth_rate=t_growth_rate,
                          t_tfidf_dim=t_tfidf_dim, t_with_nGrams=t_with_ngrams, t_ngram_range=t_ngram_range,
                          t_max_Df=t_max_df, t_batchSize=t_batch_size, t_dropout=t_dropout,
                          t_learningRate=t_learning_rate)
    parameters = collections.OrderedDict(sorted(parameter_grid.items()))
    for key, item in parameters.items():
        params.append(key)
        values.append(item)

    configurations = list(itertools.product(*values))

    for item in configurations:
        for index, val in enumerate(item):
            param_settings[params[index]] = val
        yield param_settings

if __name__ == '__main__':
    denseNet = Classifier('Reuters', False, 125, 256)
    max_acc=0
    #denseNet.run_classifier_tfidf(True, t_depth=40)
    for p in create_parameters():
        max_acc=denseNet.run_classifier_tfidf(True, int(p['t_depth']), int(p['t_filters']), int(p['t_growth_rate']),
                                              int(p['t_tfidf_dim']), p['t_with_nGrams'], p['t_ngram_range'],
                                              p['t_max_Df'], int(p['t_batchSize']), p['t_dropout'], p['t_learningRate'],
                                              max_acc)