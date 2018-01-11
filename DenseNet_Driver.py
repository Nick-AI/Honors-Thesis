import DenseNet
import DenseNet2D
import pickle
import re
import time
import json
import pydot
import graphviz
import gensim
import pandas as pd
import numpy as np
import keras.backend as K
from os import listdir
from os.path import join
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


class TfIDfClassifier:

    NGRAM_RANGE = (1, 3)
    MAX_DF = 0.4
    MAX_FEATURES = 2048
    dataset = ''
    DATA_DIR = './data/'
    VECTOR_FILE = ''

    def __init__(self, dataset):
        self.dataset = dataset
        self.vector_file = self.data_dir + "features_" + self.dataset + ".pkl"

    def create_vectors(self, train_data):
        """Creates tfidf matrix from training data
        :param train_data: string
        """
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.MAX_DF, ngram_range=self.NGRAM_RANGE,
                                         max_features=self.MAX_FEATURES)
        vectorizer.fit_transform(train_data['Text'].tolist())
        pickle.dump(vectorizer.vocabulary_, open(self.VECTOR_FILE, "wb"))
        return

    def get_vectors(self, test_data):
        """Creates vectors for given query
        :param test_data: string
        """
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=self.MAX_DF, ngram_range=self.NGRAM_RANGE,
                                     vocabulary=pickle.load(open(self.VECTOR_FILE, "rb")))
        transformer = TfidfTransformer()
        vectors = transformer.fit_transform(vectorizer.fit_transform(test_data['Text'].tolist()))
        return vectors

class Classifier:

    data_dir = './data/'
    train_data = ""
    test_data = ""
    with_plot = False
    NB_EPOCHS = 200
    BATCH_SIZE = 512
    LEARNING_RATE = 0.1
    EMBEDDING_FILE = data_dir + "W2V300_GNews.bin"
    EMBEDDINGS = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    def __init__(self, dataset, with_plot):
        self.dataset = dataset
        self.train_data = join(self.data_dir, dataset+'_train.csv')
        self.test_data = join(self.data_dir, dataset+'_test.csv')
        self.with_plot = with_plot

    @staticmethod
    def clean_text(text):
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        return " ".join(text.split()).lower()

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

    def get_embedding(self, text):
        embeddings = self.EMBEDDINGS.wv
        text = self.clean_text(text)
        text_embedding = []
        for word in text.split(' '):
            if word in embeddings:
                text_embedding.append(embeddings[word])
            elif word not in embeddings and word.isalpha():
                if word == 'a':
                    text_embedding.append(embeddings['A'])
                else:
                    word_split = list(word)
                    word_split_vec = [embeddings[elem] for elem in word_split
                                      if elem in embeddings]
                    word_vec = [sum(i) for i in zip(*word_split_vec)]
                    word_vec = np.array((np.array(word_vec) / len(word_vec)).tolist())
                    text_embedding.append(word_vec)
            else:  # if numbers or other symbols
                text_embedding.append([float(0)] * 300)
        return np.average(np.array(text_embedding), axis=0)

    def get_embedding_matrix(self, text):
        embeddings = self.EMBEDDINGS.wv
        text = self.clean_text(text)
        text = text.split(' ')
        text_embedding = []
        factor = 10
        if len(text) < 10:
            factor = len(text)
        len_substring = int(len(text)/factor)
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
        while len(text_embedding) < 10:
            text_embedding.append([float(0)] * 300)
        return text_embedding

    def run_matrix_embedding_classifier(self, t_depth=13, t_filters=6, t_growth_rate=12, in_dim=(10, 300),
                                        t_dropout=0.2):
        labels = self.label_creator()
        with open(labels, 'rb') as l_file:
            label_onehots = pickle.load(l_file)

        # training data
        train_dump = pd.read_csv(self.train_data, encoding="ISO-8859-1")
        train_labels = train_dump['Label'].as_matrix()
        train_vectors = []
        for item in train_dump['Text']:
            train_vectors.append(np.array(self.get_embedding_matrix(item)))
        train_labels = np.array([label_onehots[item]for item in train_labels])

        # testing_data
        test_dump = pd.read_csv(self.test_data, encoding="ISO-8859-1")
        test_labels = test_dump['Label'].as_matrix()
        test_vectors = []
        for item in test_dump['Text']:
            test_vectors.append(np.array(self.get_embedding_matrix(item)))
        test_labels = np.array([label_onehots[item] for item in test_labels])

        # create model
        network = DenseNet2D.DenseNet(nb_classes=len(label_onehots), img_dim=in_dim, depth=t_depth, nb_dense_block=4,
                                      growth_rate=t_growth_rate, nb_filter=t_filters, dropout_rate=t_dropout)

        # Model output
        network.summary()

        # Optimizer& intial learning rate
        learning_rate = self.LEARNING_RATE
        op = Adam(lr=learning_rate)

        network.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
        if self.with_plot:
            plot_model(network, to_file=self.data_dir + 'figures/densenet_EMatrix.png', show_shapes=True)

        # Train network
        print("Training")

        list_train_loss = []
        list_test_loss = []
        list_learning_rate = []

        for ep in range(self.NB_EPOCHS):

            if ep == int(0.5 * self.NB_EPOCHS):
                K.set_value(network.optimizer.lr, np.float32(learning_rate / 10))
                print('Learning rate changed to', learning_rate / 10)

            if ep == int(0.75 * self.NB_EPOCHS):
                K.set_value(network.optimizer.lr, np.float32(learning_rate / 100))
                print('Learning rate changed to', learning_rate / 100)

            # Create training arrays for each batch
            num_splits = len(train_vectors) / self.BATCH_SIZE
            rand_split = np.arange(len(train_vectors))
            np.random.shuffle(rand_split)
            arr_splits = np.array_split(rand_split, num_splits)

            l_train_loss = []
            start = time.time()

            # trainig
            for batch_no in arr_splits:
                sample_batch = []
                for i in batch_no:
                    sample_batch.append(train_vectors[i])
                sample_batch = np.expand_dims(sample_batch, axis=3)
                label_batch = train_labels[batch_no]
                train_logloss, train_acc = network.train_on_batch(sample_batch, label_batch)

                l_train_loss.append([train_logloss, train_acc])

            # testing
            valid_vectors = np.expand_dims(test_vectors, axis=3)
            test_logloss, test_acc = network.evaluate(valid_vectors, test_labels, verbose=0, batch_size=128)
            list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
            list_test_loss.append([test_logloss, test_acc])
            list_learning_rate.append(float(K.get_value(network.optimizer.lr)))

            print('Epoch %s/%s, Time: %s' % (ep + 1, self.NB_EPOCHS, time.time() - start))

            if (ep + 1) % 5 == 0:
                print('Train Loss: %s\nTest Loss: %s' % (train_logloss, test_logloss))
                print('Train Accuracy: %s\nTest Accuracy: %s' % (train_acc, test_acc))

            d_log = {}
            d_log['batch_size'] = self.BATCH_SIZE
            d_log['nb_epoch'] = ep
            d_log['optimizer'] = op.get_config()
            d_log['train_loss'] = list_train_loss
            d_log['test_loss'] = list_test_loss
            d_log['learning_rate'] = list_learning_rate

            json_file = join(self.data_dir, 'log_', self.dataset, '_withEMatrix.json')
            with open(json_file, 'w') as log_file:
                json.dump(d_log, log_file, indent=4, sort_keys=True)

    def run_1D_embedding_classifier(self, t_depth=22, t_filters=12, t_growth_rate=16, in_dim=300, t_dropout=0.2):
        labels = self.label_creator()
        with open(labels, 'rb') as l_file:
            label_onehots = pickle.load(l_file)

        # training data
        train_dump = pd.read_csv(self.train_data, encoding="ISO-8859-1")
        train_labels = train_dump['Label'].as_matrix()
        train_vectors = []
        for item in train_dump['Text']:
            train_vectors.append(self.get_embedding(item))
        train_vectors = np.array(train_vectors)
        train_labels = np.array([label_onehots[item]for item in train_labels])

        # testing_data
        test_dump = pd.read_csv(self.test_data, encoding="ISO-8859-1")
        test_labels = test_dump['Label'].as_matrix()
        test_vectors = []
        for item in test_dump['Text']:
            test_vectors.append(self.get_embedding(item))
        test_vectors = np.array(test_vectors)
        test_labels = np.array([label_onehots[item] for item in test_labels])

        # create model
        network = DenseNet.DenseNet(nb_classes=len(label_onehots), img_dim=in_dim, depth=t_depth, nb_dense_block=4,
                                    growth_rate=t_growth_rate, nb_filter=t_filters, dropout_rate=t_dropout)

        # Model output
        network.summary()

        # Optimizer& intial learning rate
        learning_rate = self.LEARNING_RATE
        op = Adam(lr=learning_rate)

        network.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
        if self.with_plot:
            plot_model(network, to_file=self.data_dir + 'figures/densenet_embedding.png', show_shapes=True)

        # Train network
        print("Training")

        list_train_loss = []
        list_test_loss = []
        list_learning_rate = []

        for ep in range(self.NB_EPOCHS):

            if ep == int(0.5 * self.NB_EPOCHS):
                K.set_value(network.optimizer.lr, np.float32(learning_rate / 10))
                print('Learning rate changed to', learning_rate / 10)

            if ep == int(0.75 * self.NB_EPOCHS):
                K.set_value(network.optimizer.lr, np.float32(learning_rate / 100))
                print('Learning rate changed to', learning_rate / 100)

            # Create training arrays for each batch
            num_splits = len(train_vectors) / self.BATCH_SIZE
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

            print('Epoch %s/%s, Time: %s' % (ep + 1, self.NB_EPOCHS, time.time() - start))

            if (ep + 1) % 5 == 0:
                print('Train Loss: %s\nTest Loss: %s' % (train_logloss, test_logloss))
                print('Train Accuracy: %s\nTest Accuracy: %s' % (train_acc, test_acc))

            d_log = {}
            d_log['batch_size'] = self.BATCH_SIZE
            d_log['nb_epoch'] = ep
            d_log['optimizer'] = op.get_config()
            d_log['train_loss'] = list_train_loss
            d_log['test_loss'] = list_test_loss
            d_log['learning_rate'] = list_learning_rate

            json_file = join(self.data_dir, 'log_', self.dataset, '_withEmbedding.json')
            with open(json_file, 'w') as log_file:
                json.dump(d_log, log_file, indent=4, sort_keys=True)

    def run_classifier_tfidf(self, t_depth=22, t_filters=12, t_growth_rate=16, t_tfidf_dim=2048, t_batchSize=512,
                             t_dropout=0.2):
        tfidf = TfIDfClassifier(self.dataset)
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


        #create model
        network = DenseNet.DenseNet(nb_classes=len(label_onehots), img_dim=t_tfidf_dim, depth=t_depth,
                                        nb_dense_block=4, growth_rate=t_growth_rate, nb_filter=t_filters,
                                        dropout_rate=t_dropout)

        # Model output
        network.summary()

        # Optimizer& intial learning rate
        learning_rate = self.LEARNING_RATE
        op = Adam(lr=learning_rate)

        network.compile(optimizer=op ,loss='categorical_crossentropy', metrics=['accuracy'])
        if self.with_plot:
            plot_model(network, to_file=self.data_dir+'figures/densenet_tfidf.png', show_shapes=True)

        # Train network
        print("Training")

        list_train_loss = []
        list_test_loss = []
        list_learning_rate = []

        for ep in range(self.NB_EPOCHS):

            if ep == int(0.5*self.NB_EPOCHS):
                K.set_value(network.optimizer.lr, np.float32(learning_rate/10))
                print('Learning rate changed to', learning_rate/10)

            if ep == int(0.75*self.NB_EPOCHS):
                K.set_value(network.optimizer.lr, np.float32(learning_rate/100))
                print('Learning rate changed to', learning_rate/100)

            # Create training arrays for each batch
            num_splits = len(train_vectors) / self.BATCH_SIZE
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

            print('Epoch %s/%s, Time: %s' % (ep+1, self.NB_EPOCHS, time.time()-start))

            if (ep+1) % 5 == 0:
                print('Train Loss: %s\nTest Loss: %s' % (train_logloss, test_logloss))
                print('Train Accuracy: %s\nTest Accuracy: %s' % (train_acc, test_acc))

            d_log = {}
            d_log['batch_size'] = self.BATCH_SIZE
            d_log['nb_epoch'] = ep
            d_log['optimizer'] = op.get_config()
            d_log['train_loss'] = list_train_loss
            d_log['test_loss'] = list_test_loss
            d_log['learning_rate'] = list_learning_rate

            json_file = join(self.data_dir, 'log_', self.dataset, '_withEmbedding.json')
            with open(json_file, 'w') as log_file:
                json.dump(d_log, log_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    for dataset in ['Retuers', 'SOData']:
        denseNet = Classifier(dataset, True)
        denseNet.run_matrix_embedding_classifier()
        denseNet.run_1D_embedding_classifier()
        denseNet.run_classifier_tfidf()