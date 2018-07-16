import pickle
import time
import json
from tqdm import tqdm
import Simple_RCNN
import pandas as pd
import numpy as np
import keras.backend as K
from os import listdir
from os.path import join
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
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
    vector_file = ''

    def __init__(self, dataset, with_n_grams, n_gram_range, t_max_df, nb_features):
        self.dataset = dataset
        self.uses_ngrams = with_n_grams
        self.n_gram_range = n_gram_range
        self.t_max_df = t_max_df
        self.nb_features = nb_features
        self.vector_file = self.data_dir + "features_" + self.dataset + ".pkl"

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
    dataset = ''
    directory = "./"
    data_dir = join(directory, "data/")
    train_data = ""
    test_data = ""
    nb_epochs = 100
    batch_size = 512

    def __init__(self, dataset):
        self.dataset = dataset
        self.train_data = join(self.data_dir, dataset+'_train.csv')
        self.test_data = join(self.data_dir, dataset+'_test.csv')

    def file_exists(self, filename, directory=data_dir):
        return any(filename == (self.data_dir+name) for name in listdir(directory))

    @staticmethod
    def pickle_save(data, destination_dir):
        handle = open(destination_dir, 'wb')
        pickle.dump(data, handle)
        return

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

    def simple_test(self, batch_size):
        nb_classifier = MultinomialNB()
        svm = SGDClassifier(max_iter=None)
        nn = Simple_RCNN
        tfidf = TfIDfClassifier(self.dataset, with_n_grams=True, n_gram_range=(1,3), t_max_df=0.35, nb_features=1024)

        labels = self.label_creator()
        with open(labels, 'rb') as l_file:
            label_onehots = pickle.load(l_file)

        # training data
        train_dump = pd.read_csv(self.train_data, encoding="ISO-8859-1")
        train_labels = np.array(train_dump['Label'].as_matrix())
        nn_train_labels = np.array([label_onehots[item] for item in train_labels])
        tfidf.create_vectors(train_dump)
        train_vectors = np.array(tfidf.get_vectors(train_dump).toarray())

        # testing_data
        test_dump = pd.read_csv(self.test_data, encoding="ISO-8859-1")
        test_labels = np.array(test_dump['Label'].as_matrix())
        nn_test_labels = np.array([label_onehots[item] for item in test_labels])
        test_vectors = np.array(tfidf.get_vectors(test_dump).toarray())

        #initiate NN model
        nn_classifier = nn.nn_classifier(nb_classes=len(np.unique(test_labels)), in_dimensions=(1024, 1, ), two_d=False)
        nn_classifier.summary()

        #optimizer
        learning_rate = 0.01
        op = Adam(lr=learning_rate)

        nn_classifier.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
        plot_model(nn_classifier, to_file=self.data_dir + 'figures/RCNN_architecture.png', show_shapes=True)
        list_train_loss = []
        list_test_loss = []
        list_learning_rate = []

        # for ep in range(self.nb_epochs):
        #
        #     if ep == int(0.5 * self.nb_epochs):
        #         K.set_value(nn_classifier.optimizer.lr, np.float32(learning_rate / 10))
        #         print('Learning rate changed to', learning_rate / 10)
        #
        #     if ep == int(0.75 * self.nb_epochs):
        #         K.set_value(nn_classifier.optimizer.lr, np.float32(learning_rate / 100))
        #         print('Learning rate changed to', learning_rate / 100)
        #
        #     # Create training arrays for each batch
        #     num_splits = len(train_vectors) / self.batch_size
        #     rand_split = np.arange(len(train_vectors))
        #     np.random.shuffle(rand_split)
        #     arr_splits = np.array_split(rand_split, num_splits)
        #
        #     l_train_loss = []
        #     start = time.time()
        #
        #     # trainig
        #     for batch_no in tqdm(arr_splits):
        #         sample_batch = []
        #         for i in batch_no:
        #             sample_batch.append(train_vectors[i])
        #         sample_batch = np.expand_dims(sample_batch, axis=3)
        #         label_batch = nn_train_labels[batch_no]
        #         train_logloss, train_acc = nn_classifier.train_on_batch(sample_batch, label_batch)
        #
        #         l_train_loss.append([train_logloss, train_acc])
        #
        #     # testing
        #     valid_vectors = np.expand_dims(test_vectors, axis=3)
        #     test_logloss, test_acc = nn_classifier.evaluate(valid_vectors, nn_test_labels, verbose=0, batch_size=128)
        #     list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        #     list_test_loss.append([test_logloss, test_acc])
        #     list_learning_rate.append(float(K.get_value(nn_classifier.optimizer.lr)))
        #
        #     print('Epoch %s/%s, Time: %s' % (ep + 1, self.nb_epochs, time.time() - start))
        #
        #     if (ep + 1) % 5 == 0:
        #         print('Train Loss: %s\nTest Loss: %s' % (train_logloss, test_logloss))
        #         print('Train Accuracy: %s\nTest Accuracy: %s' % (train_acc, test_acc))
        #
        #     d_log = {}
        #     d_log['batch_size'] = self.batch_size
        #     d_log['nb_epochs'] = self.nb_epochs
        #     d_log['optimizer'] = op.get_config()
        #     d_log['train_loss'] = list_train_loss
        #     d_log['test_loss'] = list_test_loss
        #     d_log['learning_rate'] = list_learning_rate
        #
        #     json_file = join(self.data_dir, 'log_' + self.dataset + '_simple_nn.json')
        #     with open(json_file, 'w') as log_file:
        #         json.dump(d_log, log_file, indent=4, sort_keys=True)
        #

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
    datasets = ['Reuters', 'SOData']
    for item in datasets:
        simple_classifiers = Classifiers(item)
        nb_acc, svm_acc = simple_classifiers.simple_test(512)
        print('Dataset:', item, '\nNaive Bayes:', nb_acc, '\nSupport Vector Machine (SGD):', svm_acc)
