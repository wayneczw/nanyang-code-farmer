import calendar
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import regex as re
import tensorflow as tf
import tensorflow_hub as hub
import warnings
import json

from argparse import ArgumentParser

from category_encoders.target_encoder import TargetEncoder

from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine import Layer
from keras.layers import Bidirectional
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import GlobalMaxPool1D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import SpatialDropout1D
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing import text

from nltk import sent_tokenize
from textblob import TextBlob

# from sklearn.base import BaseEstimator
# from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger(__name__)


raw_categorical_targets = [
    'Pattern', 'Collar Type',
    'Fashion Trend', 'Clothing Material',
    'Sleeves']

categorical_targets = [
    'Pattern', 'Collar Type',
    'Fashion Trend', 'Clothing Material',
    'Sleeves']

USE_MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
USE_EMBED = hub.Module(USE_MODULE_URL, trainable=True)
# ELMO_MODULE_URL = how"https://tfhub.dev/google/elmo/2"
# ELMO_EMBED = hub.Module(ELMO_MODULE_URL, trainable=True)


def USE_Embedding(x):
    return USE_EMBED(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
#end def


# def ELMO_Embedding(x):
#     return ELMO_EMBED(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
# #end def


def print_header(text, width=30, char='='):
    print('\n' + char * width)
    print(text)
    print(char * width)
#end def


def display_null_percentage(data):
    print_header('Percentage of Nulls')
    df = data.isnull().sum().reset_index().rename(columns={0: 'Count', 'index': 'Column'})
    df['Frequency'] = df['Count'] / data.shape[0] * 100
    pd.options.display.float_format = '{:.2f}%'.format
    print(df)
    pd.options.display.float_format = None
#end def


def display_category_counts(data, categorical_features):
    print_header('Category Counts for Categorical Features')
    for categorical_feature in categorical_features:
        print('-' * 30)
        print(categorical_feature)
        print(data[categorical_feature].value_counts(dropna=False))
#end def


def analyse(df, categorical_targets):
    print(df.info())
    print('='*100)

    print(df.head())
    print('='*100)

    print(df.describe())
    print('='*100)

    print(df.nunique())
    print('='*100)

    display_null_percentage(df)  # No missing hahaha :) :) :)
    print('='*100)

    display_category_counts(data=df, categorical_features=categorical_targets)
#end def


def plot_history(h, path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 2)
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig(path)
#end def


def read(df_path, mapping_path, quick=False):
    def get_polarity(text):
        textblob = TextBlob(text)
        pol = textblob.sentiment.polarity
        return round(pol, 2)
    #end def

    def get_subjectivity(text):
        textblob = TextBlob(text)
        subj = textblob.sentiment.subjectivity
        return round(subj, 2)
    #end def

    def clean(text):
        return re.sub('[!@#$:]', '', ' '.join(re.findall('\w{3,}', str(text).lower())))
    #end def

    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)
    if quick: df = df[:1024]

    df[raw_categorical_targets] = df[raw_categorical_targets].fillna('-1')
    
    with open(mapping_path, 'r') as f:
        mapping_dict = json.load(f)
    #end with

    for y in mapping_dict.keys():
        y_dict = mapping_dict[y]
        mapping_dict[y] = dict((v,k) for k,v in y_dict.items())
    #end for

    for y in mapping_dict.keys():
        df[y] = df[y].astype('int32')
        df[y] = df[y].map(mapping_dict[y], na_action='ignore')
    #end for

    df[raw_categorical_targets] = df[raw_categorical_targets].fillna('unk')
    logger.info("Done reading in {} data....".format(df.shape[0]))

    for y in mapping_dict.keys():
        y_dict = mapping_dict[y]
        mapping_dict[y] = dict((v,k) for k,v in y_dict.items())
    #end for

    return df, mapping_dict
#end def


def auc(y_true, y_pred):
    def _binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
        # PFA, prob false alert for binary classifier
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(1 - y_true)
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_pred - y_pred * y_true)    
        return FP/N
    #end def

    def _binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        # P_TA prob true alerts for binary classifier
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true)
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)    
        return TP/P
    #end def

    # AUC
    ptas = tf.stack([_binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([_binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1, )), pfas], axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes

    return K.sum(s, axis=0)
#end def


def text_tokenize(train, test, col_name, num_words, maxlen, embeddings_index, embed_size):
    logger.info('Computing in embedding matrixes....')
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train[col_name].tolist() + test[col_name].tolist())
    train_tokenized = tokenizer.texts_to_sequences(train[col_name].tolist())
    test_tokenized = tokenizer.texts_to_sequences(test[col_name].tolist())
    X_train = sequence.pad_sequences(train_tokenized, maxlen=maxlen)
    X_test = sequence.pad_sequences(test_tokenized, maxlen=maxlen)

    word_index = tokenizer.word_index
    #prepare embedding matrix
    num_words = min(num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    logger.info('Done computing in embedding matrixes....')

    return X_train, X_test, embedding_matrix
#end def


def build_model(
    title_input_shape, output_shape,
    dropout_rate=0, kernel_regularizer=0,
    activity_regularizer=0, bias_regularizer=0):
    
    title_input = Input(title_input_shape, name='title_input')
    title = Dense(2056)(title_input)
    title = Dropout(dropout_rate)(title)
    title = Dense(512)(title)
    title = Dropout(dropout_rate)(title)

    output = Dense(output_shape, activation='softmax', name='output')(title)

    model = Model(inputs=[title_input], outputs=[output])

    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', auc])

    return model
#end def


def batch_iter(
    X_title, y,
    batch_size=128, **kwargs):

    data_size = y.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        while True:
            # Shuffle the data at each epoch
            shuffled_indices = np.random.permutation(np.arange(data_size, dtype=np.int))

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, data_size)

                X_title_batch = [X_title[i] for i in shuffled_indices[start_index:end_index]]
                y_batch = [y[i] for i in shuffled_indices[start_index:end_index]]

                yield (
                    {'title_input': np.asarray(X_title_batch)},
                    {'output': np.asarray(y_batch)})
            #end for
        #end while
    #end def

    return num_batches, _data_generator()
#end def


def train(
    model,
    X_title_train, y_train,
    X_title_val=None, y_val=None,
    weights_path='./weights/', weights_prefix='',
    class_weight=None, batch_size=128, epochs=32, **kwargs):

    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    train_steps, train_batches = batch_iter(
        X_title_train, y_train,
        batch_size=batch_size)

    if y_val is not None:
        val_steps, val_batches = batch_iter(
            X_title_val, y_val,
            batch_size=batch_size)

    # define early stopping callback
    callbacks_list = []
    if y_val is not None:
        early_stopping = dict(monitor='val_loss', patience=3, min_delta=0.0001, verbose=1)
        model_checkpoint = dict(filepath=weights_path + weights_prefix + '_{val_loss:.5f}_{loss:.5f}_{epoch:04d}.weights.h5',
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1,
                                verbose=1)
    else:
        early_stopping = dict(monitor='loss', patience=1, min_delta=0.001, verbose=1)
        model_checkpoint = dict(filepath=weights_path + weights_prefix + '_{loss:.5f}_{epoch:04d}.weights.h5',
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1,
                                verbose=1)

    earlystop = EarlyStopping(**early_stopping)
    callbacks_list.append(earlystop)

    checkpoint = ModelCheckpoint(**model_checkpoint)
    callbacks_list.append(checkpoint)

    if y_val is not None:
        model.fit_generator(
            epochs=epochs,
            generator=train_batches,
            steps_per_epoch=train_steps,
            validation_data=val_batches,
            validation_steps=val_steps,
            callbacks=callbacks_list,
            class_weight=class_weight)
    else:
        model.fit_generator(
            epochs=epochs,
            generator=train_batches,
            steps_per_epoch=train_steps,
            callbacks=callbacks_list,
            class_weight=class_weight)

    return model
#end def


def predict_iter(
    X_title,
    batch_size=128, **kwargs):

    data_size = X_title.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)

            X_title_batch = [x for x in X_title[start_index:end_index]]
            yield ({'title_input': np.asarray(X_title_batch)})
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def test(
    model,
    X_title_test,
    lb, mapping,
    batch_size=128, **kwargs):

    def _second_largest(numbers):
        count = 0
        m1 = m2 = float('-inf')
        for x in numbers:
            count += 1
            if x > m2:
                if x >= m1:
                    m1, m2 = x, m1
                else:
                    m2 = x
        return m2 if count >= 2 else None
    #end def

    def _larger(a, b):
        return True if (a >= b) else False
    #end def

    test_steps, test_batches = predict_iter(
        X_title_test,
        batch_size=batch_size, **kwargs)

    proba = model.predict_generator(generator=test_batches, steps=test_steps)

    second_largest_proba = [_second_largest(p) for p in proba]

    pred = [[str(mapping[lb.classes_[_i]]) for _i, _p in enumerate(p) if _larger(_p, second_largest_proba[i])] for i, p in enumerate(proba)]

    return [' '.join(p[:2]) for p in pred]
#end def


def main():
    parser = ArgumentParser(description='Run machine learning experiment.')
    parser.add_argument('-i', '--train', type=str, metavar='<train_set>', required=True, help='Training data set.')
    parser.add_argument('-t', '--test', type=str, metavar='<test_set>', required=True, help='Test data set.')
    parser.add_argument('-m', '--mapping', type=str, metavar='<mapping_json>', required=True, help='Mapping json file.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    A = parser.parse_args()

    log_level = 'DEBUG'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    # set seed
    np.random.seed(A.seed)
    random.seed(A.seed)
    tf.set_random_seed(A.seed)

    ######### old ##################
    quick = False
    validate = True
    batch_size = 256
    epochs = 16
    threshold = 10 # Anything that occurs less than this will be removed.

    # read in data
    train_df, mapping_dict = read(A.train, A.mapping, quick=quick)
    # analyse(train_df, categorical_targets)
    # input()

    if validate:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=A.seed)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    lb_dict = dict()
    train_dict = dict(batch_size=batch_size, epochs=epochs)

    for y in categorical_targets:
        lb_dict[y] = LabelBinarizer()
        train_dict['X_' + y + '_train_index'] = list(train_df.loc[train_df[y] != 'unk'].index.values)
        train_dict['y_' + y + '_train'] = lb_dict[y].fit_transform(train_df[y][train_dict['X_' + y + '_train_index']])
        if validate:
            train_dict['X_' + y + '_val_index'] = list(val_df.loc[val_df[y] != 'unk'].index.values)
            train_dict['y_' + y + '_val'] = lb_dict[y].transform(val_df[y][train_dict['X_' + y + '_val_index']])
    #end for

    vec = TfidfVectorizer(
        max_features=8192,
        sublinear_tf=True,
        strip_accents='unicode',
        stop_words='english',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 4),
        dtype=np.float32,
        norm='l2',
        min_df=5,
        max_df=.9)
    train_dict['X_title_train_raw'] = vec.fit_transform(train_df['title']).toarray()
    if validate:
        train_dict['X_title_val_raw'] = vec.transform(val_df['title']).toarray()

    test_df = pd.read_csv(A.test)
    test_dict = dict(batch_size=batch_size, epochs=epochs)
    test_dict['X_title_test'] = vec.transform(test_df['title']).toarray()

    for y in categorical_targets:
        train_dict['X_title_train'] = train_dict['X_title_train_raw'][train_dict['X_' + y + '_train_index']]
        if validate:
            train_dict['X_title_val'] = train_dict['X_title_val_raw'][train_dict['X_' + y + '_val_index']]

        model = build_model(
            title_input_shape=train_dict['X_title_train'].shape[1:],
            output_shape=train_dict['y_' + y + '_train'].shape[1])
        print(model.summary())

        train_dict['model'] = model
        train_dict['y_train'] = train_dict['y_' + y + '_train']
        train_dict['y_val'] = train_dict['y_' + y + '_val']
        train_dict['weights_prefix'] = y
        model = train(**train_dict)

        test_dict['model'] = model
        test_dict['lb'] = lb_dict[y]
        test_dict['mapping'] = mapping_dict[y]
        test_df[y] = test(**test_dict)
    #end for

    test_df.to_csv('./data/output/fashion_test_proba.csv', index=False)
#end def


if __name__ == '__main__': main()
