import gc
import json
import logging
import numpy as np
import pandas as pd
import random
import tensorflow as tf

from argparse import ArgumentParser

from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


logger = logging.getLogger(__name__)

raw_categorical_targets = [
    'Benefits', 'Brand',
    'Colour_group', 'Product_texture',
    'Skin_type']

categorical_targets = [
    'Benefits', 'Brand',
    'Colour_group', 'Product_texture',
    'Skin_type']

categorical_features = [
    'nouns', 'numbers']

continuous_features = [
    "benefits_count", "high_count", "pigmentation_count", "natural_count", "light_count", "hydrating_count", "durable_count", "oil_count", "control_count", "spf_count", "colour_count", "group_count", "emas_count", "rose_count", "1 warna_count", "9 color_count", "hijau_count", "warna_count", "merah_count", "cabai_count", "warna merah cabai_count", "8 color_count", "5 color_count", "multiwarna_count", "perak_count", "krem_count", "peach_count", "coklat tua_count", "peanut_count", "biru_count", "ungu_count", "hitam_count", "abu_count", "6 color_count", "mawar_count", "4 color_count", "putih_count", "warna blush pink_count", "bening_count", "netral_count", "maroon_count", "kuning_count", "11 color_count", "warna koral_count", "3 color_count", "nude_count", "12 color_count", "ceri_count", "2 color_count", "warna fuchsia_count", "merah muda_count", "warna hotpink_count", "10 color_count", "merah semangka_count", "jeruk_count", "sawo matang_count", "7 color_count", "cokelat_count", "antique white_count", "product_count", "texture_count", "balm_count", "stick_count", "liquid_count", "crayon pensiln_count", "formula mousse_count", "cream_count", "solid_count", "powder_count", "solid powder_count", "cushion_count", "gel_count", "skin_count", "type_count", "dry_count", "sensitive_count", "fade_count", "combination_count", "normal_count", "aging_count", "age_count", "signs of aging_count", "acne_count", "greasy_count"]


def read(df_path, mapping_path=None, quick=False):
    def to_en(text, translator):
        return translator.translate(text)
    #end def

    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)
    if quick: df = df[:16384]

    if mapping_path is not None:
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

        for y in mapping_dict.keys():
            y_dict = mapping_dict[y]
            mapping_dict[y] = dict((v,k) for k,v in y_dict.items())
        #end for
    #end if

    df[categorical_features] = df[categorical_features].fillna('unk')
    df['translated'] = df['translated'].fillna('')
    df['ocr'] = df['ocr'].fillna('')

    logger.info("Done reading in {} data....".format(df.shape[0]))

    return (df, mapping_dict) if mapping_path is not None else df
#end def


def build_model(
    title_input_shape,
    # translated_input_shape,
    # ocr_input_shape,
    nouns_input_shape,
    numbers_input_shape,
    # cont_input_shape,
    output_shape,
    dropout_rate=0, kernel_regularizer=0,
    activity_regularizer=0, bias_regularizer=0):
    
    title_input = Input(title_input_shape, name='title_input')
    title = Dense(2048)(title_input)
    title = Dropout(dropout_rate)(title)
    title = Dense(min(1024, output_shape*4))(title)
    title = Dropout(dropout_rate)(title)

    # translated_input = Input(translated_input_shape, name='translated_input')
    # translated = Dense(512)(translated_input)
    # translated = Dropout(dropout_rate)(translated)
    # translated = Dense(min(256, output_shape*2))(translated)
    # translated = Dropout(dropout_rate)(translated)

    # ocr_input = Input(ocr_input_shape, name='ocr_input')
    # ocr = Dense(512)(ocr_input)
    # ocr = Dropout(dropout_rate)(ocr)
    # ocr = Dense(min(256, output_shape*4))(ocr)
    # ocr = Dropout(dropout_rate)(ocr)

    nouns_input = Input(nouns_input_shape, name='nouns_input')
    nouns = Dense(512)(nouns_input)
    nouns = Dropout(dropout_rate)(nouns)
    nouns = Dense(min(256, output_shape*2))(nouns_input)
    nouns = Dropout(dropout_rate)(nouns)

    numbers_input = Input(numbers_input_shape, name='numbers_input')
    numbers = Dense(512)(numbers_input)
    numbers = Dropout(dropout_rate)(numbers)
    numbers = Dense(min(256, output_shape*2))(numbers_input)
    numbers = Dropout(dropout_rate)(numbers)

    # cont_input = Input(cont_input_shape, name='cont_input')
    # cont = Dense(256)(cont_input)
    # cont = Dropout(dropout_rate)(cont)
    # cont = Dense(min(128, output_shape*2))(cont)
    # cont = Dropout(dropout_rate)(cont)

    # inputs = [title_input, translated_input, ocr_input, nouns_input, numbers_input, cont_input]
    inputs = [title_input, nouns_input, numbers_input]

    # x = concatenate([title, translated, ocr, nouns, numbers, cont])
    x = concatenate([title, nouns, numbers])
    # x = Dense(min(1024, output_shape*4))(x)
    # x = Dropout(dropout_rate)(x)
    # x = Dense(min(512, output_shape*2))(x)
    # x = Dropout(dropout_rate)(x)

    output = Dense(output_shape, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=[output])

    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
#end def


def batch_iter(
    X_title,
    # X_translated,
    # X_ocr,
    X_nouns,
    X_numbers,
    # X_cont,
    y,
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
                # X_translated_batch = [X_translated[i] for i in shuffled_indices[start_index:end_index]]
                # X_ocr_batch = [X_ocr[i] for i in shuffled_indices[start_index:end_index]]
                X_nouns_batch = [X_nouns[i] for i in shuffled_indices[start_index:end_index]]
                X_numbers_batch = [X_numbers[i] for i in shuffled_indices[start_index:end_index]]
                # X_cont_batch = [X_cont[i] for i in shuffled_indices[start_index:end_index]]

                y_batch = [y[i] for i in shuffled_indices[start_index:end_index]]

                yield ({
                    'title_input': np.asarray(X_title_batch),
                    # 'translated_input': np.asarray(X_translated_batch),
                    # 'ocr_input': np.asarray(X_ocr_batch),
                    'nouns_input': np.asarray(X_nouns_batch),
                    'numbers_input': np.asarray(X_numbers_batch),
                    # 'cont_input': np.asarray(X_cont_batch),
                    },
                    {'output': np.asarray(y_batch)})
            #end for
        #end while
    #end def

    return num_batches, _data_generator()
#end def


def train(
    model,
    X_title_train,
    # X_translated_train,
    # X_ocr_train，
    X_nouns_train,
    X_numbers_train,
    # X_cont_train,
    y_train,
    X_title_val=None,
    # X_translated_val=None,
    # X_ocr_val=None,
    X_nouns_val=None,
    X_numbers_val=None,
    # X_cont_val=None,
    y_val=None,
    weights_path='./weights/', weights_prefix='',
    class_weight=None, batch_size=128, epochs=32, **kwargs):

    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    train_steps, train_batches = batch_iter(
        X_title_train,
        # X_translated_train,
        # X_ocr_train,
        X_nouns_train,
        X_numbers_train,
        # X_cont_train,
        y_train,
        batch_size=batch_size)

    if y_val is not None:
        val_steps, val_batches = batch_iter(
            X_title_val,
            # X_translated_val,
            # X_ocr_val,
            X_nouns_val,
            X_numbers_val,
            # X_cont_val,
            y_val,
            batch_size=batch_size)

    # define early stopping callback
    callbacks_list = []
    if y_val is not None:
        early_stopping = dict(monitor='val_acc', patience=1, min_delta=0.0001, verbose=1)
        model_checkpoint = dict(filepath=weights_path + weights_prefix + '_{val_acc:.5f}_{acc:.5f}_{epoch:04d}.weights.h5',
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1,
                                verbose=1)
    else:
        early_stopping = dict(monitor='acc', patience=1, min_delta=0.001, verbose=1)
        model_checkpoint = dict(filepath=weights_path + weights_prefix + '_{acc:.5f}_{epoch:04d}.weights.h5',
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
    # X_translated,
    # X_ocr,
    X_nouns,
    X_numbers,
    # X_cont,
    batch_size=128, **kwargs):

    data_size = X_title.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)

            X_title_batch = [x for x in X_title[start_index:end_index]]
            # X_translated_batch = [x for x in X_translated[start_index:end_index]]
            # X_ocr_batch = [x for x in X_ocr[start_index:end_index]]
            X_nouns_batch = [x for x in X_nouns[start_index:end_index]]
            X_numbers_batch = [x for x in X_numbers[start_index:end_index]]
            # X_cont_batch = [x for x in X_cont[start_index:end_index]]
            yield ({
                'title_input': np.asarray(X_title_batch),
                # 'translated_input': np.asarray(X_translated_batch),
                # 'ocr_input': np.asarray(X_ocr_batch),
                'nouns_input': np.asarray(X_nouns_batch),
                'numbers_input': np.asarray(X_numbers_batch),
                # 'cont_input': np.asarray(X_cont_batch),
                })
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def test(
    model,
    X_title_test,
    # X_translated_test,
    # X_ocr_test,
    X_nouns_test,
    X_numbers_test,
    # X_cont_test,
    lb, mapping,
    batch_size=128, **kwargs):

    def _second_largest(numbers):
        if (len(numbers) < 2):
            return
        if ((len(numbers) == 2) and (numbers[0] == numbers[1])):
            return
        dup_items = set()
        uniq_items = []
        for x in numbers:
            if x not in dup_items:
                uniq_items.append(x)
                dup_items.add(x)
        uniq_items.sort()

        return uniq_items[-2]
    #end def

    test_steps, test_batches = predict_iter(
        X_title_test,
        # X_translated_test,
        # X_ocr_test,
        X_nouns_test,
        X_numbers_test,
        # X_cont_test,
        batch_size=batch_size, **kwargs)

    proba = model.predict_generator(generator=test_batches, steps=test_steps)

    largest = [max(p) for p in proba]
    second_largest = [_second_largest(p) for p in proba]

    first_pred = [str(mapping[lb.classes_[j]]) for i, p in enumerate(proba) for j, _p in enumerate(p) if _p == largest[i]]
    second_pred = [str(mapping[lb.classes_[j]]) for i, p in enumerate(proba) for j, _p in enumerate(p) if _p == second_largest[i]]

    return [first_pred[i] + ' ' + second_pred[i] for i in range(len(proba))]
#end def


def undersampling(df, y, proportion=0.15):
    size = df.shape[0]
    thresh = size * proportion

    label_counts = df[y].value_counts()

    keep_index = []

    for i, row in df.iterrows():
        label = row[y]
        r = min(thresh / label_counts[label], 1.0)
        if random.random() < r:
            keep_index.append(row.name)
    #end for

    return df.loc[keep_index]
#end def


def main():
    argparser = ArgumentParser(description='Run machine learning experiment.')
    argparser.add_argument('-i', '--train', type=str, metavar='<train_set>', required=True, help='Training data set.')
    argparser.add_argument('-t', '--test', type=str, metavar='<test_set>', required=True, help='Test data set.')
    argparser.add_argument('-m', '--mapping', type=str, metavar='<mapping_json>', required=True, help='Mapping json file.')
    argparser.add_argument('--seed', type=int, default=0, help='Random seed.')
    A = argparser.parse_args()

    log_level = 'INFO'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    # set seed
    np.random.seed(A.seed)
    random.seed(A.seed)
    tf.set_random_seed(A.seed)

    quick = False
    validate = False
    batch_size = 256
    epochs = 2

    # read in data
    train_df, mapping_dict = read(A.train, A.mapping, quick=quick)
    train_dict = dict(batch_size=batch_size, epochs=epochs)
    test_df = read(A.test)
    test_dict = dict(batch_size=batch_size, epochs=epochs)

    if validate:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=A.seed)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    logger.info('Encoding labels....')
    lb_dict = dict()
    for y in categorical_targets:
        lb_dict[y] = LabelBinarizer()
        to_be_trained_df = train_df.loc[train_df[y] != 'unk']
        # if y == 'Benefits':
        #     proportion = max(1 / len(mapping_dict[y]), 0.03)
        #     to_be_trained_df = undersampling(
        #         to_be_trained_df, y, proportion)

        train_dict['X_' + y + '_train_index'] = list(to_be_trained_df.index.values)
        train_dict['y_' + y + '_train'] = lb_dict[y].fit_transform(to_be_trained_df[y][train_dict['X_' + y + '_train_index']])
        if validate:
            to_be_validated_df = val_df.loc[val_df[y] != 'unk']
            train_dict['X_' + y + '_val_index'] = list(to_be_validated_df.index.values)
            train_dict['y_' + y + '_val'] = lb_dict[y].transform(to_be_validated_df[y][train_dict['X_' + y + '_val_index']])
    #end for
    logger.info('Done encoding labels....')

    for y in categorical_targets:
        print('='*50)
        print(y)
        print('='*50)
        title_vec = CountVectorizer(
            max_features=50000,
            strip_accents='unicode',
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            dtype=np.float32,
            # min_df=5,
            # max_df=.9
            ).fit(train_df['title'][train_dict['X_' + y + '_train_index']].append(test_df['title']))
            # ).fit(train_df['title'][train_dict['X_' + y + '_train_index']].append(val_df['title'][train_dict['X_' + y + '_val_index']]).append(test_df['title']))

        # translated_vec = CountVectorizer(
        #     max_features=10000,
        #     strip_accents='unicode',
        #     stop_words='english',
        #     analyzer='word',
        #     token_pattern=r'\w{1,}',
        #     ngram_range=(1, 2),
        #     dtype=np.float32,
        #     # min_df=5,
        #     # max_df=.9
        #     ).fit(train_df['translated'][train_dict['X_' + y + '_train_index']].append(test_df['translated']))
        #     # ).fit(train_df['translated'][train_dict['X_' + y + '_train_index']].append(val_df['translated'][train_dict['X_' + y + '_val_index']]).append(test_df['translated']))

        # ocr_vec = CountVectorizer(
        #     max_features=10000,
        #     strip_accents='unicode',
        #     stop_words='english',
        #     analyzer='char_wb',
        #     token_pattern=r'\w{1,}',
        #     ngram_range=(3, 5),
        #     dtype=np.float32,
        #     # min_df=5,
        #     # max_df=.9
        #     # ).fit(train_df['ocr'][train_dict['X_' + y + '_train_index']].append(test_df['ocr']))
        #     ).fit(train_df['ocr'][train_dict['X_' + y + '_train_index']].append(val_df['ocr'][train_dict['X_' + y + '_val_index']]).append(test_df['ocr']))

        nouns_vec = CountVectorizer(
            max_features=20000,
            strip_accents='unicode',
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            dtype=np.float32,
            # min_df=5,
            # max_df=.9
            ).fit(train_df['nouns'][train_dict['X_' + y + '_train_index']].append(test_df['nouns']))
            # ).fit(train_df['nouns'][train_dict['X_' + y + '_train_index']].append(val_df['nouns'][train_dict['X_' + y + '_val_index']]).append(test_df['nouns']))

        numbers_vec = CountVectorizer(
            max_features=20000,
            strip_accents='unicode',
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            dtype=np.float32,
            # min_df=5,
            # max_df=.9
            ).fit(train_df['numbers'][train_dict['X_' + y + '_train_index']].append(test_df['numbers']))
            # ).fit(train_df['numbers'][train_dict['X_' + y + '_train_index']].append(val_df['numbers'][train_dict['X_' + y + '_val_index']]).append(test_df['numbers']))

        train_dict['X_title_train'] = title_vec.transform(train_df['title'][train_dict['X_' + y + '_train_index']]).toarray()
        # train_dict['X_translated_train'] = translated_vec.transform(train_df['translated'][train_dict['X_' + y + '_train_index']]).toarray()
        # train_dict['X_ocr_train'] = ocr_vec.transform(train_df['ocr'][train_dict['X_' + y + '_train_index']]).toarray()
        train_dict['X_nouns_train'] = nouns_vec.transform(train_df['nouns'][train_dict['X_' + y + '_train_index']]).toarray()
        train_dict['X_numbers_train'] = numbers_vec.transform(train_df['numbers'][train_dict['X_' + y + '_train_index']]).toarray()
        # train_dict['X_cont_train'] = train_df[continuous_features].values[train_dict['X_' + y + '_train_index']]

        if validate:
            train_dict['X_title_val'] = title_vec.transform(val_df['title'][train_dict['X_' + y + '_val_index']]).toarray()
            # train_dict['X_translated_val'] = translated_vec.transform(val_df['translated'][train_dict['X_' + y + '_val_index']]).toarray()
            # train_dict['X_ocr_val'] = ocr_vec.transform(val_df['ocr'][train_dict['X_' + y + '_val_index']]).toarray()
            train_dict['X_nouns_val'] = nouns_vec.transform(val_df['nouns'][train_dict['X_' + y + '_val_index']]).toarray()
            train_dict['X_numbers_val'] = numbers_vec.transform(val_df['numbers'][train_dict['X_' + y + '_val_index']]).toarray()
            # train_dict['X_cont_val'] = val_df[continuous_features].values[train_dict['X_' + y + '_val_index']]

        model = build_model(
            title_input_shape=train_dict['X_title_train'].shape[1:],
            # translated_input_shape=train_dict['X_translated_train'].shape[1:],
            # ocr_input_shape=train_dict['X_ocr_train'].shape[1:],
            nouns_input_shape=train_dict['X_nouns_train'].shape[1:],
            numbers_input_shape=train_dict['X_numbers_train'].shape[1:],
            # cont_input_shape=train_dict['X_cont_train'].shape[1:],
            output_shape=train_dict['y_' + y + '_train'].shape[1])
        # print(model.summary())

        train_dict['model'] = model
        train_dict['y_train'] = train_dict['y_' + y + '_train']
        if validate:
            train_dict['y_val'] = train_dict['y_' + y + '_val']
        train_dict['weights_prefix'] = y

        model = train(**train_dict)
        for k in ['X_title_train', 'X_nouns_train', 'X_numbers_train', 'X_title_val', 'X_nouns_val', 'X_numbers_val', 'model', 'y_train', 'y_val', 'weights_prefix']:
            try:
                del train_dict[k]
            except KeyError:
                pass
        #end for
        gc.collect()

        test_dict['model'] = model
        test_dict['lb'] = lb_dict[y]
        test_dict['mapping'] = mapping_dict[y]
        test_dict['X_title_test'] = title_vec.transform(test_df['title'].values).toarray()
        # test_dict['X_translated_test'] = translated_vec.transform(test_df['translated'].values).toarray()
        # test_dict['X_ocr_test'] = ocr_vec.transform(test_df['ocr'].values).toarray()
        test_dict['X_nouns_test'] = nouns_vec.transform(test_df['nouns'].values).toarray()
        test_dict['X_numbers_test'] = numbers_vec.transform(test_df['numbers'].values).toarray()
        # test_dict['X_cont_test'] = test_df[continuous_features].values

        del title_vec, nouns_vec, numbers_vec
        gc.collect()

        test_df[y] = test(**test_dict)
        for k in ['X_title_test', 'X_nouns_test', 'X_numbers_test', 'model', 'lb', 'mapping']:
            del test_dict[k]
        #end for

        del model
        gc.collect()
    #end for

    test_df.to_csv('./data/full_50000_n20000_num20000_beauty_test_proba.csv', index=False)
#end def


if __name__ == '__main__': main()
