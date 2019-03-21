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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


logger = logging.getLogger(__name__)

raw_categorical_targets = [
    'Brand', 'Phone Model', 'Operating System', 'Features',
    'Network Connections', 'Memory RAM',
    'Warranty Period',
    'Storage Capacity', 'Color Family',
    'Camera', 'Phone Screen Size']

categorical_targets = [
    'Brand', 'Phone Model', 'Operating System', 'Features',
    'Network Connections', 'Memory RAM',
    'Warranty Period',
    'Storage Capacity', 'Color Family',
    'Camera', 'Phone Screen Size']

categorical_features = [
    'nouns', 'numbers']

continuous_features = [
    "operating system_count", "operating_count", "system_count", "os_count", "symbian_count", "windows_count", "samsung_count", "blackberry_count", "nokia_count", "android_count", "ios_count", "features_count", "expandable memory_count", "touchscreen_count", "fingerprint sensor_count", "dustproof_count", "waterproof_count", "wifi_count", "gps_count", "network_count", "connections_count", "4g_count", "2g_count", "3g_count", "3.5g_count", "memory_count", "ram_count", "4gb_count", "2gb_count", "1.5gb_count", "16gb_count", "512mb_count", "8gb_count", "3gb_count", "10gb_count", "1gb_count", "6gb_count", "warranty_count", "period_count", "yr_count", "mth_count", "year_count", "month_count", "years_count", "months_count", "7 months_count", "4 months_count", "6 months_count", "3 months_count", "10 years_count", "2 month_count", "11 months_count", "10 months_count", "5 months_count", "3 years_count", "2 years_count", "1 month_count", "18 months_count", "1 year_count", "storage_count", "capacity_count", "gb_count", "mb_count", "256gb_count", "128gb_count", "64gb_count", "512gb_count", "4mb_count", "128mb_count", "32gb_count", "256mb_count", "color_count", "colour_count", "family_count", "blue_count", "gold_count", "brown_count", "navy blue_count", "yellow_count", "neutral_count", "rose gold_count", "light blue_count", "dark grey_count", "silver_count", "pink_count", "gray_count", "army_count", "green_count", "army green_count", "deep_count", "deep blue_count", "purple_count", "rose_count", "light_count", "grey_count", "light grey_count", "black_count", "deep black_count", "off_count", "white_count", "off white_count", "multicolor_count", "apricot_count", "orange_count", "red_count", "camera_count", "mp_count", "single_count", "42mp_count", "dua slot_count", "5 mp_count", "3 mp_count", "1 mp_count", "8 mp_count", "single camera_count", "24 mp_count", "16mp_count", "13mp_count", "6 mp_count", "10mp_count", "2 mp_count", "20 mp_count", "4 mp_count", "phone_count", "screen_count", "size_count", "inch_count", "inches_count", "4.6 to 5 inches_count", "4.1 to 4.5 inches_count", "less than 3.5 inches_count", "3.6 to 4 inches_count", "more than 5.6 inches_count", "5.1 to 5.5 inches_count"]


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
    translated_input_shape,
    # ocr_input_shape,
    nouns_input_shape,
    numbers_input_shape,
    # cont_input_shape,
    output_shape,
    brand_input_shape=None,
    model_input_shape=None,
    dropout_rate=0, kernel_regularizer=0,
    activity_regularizer=0, bias_regularizer=0):
    
    title_input = Input(title_input_shape, name='title_input')
    title = Dense(2048)(title_input)
    title = Dropout(dropout_rate)(title)
    title = Dense(min(1024, output_shape*4))(title)
    title = Dropout(dropout_rate)(title)

    translated_input = Input(translated_input_shape, name='translated_input')
    translated = Dense(512)(translated_input)
    translated = Dropout(dropout_rate)(translated)
    translated = Dense(min(256, output_shape*2))(translated)
    translated = Dropout(dropout_rate)(translated)

    # ocr_input = Input(ocr_input_shape, name='ocr_input')
    # ocr = Dense(512)(ocr_input)
    # ocr = Dropout(dropout_rate)(ocr)
    # ocr = Dense(min(256, output_shape*4))(ocr)
    # ocr = Dropout(dropout_rate)(ocr)

    nouns_input = Input(nouns_input_shape, name='nouns_input')
    nouns = Dense(512)(nouns_input)
    nouns = Dropout(dropout_rate)(nouns)
    nouns = Dense(min(256, output_shape*2))(nouns)
    nouns = Dropout(dropout_rate)(nouns)

    numbers_input = Input(numbers_input_shape, name='numbers_input')
    numbers = Dense(512)(numbers_input)
    numbers = Dropout(dropout_rate)(numbers)
    numbers = Dense(min(256, output_shape*2))(numbers)
    numbers = Dropout(dropout_rate)(numbers)

    # cont_input = Input(cont_input_shape, name='cont_input')
    # cont = Dense(256)(cont_input)
    # cont = Dropout(dropout_rate)(cont)
    # cont = Dense(min(128, output_shape*2))(cont)
    # cont = Dropout(dropout_rate)(cont)

    # inputs = [title_input, translated_input, ocr_input, nouns_input, numbers_input, cont_input]
    inputs = [title_input, translated_input, nouns_input, numbers_input]

    if brand_input_shape is not None:
        brand_input = Input(brand_input_shape, name='brand_input')
        brand = Dense(16)(brand_input)
        brand = Dropout(dropout_rate)(brand)
        inputs.append(brand_input)

    if model_input_shape is not None:
        model_input = Input(model_input_shape, name='model_input')
        model = Dense(16)(model_input)
        model = Dropout(dropout_rate)(model)
        inputs.append(model_input)

    if brand_input_shape is not None and model_input_shape is not None:
        # x = concatenate([title, translated, ocr, nouns, numbers, cont, brand, model])
        x = concatenate([title, translated, nouns, numbers, brand, model])
        # x = Dense(min(1024, output_shape*4))(x)
        # x = Dropout(dropout_rate)(x)
        # x = Dense(min(512, output_shape*2))(x)
        # x = Dropout(dropout_rate)(x)
    elif brand_input_shape is not None:
        # x = concatenate([title, translated, ocr, nouns, numbers, cont, brand])
        x = concatenate([title, translated, nouns, numbers, brand])
        # x = Dense(min(1024, output_shape*4))(x)
        # x = Dropout(dropout_rate)(x)
        # x = Dense(min(512, output_shape*2))(x)
        # x = Dropout(dropout_rate)(x)
    else:
        # x = concatenate([title, translated, ocr, nouns, numbers, cont])
        x = concatenate([title, translated, nouns, numbers])
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
    X_translated,
    # X_ocr,
    X_nouns,
    X_numbers,
    # X_cont,
    y,
    X_brand=None, X_model=None,
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
                X_translated_batch = [X_translated[i] for i in shuffled_indices[start_index:end_index]]
                # X_ocr_batch = [X_ocr[i] for i in shuffled_indices[start_index:end_index]]
                X_nouns_batch = [X_nouns[i] for i in shuffled_indices[start_index:end_index]]
                X_numbers_batch = [X_numbers[i] for i in shuffled_indices[start_index:end_index]]
                # X_cont_batch = [X_cont[i] for i in shuffled_indices[start_index:end_index]]

                y_batch = [y[i] for i in shuffled_indices[start_index:end_index]]

                if X_brand is not None:
                    X_brand_batch = [x for x in X_brand[start_index:end_index]]
                if X_model is not None:
                    X_model_batch = [x for x in X_model[start_index:end_index]]
                
                if X_model is not None:
                    yield ({
                        'title_input': np.asarray(X_title_batch),
                        'translated_input': np.asarray(X_translated_batch),
                        # 'ocr_input': np.asarray(X_ocr_batch),
                        'nouns_input': np.asarray(X_nouns_batch),
                        'numbers_input': np.asarray(X_numbers_batch),
                        # 'cont_input': np.asarray(X_cont_batch),
                        'brand_input': np.asarray(X_brand_batch),
                        'model_input': np.asarray(X_model_batch),
                        },
                        {'output': np.asarray(y_batch)})
                elif X_brand is not None:
                    yield ({
                        'title_input': np.asarray(X_title_batch),
                        'translated_input': np.asarray(X_translated_batch),
                        # 'ocr_input': np.asarray(X_ocr_batch),
                        'nouns_input': np.asarray(X_nouns_batch),
                        'numbers_input': np.asarray(X_numbers_batch),
                        # 'cont_input': np.asarray(X_cont_batch),
                        'brand_input': np.asarray(X_brand_batch),
                        },
                        {'output': np.asarray(y_batch)})
                else:
                    yield ({
                        'title_input': np.asarray(X_title_batch),
                        'translated_input': np.asarray(X_translated_batch),
                        # 'ocr_input': np.asarray(X_ocr_batch),
                        'nouns_input': np.asarray(X_nouns_batch),
                        'numbers_input': np.asarray(X_numbers_batch),
                        # 'cont_input': np.asarray(X_cont_batch),
                        },
                        {'output': np.asarray(y_batch)})
                #end if
            #end for
        #end while
    #end def

    return num_batches, _data_generator()
#end def


def train(
    model,
    X_title_train,
    X_translated_train,
    # X_ocr_train,
    X_nouns_train,
    X_numbers_train,
    # X_cont_train,
    y_train,
    X_brand_train=None,
    X_model_train=None,
    X_title_val=None,
    X_translated_val=None,
    # X_ocr_val=None,
    X_nouns_val=None,
    X_numbers_val=None,
    # X_cont_val=None,
    y_val=None,
    X_brand_val=None, X_model_val=None,
    weights_path='./weights/', weights_prefix='',
    # weights_path='.\\weights\\', weights_prefix='',
    class_weight=None, batch_size=128, epochs=32, **kwargs):

    tf_session = tf.Session()
    K.set_session(tf_session)
    K.get_session().run(tf.tables_initializer())

    init_op = tf.global_variables_initializer()
    tf_session.run(init_op)

    train_steps, train_batches = batch_iter(
        X_title_train,
        X_translated_train,
        # X_ocr_train,
        X_nouns_train,
        X_numbers_train,
        # X_cont_train,
        y_train,
        X_brand_train, X_model_train,
        batch_size=batch_size)

    if y_val is not None:
        val_steps, val_batches = batch_iter(
            X_title_val,
            X_translated_val,
            # X_ocr_val,
            X_nouns_val,
            X_numbers_val,
            # X_cont_val,
            y_val,
            X_brand_val, X_model_val,
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
    X_translated,
    # X_ocr,
    X_nouns,
    X_numbers,
    # X_cont,
    X_brand=None, X_model=None,
    batch_size=128, **kwargs):

    data_size = X_title.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)

            X_title_batch = [x for x in X_title[start_index:end_index]]
            X_translated_batch = [x for x in X_translated[start_index:end_index]]
            # X_ocr_batch = [x for x in X_ocr[start_index:end_index]]
            X_nouns_batch = [x for x in X_nouns[start_index:end_index]]
            X_numbers_batch = [x for x in X_numbers[start_index:end_index]]
            # X_cont_batch = [x for x in X_cont[start_index:end_index]]

            if X_brand is not None:
                X_brand_batch = [x for x in X_brand[start_index:end_index]]
            if X_model is not None:
                X_model_batch = [x for x in X_model[start_index:end_index]]

            if X_model is not None:
                yield ({
                    'title_input': np.asarray(X_title_batch),
                    'translated_input': np.asarray(X_translated_batch),
                    # 'ocr_input': np.asarray(X_ocr_batch),
                    'nouns_input': np.asarray(X_nouns_batch),
                    'numbers_input': np.asarray(X_numbers_batch),
                    # 'cont_input': np.asarray(X_cont_batch),
                    'brand_input': np.asarray(X_brand_batch),
                    'model_input': np.asarray(X_model_batch),
                    })
            elif X_brand is not None:
                yield ({
                    'title_input': np.asarray(X_title_batch),
                    'translated_input': np.asarray(X_translated_batch),
                    # 'ocr_input': np.asarray(X_ocr_batch),
                    'nouns_input': np.asarray(X_nouns_batch),
                    'numbers_input': np.asarray(X_numbers_batch),
                    # 'cont_input': np.asarray(X_cont_batch),
                    'brand_input': np.asarray(X_brand_batch),
                    })
            else:
                yield ({
                    'title_input': np.asarray(X_title_batch),
                    'translated_input': np.asarray(X_translated_batch),
                    # 'ocr_input': np.asarray(X_ocr_batch),
                    'nouns_input': np.asarray(X_nouns_batch),
                    'numbers_input': np.asarray(X_numbers_batch),
                    # 'cont_input': np.asarray(X_cont_batch),
                    })
            #end if
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def test(
    model,
    X_title_test,
    X_translated_test,
    # X_ocr_test,
    X_nouns_test,
    X_numbers_test,
    # X_cont_test,
    lb, mapping,
    X_brand_test=None, X_model_test=None,
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
        X_translated_test,
        # X_ocr_test,
        X_nouns_test,
        X_numbers_test,
        # X_cont_test,
        X_brand_test, X_model_test,
        batch_size=batch_size, **kwargs)

    proba = model.predict_generator(generator=test_batches, steps=test_steps)

    largest = [max(p) for p in proba]
    second_largest = [_second_largest(p) for p in proba]

    first_pred = [str(mapping[lb.classes_[j]]) for i, p in enumerate(proba) for j, _p in enumerate(p) if _p == largest[i]]
    second_pred = [str(mapping[lb.classes_[j]]) for i, p in enumerate(proba) for j, _p in enumerate(p) if _p == second_largest[i]]

    return [first_pred[i] + ' ' + second_pred[i] for i in range(len(proba))]
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
            max_features=20000,
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

        translated_vec = CountVectorizer(
            max_features=10000,
            strip_accents='unicode',
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            dtype=np.float32,
            # min_df=5,
            # max_df=.9
            ).fit(train_df['translated'][train_dict['X_' + y + '_train_index']].append(test_df['translated']))
            # ).fit(train_df['translated'][train_dict['X_' + y + '_train_index']].append(val_df['translated'][train_dict['X_' + y + '_val_index']]).append(test_df['translated']))

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
            max_features=10000,
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
            max_features=5000,
            strip_accents='unicode',
            stop_words='english',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 2),
            dtype=np.float32,
            # min_df=5,
            # max_df=.9
            ).fit(train_df['numbers'][train_dict['X_' + y + '_train_index']].append(test_df['numbers']))
            # ).fit(train_df['numbers'][train_dict['X_' + y + '_train_index']].append(val_df['numbers'][train_dict['X_' + y + '_val_index']]).append(test_df['numbers']))

        train_dict['X_title_train'] = title_vec.transform(train_df['title'][train_dict['X_' + y + '_train_index']]).toarray()
        train_dict['X_translated_train'] = translated_vec.transform(train_df['translated'][train_dict['X_' + y + '_train_index']]).toarray()
        # train_dict['X_ocr_train'] = ocr_vec.transform(train_df['ocr'][train_dict['X_' + y + '_train_index']]).toarray()
        train_dict['X_nouns_train'] = nouns_vec.transform(train_df['nouns'][train_dict['X_' + y + '_train_index']]).toarray()
        train_dict['X_numbers_train'] = numbers_vec.transform(train_df['numbers'][train_dict['X_' + y + '_train_index']]).toarray()
        # train_dict['X_cont_train'] = train_df[continuous_features].values[train_dict['X_' + y + '_train_index']]

        brand_input_shape = None
        model_input_shape = None

        if y != 'Brand':
            train_dict['X_brand_train'] = lb_dict['Brand'].transform(train_df['Brand'][train_dict['X_' + y + '_train_index']])
            brand_input_shape = train_dict['X_brand_train'].shape[1:]
            if y != 'Phone Model':
                train_dict['X_model_train'] = lb_dict['Phone Model'].transform(train_df['Phone Model'][train_dict['X_' + y + '_train_index']])
                model_input_shape = train_dict['X_model_train'].shape[1:]

        if validate:
            train_dict['X_title_val'] = title_vec.transform(val_df['title'][train_dict['X_' + y + '_val_index']]).toarray()
            train_dict['X_translated_val'] = translated_vec.transform(val_df['translated'][train_dict['X_' + y + '_val_index']]).toarray()
            # train_dict['X_ocr_val'] = ocr_vec.transform(val_df['ocr'][train_dict['X_' + y + '_val_index']]).toarray()
            train_dict['X_nouns_val'] = nouns_vec.transform(val_df['nouns'][train_dict['X_' + y + '_val_index']]).toarray()
            train_dict['X_numbers_val'] = numbers_vec.transform(val_df['numbers'][train_dict['X_' + y + '_val_index']]).toarray()
            # train_dict['X_cont_val'] = val_df[continuous_features].values[train_dict['X_' + y + '_val_index']]
            if y != 'Brand':
                train_dict['X_brand_val'] = lb_dict['Brand'].transform(train_df['Brand'][train_dict['X_' + y + '_val_index']])
                if y != 'Phone Model':
                    train_dict['X_model_val'] = lb_dict['Phone Model'].transform(train_df['Phone Model'][train_dict['X_' + y + '_val_index']])

        model = build_model(
            title_input_shape=train_dict['X_title_train'].shape[1:],
            translated_input_shape=train_dict['X_translated_train'].shape[1:],
            # ocr_input_shape=train_dict['X_ocr_train'].shape[1:],
            nouns_input_shape=train_dict['X_nouns_train'].shape[1:],
            numbers_input_shape=train_dict['X_numbers_train'].shape[1:],
            # cont_input_shape=train_dict['X_cont_train'].shape[1:],
            brand_input_shape=brand_input_shape,
            model_input_shape=model_input_shape,
            output_shape=train_dict['y_' + y + '_train'].shape[1])
        # print(model.summary())

        train_dict['model'] = model
        train_dict['y_train'] = train_dict['y_' + y + '_train']
        if validate:
            train_dict['y_val'] = train_dict['y_' + y + '_val']
        train_dict['weights_prefix'] = y
        model = train(**train_dict)

        test_dict['model'] = model
        test_dict['lb'] = lb_dict[y]
        test_dict['mapping'] = mapping_dict[y]
        test_dict['X_title_test'] = title_vec.transform(test_df['title'].values).toarray()
        test_dict['X_translated_test'] = translated_vec.transform(test_df['translated'].values).toarray()
        # test_dict['X_ocr_test'] = ocr_vec.transform(test_df['ocr'].values).toarray()
        test_dict['X_nouns_test'] = nouns_vec.transform(test_df['nouns'].values).toarray()
        test_dict['X_numbers_test'] = numbers_vec.transform(test_df['numbers'].values).toarray()
        # test_dict['X_cont_test'] = test_df[continuous_features].values
        if y != 'Brand':
            brand_mapping = dict((v, k) for k, v in mapping_dict['Brand'].items())
            brands = test_df['Brand'].apply(lambda x: brand_mapping[int(x.split()[0])]).values
            brands = lb_dict['Brand'].transform(brands)
            test_dict['X_brand_test'] = brands
            if y != 'Phone Model':
                model_mapping = dict((v, k) for k, v in mapping_dict['Phone Model'].items())
                models = test_df['Phone Model'].apply(lambda x: model_mapping[int(x.split()[0])]).values
                models = lb_dict['Phone Model'].transform(models)
                test_dict['X_model_test'] = models

        test_df[y] = test(**test_dict)
    #end for

    test_df.to_csv('./data/full_t_mobile_test_proba.csv', index=False)
    # test_df.to_csv('.\\data\\mobile_test_proba.csv', index=False)
#end def


if __name__ == '__main__': main()
