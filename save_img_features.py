import cv2
import gc
import joblib
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from multiprocessing import cpu_count

from keras.applications import NASNetMobile
from keras.applications import ResNet50
from keras.applications.inception_v3 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model

logger = logging.getLogger(__name__)
ncores = cpu_count()


def predict_iter(X_data, batch_size=1024):
    data_size = X_data.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1

    def _data_generator():
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_size)
            X_batch = [d for d in X_data[start_index:end_index]]

            yield ({'input': np.asarray(X_batch)})
        #end for
    #end def

    return num_batches, _data_generator()
#end def


def get_img_features(model, data):
    data_steps, data_batches = predict_iter(data, batch_size=1024)
    features = model.predict_generator(generator=data_batches, steps=data_steps)

    return features
#end def


def resize331(img_path):
    # if not img_path.endswith('.jpg'):
    #     img_path += '.jpg'
    # elif img_path.startswith('new_'):
    #     img_path = img_path[4:]
    img = cv2.imread(img_path)
    return cv2.resize(img, (331, 331))
#end def


def resize299(img_path):
    # if not img_path.endswith('.jpg'):
    #     img_path += '.jpg'
    # elif img_path.startswith('new_'):
    #     img_path = img_path[4:]
    img = cv2.imread(img_path)
    return cv2.resize(img, (299, 299))
#end def


def resize224(img_path):
    # if not img_path.endswith('.jpg'):
    #     img_path += '.jpg'
    # elif img_path.startswith('new_'):
    #     img_path = img_path[4:]
    img = cv2.imread(img_path)
    return cv2.resize(img, (224, 224))
#end def


def read_img(df, width=299):
    with mp.Pool(ncores) as pool:
        if width == 224:
            imgs = pool.imap(resize224, df['image_path'], chunksize=10)
        elif width == 299:
            imgs = pool.imap(resize299, df['image_path'], chunksize=10)
        else:
            imgs = pool.imap(resize331, df['image_path'], chunksize=10)
        imgs = [i for i in imgs]
    #end with

    return np.asarray(imgs)
#end def


def main():
    argparser = ArgumentParser(description='Run machine learning experiment.')
    argparser.add_argument('-i', '--train', type=str, metavar='<train_set>', required=True, help='Training data set.')
    argparser.add_argument('-t', '--test', type=str, metavar='<test_set>', required=True, help='Test data set.')
    # argparser.add_argument('--image_folder', type=str, default="", required=True, help='Path to folder which contains the x_image folder')
    A = argparser.parse_args()

    log_level = 'INFO'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    # define model
    width = 224
    # cnn_model = ResNet50(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    cnn_model = NASNetMobile(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    x_input = Input((width, width, 3), name='input')
    x = Lambda(preprocess_input, name='preprocessing')(x_input)
    x = cnn_model(x)  # Transfer Learning
    output = GlobalAveragePooling2D(name='output')(x)
    model = Model(inputs=[x_input], outputs=[output])

    # load data
    logger.info('Featurizing train....')
    train_df = pd.read_csv(A.train)
    train_df['image_path'] = train_df['image_path'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
    train_df['image_path'] = train_df['image_path'].apply(lambda x: x[4:] if x.startswith('new_') else x)
    # train_df = train_df[:128]

    # get features
    batch_size = 2048
    data_size = train_df.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1
    train = None
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_size)
        _df = train_df[start_index:end_index]
        _train = get_img_features(model, read_img(_df, width=width))

        if train is None:
            train = _train
        else:
            train = np.concatenate((train, _train))
        #end if
        logger.info('Done with {}/{} batches....'.format(i+1, num_batches))
    #end for

    joblib.dump(train, A.train.split('.')[0] + '_img_features.joblib', compress=True)
    del train_df, train
    gc.collect()

    logger.info('Featurizing test....')
    test_df = pd.read_csv(A.test)
    test_df['image_path'] = test_df['image_path'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
    test_df['image_path'] = test_df['image_path'].apply(lambda x: x[4:] if x.startswith('new_') else x)
    # test_df = test_df[:128]

    batch_size = 4096
    data_size = test_df.shape[0]
    num_batches = int((data_size - 1) / batch_size) + 1
    test = None
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_size)
        _df = test_df[start_index:end_index]
        _test = get_img_features(model, read_img(_df, width=width))
        
        if test is None:
            test = _test
        else:
            test = np.concatenate((test, _test))
        #end if
        logger.info('Done with {}/{} batches....'.format(i+1, num_batches))
    #end for

    joblib.dump(test, A.test.split('.')[0] + '_img_features.joblib', compress=True)
#end def


if __name__ == '__main__': main()
