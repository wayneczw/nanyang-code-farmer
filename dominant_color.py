import cv2
import os
import time
import pandas as pd
from argparse import ArgumentParser
import multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
import math
from tqdm import tqdm
import numpy as np


# Using webcolors==1.8.1
# get all color names and rgb values in css3
COLOR_NAMES = webcolors.css3_hex_to_names.values()
COLOR_MAP = {c: list(webcolors.name_to_rgb(c)) for c in COLOR_NAMES}


def get_dominant_color(img_path, k=5, size=(160, 160)):
    img = cv2.imread(img_path)
    if img is None:
        print("Not found: " + img_path)
        return ""
    else:
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        x, y, w, h = preprocess(img)
        if w > 0 and h > 0:
            img = img[y:y + h, x:x + w]
        top_rgb = get_top_colors(img)
        res = []
        for rgb in top_rgb:
            res.append(get_closest_color(rgb, COLOR_MAP))
        return ' '.join(res)


def distance(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def get_closest_color(rgb, color_map):
    dist_dict = {k: distance(rgb, v) for (k, v) in COLOR_MAP.items()}
    return min(dist_dict, key=dist_dict.get)


def get_top_colors(img, k=5, top=3):

    # reshape the image to be a list of pixels
    pixels = img.reshape((img.shape[0] * img.shape[1], 3))
    # cluster and assign labels to the pixels 
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(pixels)

    label_counts = Counter(labels)
    top_rgb = []
    for (i, counts) in label_counts.most_common(top):
        color = clt.cluster_centers_[i]
        # bgr to rgb
        color[0], color[2] = color[2], color[0]
        top_rgb.append(color)
    return top_rgb


def preprocess(im):
    """ Return bbox (left, top, width height) of image without white outline """
    # Change image to gray-scale
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Convert image to black and white
    (thresh, img_gray) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Crop image to bounding box
    im_inverted = 255 * (img_gray < 128).astype(np.uint8)  # Inverts graph to white
    coordinates = cv2.findNonZero(im_inverted)
    # x, y, w, h = cv.boundingRect(coordinates)  # Get bounding box
    # im = im[y:y + h, x:x + w]
    return cv2.boundingRect(coordinates)


def main():
    argparser = ArgumentParser(description='Run ocr on images in csv file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    argparser.add_argument('-i', '--image_folder', type=str, default="", help='Path to folder which contains the x_image folder')
    argparser.add_argument('-s', '--start_row', type=int, default=0, help='Starting row allocated')
    argparser.add_argument('-e', '--end_row', type=int, help='Ending row allocated (inclusive)')
    A = argparser.parse_args()

    df = pd.read_csv(A.file)

    if A.end_row is None:
        A.end_row = df.shape[0] - 1

    new_csv = A.file.rsplit('.', 1)[0] + '_color_{}_{}.csv'.format(A.start_row, A.end_row)

    df = df[A.start_row: A.end_row + 1]

    row_count, col_count = df.shape

    df['temp_path'] = [os.path.join(A.image_folder, path) for path in df['image_path']]

    ncores = cpu_count()

    color_res = []
    with mp.Pool(ncores) as pool:
        color_gen = pool.imap(get_dominant_color, df['temp_path'], chunksize=10)

        with tqdm(total=row_count) as pbar:
            for res in color_gen:
                color_res.append(res)
                pbar.update(1)
    
    print("Loading to data frame...")
    df['dominant_colors'] = color_res

    df.drop(columns=['temp_path'], inplace=True)
    df.to_csv(new_csv, index=False)
    print("Completed. New csv: " + new_csv)


if __name__ == '__main__':
    main()
