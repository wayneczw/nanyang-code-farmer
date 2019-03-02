import cv2
import time
import re
import string
from nltk import word_tokenize
from argparse import ArgumentParser
import pytesseract
import multiprocessing as mp
from multiprocessing import cpu_count
import pandas as pd
from tqdm import tqdm
import sys


ncores = cpu_count()


def ocr(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Not found: " + img_path)
        return ""
    else:
        text = pytesseract.image_to_string(img)
        return clean(text)


def clean(text):
    text = re.sub(r'[{}]'.format(string.punctuation), " ", text.lower())
    return ' '.join(word_tokenize(text))


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

    new_csv = A.file.rsplit('.', 1)[0] + '_ocr_{}_{}.csv'.format(A.start_row, A.end_row)

    df = df[A.start_row: A.end_row + 1]

    row_count, col_count = df.shape

    df['temp_path'] = A.image_folder + df['image_path']

    ncores = cpu_count()

    ocr_res = []
    with mp.Pool(ncores) as pool:
        ocr_gen = pool.imap(ocr, df['temp_path'], chunksize=10)

        with tqdm(total=row_count) as pbar:
            for res in ocr_gen:
                ocr_res.append(res)
                pbar.update(1)

    print("Loading to data frame...")
    df['ocr_result'] = ocr_res

    df.drop(columns=['temp_path'], inplace=True)
    df.to_csv(new_csv, index=False)
    print("Completed. New csv: " + new_csv)

if __name__ == '__main__':
    main()
