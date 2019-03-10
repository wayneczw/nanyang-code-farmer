import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os

OCR_COL = 'ocr_result'
DOMINANT_COL = 'dominant_colors'
TRANSLATION_COL = 'translated'


def argParse():
    argparser = ArgumentParser(description='Combine csv files')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to base csv file.')
    argparser.add_argument('-o', '--ocr', type=str, help='Path to csv file containing ocr column.')
    argparser.add_argument('-t', '--translation', type=str, help='Path to csv file containing translation column.')
    argparser.add_argument('-d', '--dominant_colors', type=str, help='Path to csv file containing dominant color column.')

    A = argparser.parse_args()
    return A


def add_column(base_df, new_df, col_name):
    col = new_df[col_name]
    base_df[col_name] = pd.Series(col)
    return base_df


def main():

    A = argParse()

    # base csv
    df = pd.read_csv(A.file)

    # ocr
    if A.ocr is not None:
        df = add_column(df, pd.read_csv(A.ocr), OCR_COL)

    # translation
    if A.translation is not None:
        df = add_column(df, pd.read_csv(A.translation), TRANSLATION_COL)

    # dominant colors
    if A.dominant_colors is not None:
        df = add_column(df, pd.read_csv(A.dominant_colors), DOMINANT_COL)

    dir_name, base_name = os.path.split(A.file)
    new_name = os.path.join('combined', base_name.rsplit('.', 1)[0] + '_combined.csv')
    new_path = os.path.join(dir_name, new_name)
    print(df)
    df.to_csv(new_path, index=False)


if __name__ == '__main__':
    main()
