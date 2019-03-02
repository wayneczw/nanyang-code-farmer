import csv
import sys
import time
import pandas as pd
from argparse import ArgumentParser


def check_attributes(df, attributes):
    df.columns = map(str.lower, df.columns)
    for attr in attributes:
        if attr.lower() not in df.columns:
            print("Cannot find {} column in the csv file. Please check your attribute inputs.".format(attr))
            sys.exit(0)


def select_rows(df, attributes):
    df.columns = map(str.lower, df.columns)
    df = df.dropna(axis=0, subset=attributes, how='all')
    return df


def main():
    argparser = ArgumentParser(description='Get rows if any of the specified attribute is not null')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to original csv file. e.g. csv/beauty_data_info_train_competition.csv')
    argparser.add_argument('-a', '--attributes', type=str, nargs='+', required=True, help='Attributes selected (not case sensitive), if there is a space, use "", e.g. "Warranty Period" "Memory RAM"')
    A = argparser.parse_args()

    df = pd.read_csv(A.file)
    check_attributes(df, A.attributes)
    new_df = select_rows(df, A.attributes)
    new_df.to_csv(A.file.rsplit('.', 1)[0] + '_selected.csv')


if __name__ == '__main__':
    main()
