from argparse import ArgumentParser
import pandas as pd
import numpy as np
from keyword_extraction.prettify_json import read_json, write_json
from collections import Counter
import random


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to training csv file.')
    argparser.add_argument('-a', '--attribute_mapping', type=str, required=True, help='Path to attribute mapping (profile) json file')
    argparser.add_argument('-o', '--output', type=str, required=True, help='Path to output json file with extension')

    A = argparser.parse_args()
    return A


def count_majority():
    A = argParse()
    df = pd.read_csv(A.file)
    count_dict = dict()
    attribute_mapping = read_json(A.attribute_mapping)

    for attr in attribute_mapping.keys():
        col = list(df[attr].dropna())
        count_dict[attr] = Counter(col)

    write_json(A.output, count_dict)
    print('writing to {}...'.format(A.output))

if __name__ == '__main__':
    count_majority()
