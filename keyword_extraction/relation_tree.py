from argparse import ArgumentParser
from prettify_json import read_json, write_json
import pandas as pd
import numpy as np
import math
from collections import Counter
from tqdm import tqdm


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    argparser.add_argument('-j', '--json', type=str, required=True, help='Path to json file.')
    argparser.add_argument('-k', '--keyname', type=str, required=True, help='Name of the column.')
    A = argparser.parse_args()
    return A


def init_res(jsondict, keyname):
    categories = list(jsondict.keys())
    categories.remove(keyname)
    res = dict()
    possible_values = jsondict[keyname].values()
    for val in possible_values:
        res[val] = dict()
        for cat in categories:
            res[val][cat] = list()  # list to store all possible values, later convert to counter

    return res


def post_process(res):

    ''' convert the list to counter_dict '''

    print('post_process...')
    with tqdm(total=len(res.keys())) as pbar:
        for keyname, keyvalue in res.items():
            for cat, catlist in keyvalue.items():
                res[keyname][cat] = Counter(catlist)
            pbar.update(1)

    return res


def main():
    A = argParse()
    mapping = read_json(A.json)
    df = pd.read_csv(A.file)
    keyname = A.keyname

    create_tree(keyname, mapping, df)


def create_tree(keyname, mapping, df):
    """ return model to brand mapping dict {model_number: {brand number}} """

    categories = list(mapping.keys())
    categories.remove(keyname)

    res = init_res(mapping, keyname)

    total_row = df.shape[0]

    print('appending possible attr value...')

    with tqdm(total=total_row) as pbar:

        for i in range(total_row):
            keyvalue = df.ix[i, keyname]

            if not np.isnan(keyvalue):
                for cat in categories:
                    attr_value = df.ix[i, cat]
                    if not np.isnan(attr_value):
                        res[keyvalue][cat].append(attr_value)
            pbar.update(1)

    res = post_process(res)

    filename = "{}_to_others.json".format(keyname.replace(' ', '_'))
    print('writing to {}...'.format(filename))
    write_json(filename, res)


if __name__ == '__main__':
    main()
