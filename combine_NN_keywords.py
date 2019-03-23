from argparse import ArgumentParser
import pandas as pd
import numpy as np
from keyword_extraction.prettify_json import read_json
from collections import Counter
import random
from tqdm import tqdm
import os


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-p', '--proba', type=str, required=True, help='Path to proba csv file.')
    argparser.add_argument('-k', '--keywords', type=str, required=True, help='Path to keywords csv file')
    argparser.add_argument('-a', '--attribute_mapping', type=str, required=True, help='Path to attribute mapping (profile) json file')
    argparser.add_argument('-c', '--majority_count', type=str, required=True, help='Path to majority count json file')
    argparser.add_argument('-o', '--overwrite', type=str, nargs='+', help='Attribute(s) to overwrite prediction from NN')
    argparser.add_argument('-b', '--branddict', type=str, help='Path to Brand_to_others.json')
    argparser.add_argument('-m', '--modeldict', type=str, help='Path to Phone_model_to_others.json')

    A = argparser.parse_args()
    return A


def internal_combine_with_overwriting(kw_pred, nn_pred):
    pred = []
    kw_pred = list(set(map(int, map(float, kw_pred.split()))))
    nn_pred = list(map(int, nn_pred.split()))
    if len(kw_pred) == 0:
        pred = nn_pred
    elif len(kw_pred) == 1:
        pred.append(kw_pred[0])
        # append the other one from nn_pred (if not duplicate)
        for p in nn_pred:
            if p not in pred:
                pred.append(p)
                break
    else:
        pred = kw_pred[:2]
    return ' '.join(list(map(str, pred)))


def combine_with_overwriting(kw_col, nn_col, attr):
    print("Combine with overwriting " + attr + ' ...')
    kw_col.fillna('', inplace=True)
    kw_col = kw_col.astype(str)
    out = []
    with tqdm(total=len(nn_col)) as pbar:
        for i in range(len(nn_col)):
            out.append(internal_combine_with_overwriting(kw_col[i], nn_col[i]))
            pbar.update(1)
    return out


def resolve_dup_mobile(combine, attributes, majority_count, branddict, modeldict):
    total_row = len(combine[attributes[0]])
    with tqdm(total=total_row) as pbar:
        for i in range(total_row):
            temp_brand = combine["Brand"][i].split()[0]      # str
            temp_model = combine["Phone Model"][i].split()[0]        # str
            for attr in attributes:
                pred_str = combine.ix[i, attr]
                pred = list(map(int, pred_str.split()))
                if len(set(pred)) == 1:
                    print('duplicate found in row {} attr {} :\n{}'.format(i, attr, combine.iloc[i]))
                    model_counter = Counter(modeldict[temp_model][attr])
                    brand_counter = Counter(modeldict[temp_brand][attr])
                    pred = internal_resolve_dup_mobile([pred[0]], attr, model_counter, brand_counter, Counter(majority_count[attr]))
                    combine.ix[i, attr] = ' '.join(list(map(str, pred)))
                assert len(pred) == 2
            pbar.update(1)

    return combine


def internal_resolve_dup_mobile(pred, attr, model_counter, brand_counter, majority_counter):
    # use phone model first
    if len(model_counter) > 0:
        pred = internal_resolve_dup(pred, attr, model_counter)
    if len(pred) == 1 and len(brand_counter) > 0:
        pred = internal_resolve_dup(pred, attr, brand_counter)
    if len(pred) == 1:
        pred = internal_resolve_dup(pred, attr, majority_counter)
    return pred


def resolve_dup(combine, attributes, majority_count):
    total_row = len(combine[attributes[0]])
    with tqdm(total=total_row) as pbar:
        for i in range(total_row):
            for attr in attributes:
                pred_str = combine.ix[i, attr]
                pred = list(map(int, pred_str.split()))
                if len(set(pred)) == 1:
                    print('duplicate found in rwo {} attr {} :\n{}'.format(i, attr, combine.iloc[i]))
                    pred = internal_resolve_dup([pred[0]], attr, Counter(majority_count[attr]))
                    combine.ix[i, attr] = ' '.join(list(map(str, pred)))
                assert len(pred) == 2
            pbar.update(1)

    return combine


def internal_resolve_dup(pred, attr, majority_counter):
    for name, count in majority_counter.most_common():
        if int(float(name)) not in pred:
            pred.append(name)
            break
    return pred


def main():

    A = argParse()
    dfp = pd.read_csv(A.proba)
    dfk = pd.read_csv(A.keywords)

    # two df should have the same number of rows
    assert dfp.shape[0] == dfk.shape[0]

    combine = dfp[['itemid', 'title', 'image_path']].copy()
    attribute_mapping = read_json(A.attribute_mapping)
    attributes = list(attribute_mapping.keys())
    attributes_keywords = [('Keyword ' + attr) for attr in attributes]

    for attr in attributes:
        if A.overwrite is not None and attr in A.overwrite:
            out = combine_with_overwriting(dfk["Keyword " + attr], dfp[attr], attr)
            combine[attr] = out
        else:
            combine[attr] = list(dfp[attr])

    dirname, basename = os.path.split(A.keywords)
    # resolve duplicate
    majority_count = read_json(A.majority_count)
    if basename.startswith('mobile'):
        branddict = read_json(A.branddict)
        modeldict = read_json(A.modeldict)
        combine = resolve_dup_mobile(combine, attributes, majority_count, branddict, modeldict)
    else:
        combine = resolve_dup(combine, attributes, majority_count)

    filename = 'data/{}_combine.csv'.format(basename.split('_')[0])
    print("Writing prediction to {} ... ".format(filename))
    for k, v in combine.items():
        combine[k] = list(v)
    combine.to_csv(filename, index=False)

    return


if __name__ == '__main__':
    main()
