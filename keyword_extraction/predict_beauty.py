from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
from prettify_json import read_json
from tqdm import tqdm
import os
import re
from collections import Counter
import random


SPELLING_CORRECTION_DICT = {}


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    argparser.add_argument('-a', '--attribute_mapping', type=str, required=True, help='Path to attribute mapping (profile) json file')
    argparser.add_argument('-t', '--translation_mapping', type=str, required=True, help='Path to translation mapping json file')
    argparser.add_argument('-p', '--predicting', action='store_true', help='If it is making prediction without label')

    A = argparser.parse_args()
    return A


def concat_title_translate(df):
    """ concat title and translate in one col, return new_col of concated str """
    print("Creating concatenated col ...")
    new_col = df[['title', 'translated']].fillna('').apply(lambda x: ' '.join(x), axis=1)
    return new_col


def spelling_correction(content, spelling_correction_dict):
    for word, corrected in spelling_correction_dict.items():
        if word in content:
            content.replace(word, corrected)
    return content


def language_correction(content, language_correction_dict):
    for word_id, word_en in language_correction_dict.items():
        if word_id in content:
            content = content.replace(word_id, word_en)
    return content


def cleaning(col, language_correction_dict, spelling_correction_dict=SPELLING_CORRECTION_DICT):
    """ clean the column with spelling and language correction """
    print("Cleaning concatenated col ... ")

    with tqdm(total=len(col)) as pbar:
        for i, content in enumerate(col):
            content = spelling_correction(content, spelling_correction_dict)
            content = language_correction(content, language_correction_dict)
            col[i] = content
            # print(col[i])
            pbar.update(1)

    return col


def find_exact_model(src, attr_dict):
    """ src: a title (one row)
        attr_dict: corrected attrs:num dictionary for a single attr
        return matching_result (name) and matching_num_result (coded number) as list
    """
    concat_src = src.replace(' ', '')
    matching_result = set()
    matching_num_result = set()
    for attr, num in attr_dict.items():
        if attr.replace(' ', '') in concat_src:
            matching_result.add(attr)
            matching_num_result.add(int(num))
    return list(matching_result), list(matching_num_result)


def find_exact(src, attr_dict):
    """ src: a title (one row)
        attr_dict: corrected attrs:num dictionary for a single attr
        return matching_result (in coded number) as list
    """
    matching_result = set()
    for attr, num in attr_dict.items():
        if attr in src:
            matching_result.add(int(num))
    return list(matching_result)


def predict(cleaned_col, attribute_mapping, df, predicting=False):
    total_row = len(cleaned_col)

    predictions = dict()
    for attr in attribute_mapping.keys():
        predictions[attr] = []
        attr_dict = attribute_mapping[attr]
        print("Predicting {} ... ".format(attr))
        with tqdm(total=total_row) as pbar:
            for item in cleaned_col:
                predictions[attr].append(find_exact(item, attr_dict))
                pbar.update(1)

    # if < 2 prediction, use NN model output
    if predicting:
        predictions = add_nn_output(predictions, df)

    return predictions


def add_nn_output(prediction, nn_output):

    def internal_add_nn_output(pred, nn_pred):
        # print(pred, nn_pred)
        nn_pred = list(map(int, nn_pred.split()))
        if len(pred) == 0:
            pred = nn_pred
        elif len(pred) == 1:
            for p in nn_pred:
                if p not in pred:
                    pred.append(p)
                    break
        else:
            new_pred = []
            for p in nn_pred:
                if p in pred:
                    new_pred.append(p)
            while len(new_pred) < 2:
                for p in nn_pred:
                    if p not in pred:
                        new_pred.append(p)

            pred = new_pred
        if len(pred) == 1:
            pred.append(pred[0])
        return pred

    print("Adding output from NN model ... ")
    total_row = nn_output.shape[0]
    with tqdm(total=total_row) as pbar:
        for i in range(total_row):
            for attr, pred in prediction.items():
                prediction[attr][i] = internal_add_nn_output(pred[i], nn_output.ix[i, attr])

                # print('final:', prediction[attr][i])
                # if len(prediction[attr][i]) < 2:
                #     print(attr, i, prediction[attr][i], pred[i], nn_output.ix[i, attr])
                #     input()
                # there should be exactly two predictions
                
                assert len(prediction[attr][i]) == 2
            pbar.update(1)
    return prediction


def post_process(predicted_brand, predicted_model, other_predictions, branddict, modeldict):

    print('Post processing using brand and phone model ... ')
    with tqdm(total=len(predicted_brand)) as pbar:
        for i in range(len(predicted_brand)):
            round2 = False
            # if phone model is not null, use it, otherwise, use brand
            if len(predicted_model[i]) > 0:
                for attr in other_predictions.keys():
                    other_predictions[attr][i] = cross_reference(other_predictions[attr][i], attr, predicted_model[i], modeldict)
                    round2 = True
            if len(predicted_brand[i]) > 0:
                for attr in other_predictions.keys():
                    other_predictions[attr][i] = cross_reference(other_predictions[attr][i], attr, predicted_brand[i], branddict, round2)

            pbar.update(1)

    return other_predictions


def sort_model(model):
    # if more than 1 phone model, take the longest and second longest
    if len(model) > 1:
        model.sort(key=len)
        model.reverse()
        model = model[:2]
    return model


def cross_reference(attr, attr_name, model, modeldict, round2=False):

    # if model is not null, use the longest matched model to cross reference brand
    if len(model) > 0:
        longest_model = model[0]
        longest_model_dict = modeldict[str(longest_model)]
        counter = Counter(longest_model_dict[attr_name])
        most_likely_two = counter.most_common()

        # if no brand, get the most common brand
        if len(attr) == 0:
            try:
                attr.append(int(float(most_likely_two[0][0])))
                attr.append(int(float(most_likely_two[1][0])))
            except IndexError:
                pass

        elif len(attr) == 1:
            try:
                if attr[0] == int(float(most_likely_two[0][0])):
                    attr.append(int(float(most_likely_two[1][0])))
                else:
                    attr.append(int(float(most_likely_two[0][0])))
            except IndexError:
                pass

        elif len(attr) > 1 and round2 is False:
            updated_attr = []
            for i in range(len(most_likely_two)):
                if int(float(most_likely_two[i][0])) in attr:
                    updated_attr.append(int(float(most_likely_two[i][0])))
                    attr.remove(int(float(most_likely_two[i][0])))

            while len(updated_attr) < 2 and len(attr) > 0:
                # randomly select one from the brand and add in hahahahaha
                chosen = random.choice(attr)
                updated_attr.append(int(float(chosen)))
                attr.remove(int(float(chosen)))

            attr = updated_attr

    return attr


def main():
    A = argParse()

    df = pd.read_csv(A.file)
    # df = df[:20]
    translation_mapping = read_json(A.translation_mapping)
    attribute_mapping = read_json(A.attribute_mapping)

    ## 1. cleaning for beauty attribute mapping

    ## 2. concate and clean title
    concat_col = concat_title_translate(df)
    cleaned_col = cleaning(concat_col, translation_mapping)
    df['clean_title'] = cleaned_col

    ## 3. make prediction
    prediction = predict(cleaned_col, attribute_mapping, df, A.predicting)

    ## 4. write prediction to df
    new_file_name = '../data/beauty_test_keywords.csv'
    new_keys = []
    for k, v in prediction.items():
        # if A.predicting is True:
        #     new_key = k
        # else:
        #     new_key = "Keyword " + k

        new_key = "Keyword " + k
        new_keys.append(new_key)
        df[new_key] = [' '.join(list(map(str, lst))) for lst in v]

    new_keys += ["itemid", "clean_title", "image_path", "title", "translated"]
    df = df[new_keys]
    df.to_csv(new_file_name, index=False)

if __name__ == '__main__':
    main()
