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


SPELLING_CORRECTION_DICT = {
    'xiomi': 'xiaomi',
    'samaung': 'samsung',
    'galaxi': 'galaxy'
}


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    argparser.add_argument('-a', '--attribute_mapping', type=str, required=True, help='Path to attribute mapping (profile) json file')
    argparser.add_argument('-t', '--translation_mapping', type=str, required=True, help='Path to translation mapping json file')
    argparser.add_argument('-b', '--branddict', type=str, required=True, help='Path to Brand_to_others.json')
    argparser.add_argument('-m', '--modeldict', type=str, required=True, help='Path to Phone_model_to_others.json')
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


def add_apple(content):
    """ add "apple" before the word "iphone" / "ipad" / "macbook"
        (if there is one and prev word is not apple)
        for exact matching of brand and model """
    content_lst = content.split()
    new_lst = []
    for i, word in enumerate(content_lst):
        if ("iphone" in word or "ipad" in word or "macbook" in word) and (content_lst[i - 1] != "apple"):
            new_lst.append("apple")
        new_lst.append(word)

    return ' '.join(new_lst)


def add_samsung(content):
    """ add "samsung" before the word "galaxy"
        (if there is one and prev word is not samsung)
        for exact matching of brand and model """
    content_lst = content.split()
    new_lst = []
    for i, word in enumerate(content_lst):
        if ("galaxy" in word) and (content_lst[i - 1] != "samsung"):
            new_lst.append("samsung")
        new_lst.append(word)

    return ' '.join(new_lst)


def remove_space_before_unit(content):
    content = content + '.'

    def remove_space(m):
        return m.group(1) + m.group(2) + m.group(3)

    content = re.sub(r"(\d+\.?\d?)\s+([gm]b|mp|g)([\s\.])", remove_space, content)
    return content[:-1]


def cleaning(col, language_correction_dict, spelling_correction_dict=SPELLING_CORRECTION_DICT):
    """ clean the column with spelling and language correction """
    print("Cleaning concatenated col ... ")

    with tqdm(total=len(col)) as pbar:
        for i, content in enumerate(col):
            content = spelling_correction(content, spelling_correction_dict)
            content = language_correction(content, language_correction_dict)
            content = remove_space_before_unit(content)
            content = add_apple(content)
            content = add_samsung(content)
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


def find_phone_screen_size(src, screen_size_dict):
    src = src + '.'
    # find floating points number followed by space or fullstop
    res1 = re.findall(r"(\d+\.\d+)[\s\.]", src)
    # find number followed by "inch"
    res2 = re.findall(r"(\d+\.?\d+?)\s?inch", src)
    res = list(set(map(float, res1 + res2)))

    matching_result = []
    for k, v in screen_size_dict.items():
        for r in res:
            if r >= k[0] and r <= k[1]:
                matching_result.append(v)
    return matching_result


def predict(cleaned_col, attribute_mapping, branddict, modeldict, df, predicting=False):
    total_row = len(cleaned_col)

    # 1. predict brand
    predicted_brand = []
    print("Predicting brand ... ")
    with tqdm(total=total_row) as pbar:
        for item in cleaned_col:
            predicted_brand.append(find_exact(item, attribute_mapping['Brand']))
            pbar.update(1)

    # 2. predict phone model
    predicted_model = []
    print("Predicting model ... ")
    with tqdm(total=total_row) as pbar:
        for item in cleaned_col:
            temp_model, _ = find_exact_model(item, attribute_mapping['Phone Model'])
            temp_model = sort_model(temp_model)
            temp_model_num = [attribute_mapping['Phone Model'][m] for m in temp_model]
            predicted_model.append(temp_model_num)
            pbar.update(1)

    assert len(predicted_brand) == len(predicted_model) == total_row
    # 3. cross reference brand and model
    print("Cross referencing brand and model ...")
    with tqdm(total=total_row) as pbar:
        for i in range(total_row):
            predicted_brand[i] = cross_reference(predicted_brand[i], "Brand", predicted_model[i], modeldict)
            pbar.update(1)

    # 4. special finding for phone screen size
    screen_size_dict = dict()
    for k, v in attribute_mapping['Phone Screen Size'].items():
        lower_range, upper_range = k.split()
        screen_size_dict[(float(lower_range), float(upper_range))] = int(v)
    predicted_screen_size = []
    print("Predicting Phone Screen Size ... ")
    with tqdm(total=total_row) as pbar:
        for item in cleaned_col:
            predicted_screen_size.append(find_phone_screen_size(item, screen_size_dict))
            pbar.update(1)

    # 5. predict the rest of model attributes
    other_predictions = dict()
    for attr in attribute_mapping.keys():
        if attr not in ['Brand', 'Phone Model', 'Phone Screen Size']:
            other_predictions[attr] = []
            attr_dict = attribute_mapping[attr]
            print("Predicting {} ... ".format(attr))
            with tqdm(total=total_row) as pbar:
                for item in cleaned_col:
                    other_predictions[attr].append(find_exact(item, attr_dict))
                    pbar.update(1)

    other_predictions['Phone Screen Size'] = predicted_screen_size

    # 6. post processing using phone model and brand
    # other_predictions = post_process(predicted_brand, predicted_model, other_predictions, branddict, modeldict)

    prediction = other_predictions
    prediction['Brand'] = predicted_brand
    prediction['Phone Model'] = predicted_model

    # 7. if < 2 prediction, use NN model output
    if predicting:
        prediction = add_nn_output(prediction, df)

    return prediction


def add_nn_output(prediction, nn_output):

    def internal_add_nn_output(pred, nn_pred):
        nn_pred = list(map(int, nn_pred.split()))
        if len(pred) == 0:
            pred = nn_pred
        elif len(pred) == 1:
            for p in nn_pred:
                if p not in pred:
                    pred.append(p)
                    break

        return pred

    print("Adding output from NN model ... ")
    total_row = nn_output.shape[0]
    with tqdm(total=total_row) as pbar:
        for i in range(total_row):
            for attr, pred in prediction.items():
                if len(pred[i]) < 2:
                    prediction[attr][i] = internal_add_nn_output(pred[i], nn_output.ix[i, attr])

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

        # # Leave it to the model
        # elif len(attr) == 1:
        #     try:
        #         if attr[0] == int(float(most_likely_two[0][0])):
        #             attr.append(int(float(most_likely_two[1][0])))
        #         else:
        #             attr.append(int(float(most_likely_two[0][0])))
        #     except IndexError:
        #         pass

        elif len(attr) > 1 and round2 is False:
            updated_attr = []
            for i in range(len(most_likely_two)):
                if int(float(most_likely_two[i][0])) in attr:
                    updated_attr.append(int(float(most_likely_two[i][0])))
                    attr.remove(int(float(most_likely_two[i][0])))

            # Leave it to the model
            # while len(updated_attr) < 2 and len(attr) > 0:
            #     # randomly select one from the brand and add in 
            #     chosen = random.choice(attr)
            #     updated_attr.append(int(float(chosen)))
            #     attr.remove(int(float(chosen)))

            attr = updated_attr

    return attr[:2]


def main():
    A = argParse()

    df = pd.read_csv(A.file)
    # df = df[:20]
    translation_mapping = read_json(A.translation_mapping)
    attribute_mapping = read_json(A.attribute_mapping)
    branddict = read_json(A.branddict)
    modeldict = read_json(A.modeldict)

    ## 1. cleaning for mobile attribute mapping
    # (1) remove space between number and unit in Camera
    camera_attr = attribute_mapping['Camera']
    clean_camera_attr = dict()
    for k, v in camera_attr.items():
        if k[0].isdigit():
            new_key = k.replace(' ', '')
        else:
            new_key = k
        clean_camera_attr[new_key] = v
    attribute_mapping['Camera'] = clean_camera_attr
    # print(attribute_mapping['Camera'])

    #  (2) process Phone Screen Size
    screen_size = attribute_mapping['Phone Screen Size']
    clean_screen_size = dict()
    for k, v in screen_size.items():
        new_key = []
        if "less" in k:
            new_key.append(str(0))
        key_list = k.split()
        for word in key_list:
            try:
                temp = float(word)
                new_key.append(str(temp))
            except:
                continue
        if "more" in k:
            new_key.append(str(100))
        assert len(new_key) == 2
        clean_screen_size[' '.join(new_key)] = v
    attribute_mapping['Phone Screen Size'] = clean_screen_size

    # (3) process Warranty Period
    warranty_period = attribute_mapping['Warranty Period']
    clean_warranty_period = dict()
    for k, v in warranty_period.items():
        if k.endswith('s'):
            new_key = k[:-1]
        else:
            new_key = k
        clean_warranty_period[new_key] = v
    attribute_mapping['Warranty Period'] = clean_warranty_period
    print(attribute_mapping['Warranty Period'])

    ## 2. concate and clean title
    concat_col = concat_title_translate(df)
    cleaned_col = cleaning(concat_col, translation_mapping)

    ## 3. make prediction
    prediction = predict(cleaned_col, attribute_mapping, branddict, modeldict, df, A.predicting)

    ## 4. write prediction to df
    for k, v in prediction.items():
        if A.predicting is True:
            new_key = k
            new_file_name = '../data/mobile_test_predicted.csv'
        else:
            new_key = "Predicted " + k
            new_file_name = 'mobile_predicted.csv'

        df[new_key] = [' '.join(list(map(str, lst))) for lst in v]

    df.to_csv(new_file_name, index=False)

if __name__ == '__main__':
    main()
