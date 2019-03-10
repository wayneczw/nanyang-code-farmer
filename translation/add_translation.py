import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import os


def argParse():
    argparser = ArgumentParser(description='Run ocr on images in csv file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    argparser.add_argument('-t', '--text', type=str, required=True, help='Path to text file.')
    argparser.add_argument('-a', '--additional_found', type=str, help='Path to manually translate text file.')

    A = argparser.parse_args()
    return A


def find_again(index_lst, titles, A):
    """ creates the text files for rows to be done by manual translation """
    filename = A.text.rsplit('.', 1)[0] + '_to_find.txt'
    with open(filename, 'w+') as f:
        for item in index_lst:
            f.write('[{}] {}\r'.format(item, titles[item]))

    print("Please refer to to_find.txt to manually translate the selected titles.")


def preprocess(id_lst, titles, text):

    lst = text.split('[')[1:]   # first one empty
    new_lst = [item.split(']') for item in lst]   # [['3', 'text'], ['5', 'text'], ...]
    for i in new_lst:
        i[0] = i[0].strip().replace(' ', '').replace(',', '')

    # examine any unclean translated title
    print("unclean number (if any):")
    for w in new_lst:
        if not w[0].isdigit():
            print(w)
    print("[end of unclean number]")

    clean_dict = dict()
    for item in new_lst:
        if item[0].isdigit():
            clean_dict[int(item[0])] = item[1].strip().lower()
    diff = set(id_lst) - (clean_dict.keys())

    # make sure the number of rows tally
    assert len(id_lst) == len(clean_dict.keys()) + len(diff) - len(clean_dict.keys() - set(id_lst))

    return diff, clean_dict


def add_manually_found(manual_text, clean_dict):
    with open(manual_text, 'r') as text_file:
        text = text_file.read()
    lst = text.split('\n')
    for item in lst:
        if item != '':
            a, b = item.split(']')
            num = int(a[1:])
            clean_dict[num] = b.strip().lower()

    return clean_dict


def main():

    A = argParse()

    df = pd.read_csv(A.file)
    langs = df['lang']
    titles = df['title']
    id_lst = [i for i in range(len(langs)) if langs[i] == 'id']
    total_row = len(id_lst)
    print("Total {} rows in malay".format(total_row))

    with open(A.text, 'r') as text_file:
        text = text_file.read()

    diff, clean_dict = preprocess(id_lst, titles, text)
    print("row to find again:", diff)

    # find_again(diff, titles, A)

    clean_dict = add_manually_found(A.additional_found, clean_dict)
    translated = []

    with tqdm(total=len(langs)) as pbar:
        for i, lang in enumerate(langs):
            if lang == 'id':
                translated.append(clean_dict[i])
            else:
                translated.append(None)
            pbar.update(1)

    df['translated'] = pd.Series(translated)
    print(df)

    new_file = A.file.rsplit('.', 1)[0] + '_translated.csv'
    df.to_csv(new_file, index=False)
    print('translated csv: ' + new_file)


if __name__ == '__main__':
    main()
