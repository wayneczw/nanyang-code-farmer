import pandas as pd
from argparse import ArgumentParser
from googletrans import Translator
from tqdm import tqdm
import os


def argParse():
    argparser = ArgumentParser(description='Run translation')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    # argparser.add_argument('-i', '--image_folder', type=str, default="", help='Path to folder which contains the x_image folder')
    argparser.add_argument('-s', '--start_row', type=int, default=0, help='Starting row allocated')
    argparser.add_argument('-e', '--end_row', type=int, help='Ending row allocated (inclusive)')
    argparser.add_argument('-i', '--increment', type=int, default=4950, help='increment when sending to translate')
    A = argparser.parse_args()
    return A


def main():
    A = argParse()
    f = A.file
    df = pd.read_csv(f)
    # print(df.columns)
    # print(df['lang'])

    msg = ""
    langs = df['lang']
    titles = df['title']
    print(df.shape)

    if A.end_row is None:
        A.end_row = df.shape[0] - 1

    row_count = 0
    for i in range(A.start_row, A.end_row):
        if langs[i] == 'id':
            msg += '[' + str(i) + '] '
            row_count += 1
            msg += titles[i]
            msg += '.\n'

    print('Total malay rows: ', row_count)
    print('Total number of characters:', len(msg))
    print('First 200 characters:\n' + msg[:200])

    dirname, basename = os.path.split(A.file)
    text_file_name = basename.rsplit('.', 1)[0] + '_translated_{}_{}.txt'.format(A.start_row, A.end_row)
    text_file_name = os.path.join('textfiles', text_file_name)
    print('Writing translated text to: ' + text_file_name)

    translator = Translator()

    send_translation(msg, text_file_name, A.increment)


def send_translation(msg, filename, inc=4950):
    translator = Translator()
    count = 0

    with tqdm(total=len(msg)) as pbar:
        with open(filename, 'wb+') as text_file:
            while count < len(msg):
                end_count = count + inc
                if end_count > len(msg):
                    end_count = len(msg)
                else:
                    while not msg[end_count].isspace():
                        end_count += 1
                translated = translator.translate(msg[count:end_count], src="id", dest='en')
                text_file.write(translated.text.encode('utf-8'))
                pbar.update(end_count - count)
                count = end_count


if __name__ == '__main__':
    main()
