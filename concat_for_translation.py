import pandas as pd
from argparse import ArgumentParser
from googletrans import Translator
from tqdm import tqdm


def argParse():
    argparser = ArgumentParser(description='Run translation')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to csv file.')
    # argparser.add_argument('-i', '--image_folder', type=str, default="", help='Path to folder which contains the x_image folder')
    # argparser.add_argument('-s', '--start_row', type=int, default=0, help='Starting row allocated')
    # argparser.add_argument('-e', '--end_row', type=int, help='Ending row allocated (inclusive)')
    A = argparser.parse_args()
    return A


def main():
    A = argParse()
    f = A.file
    df = pd.read_csv(f)
    # print(df.columns)
    # print(df['lang'])

    translator = Translator()

    msg = ""
    langs = df['lang']
    titles = df['title']
    print(df.shape)

    row_count = 0
    for i in range(df.shape[0]):
    # for i in range(100):
        if langs[i] == 'id':
            msg += '[' + str(i) + ']'
            row_count += 1
            msg += titles[i]
    print(row_count)
    print(len(msg))

    # print(msg)
    # translated = translator.translate(msg)
    # print(translated)
    full_translated = ""

    count = 0
    with tqdm(total=len(msg)) as pbar:
        with open('translated.txt', 'a') as text_file:
            while count < len(msg):
                translated = translator.translate(msg[count:count + 5000], src="id", dest='en')
                text_file.write(translated.text)
                pbar.update(5000)
                count += 5000
            translated = translator.translate(msg[count:], src="id", dest='en')
            text_file.write(translated.text)
            pbar.update(len(msg[count:]))



if __name__ == '__main__':
    main()
