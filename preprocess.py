import logging
import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
from translate import Translator
from langdetect import detect
from quantulum import parser
import multiprocessing as mp
from multiprocessing import cpu_count
from langdetect.detector_factory import init_factory

logger = logging.getLogger(__name__)

raw_categorical_targets = [
    'Benefits', 'Brand',
    'Colour_group', 'Product_texture',
    'Skin_type']

categorical_targets = [
    'Benefits', 'Brand',
    'Colour_group', 'Product_texture',
    'Skin_type']

categorical_features = [
    'nouns', 'numbers']

ncores = cpu_count()

NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
chunker = RegexpParser(NP)


def get_continuous_chunks(text):
    chunked = chunker.parse(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append("-".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = "-".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return ' '.join(continuous_chunk) if continuous_chunk else np.nan
#end def


def get_lang(text):
    try:
        return detect(text)
    except Exception:
        return np.nan
#end def


def to_en(text):
    translator = Translator(to_lang="en", from_lang='ms', provider='mymemory')
    return translator.translate(text)
#end def


def get_num(text):
    qty = parser.parse(text)
    qty = ['/'.join([str(q.value), str(q.unit.name)]) for q in qty]
    return ' '.join(qty) if qty else np.nan
#end def


def read(df_path, quick=False):
    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)
    if quick: df = df[:1024]

    # Lang Detection
    logger.info("Detecting language....")
    with mp.Pool(ncores) as pool:
        langs = pool.imap(get_lang, df['title'], chunksize=10)
        langs = [lang for lang in langs]
    #end with
    df['lang'] = pd.Series(langs)
    #end try
    logger.info("Done detecting language....")

    # # Translation
    # logger.info('Translating....')
    # df['title'] = df.apply(lambda x: to_en(x.title) if x.lang == 'id' else x.title, axis=1)
    # logger.info("Done translating....")

    # Get number/unit
    logger.info("Extracting number/unit....")
    try:
        with mp.Pool(ncores) as pool:
            numbers = pool.imap(get_num, df['title'], chunksize=10)
            numbers = [num for num in numbers]
        #end with
        df['numbers'] = pd.Series(numbers)
    except UnboundLocalError:
        df['numbers'] = df['title'].apply(lambda text: get_num(text))
    #end try

    logger.info("Done extracting number/unit....")

    # Get NP
    logger.info("Extracting noun phrases....")
    with mp.Pool(ncores) as pool:
        nouns = pool.imap(get_continuous_chunks, df['title'], chunksize=10)
        nouns = [noun for noun in nouns]
    #end with
    df['nouns'] = pd.Series(nouns)
    # df['nouns'] = df['title'].apply(lambda sent: get_continuous_chunks(sent))
    logger.info("Done extracting noun phrases....")

    df[categorical_features] = df[categorical_features].fillna('unk')

    logger.info("Done reading in {} data....".format(df.shape[0]))

    return df
#end def


def main():
    argparser = ArgumentParser(description='Run machine learning experiment.')
    argparser.add_argument('-f', '--files', type=str, nargs='+', metavar='<train_set>', required=True, help='Training data set.')
    argparser.add_argument('--seed', type=int, default=0, help='Random seed.')
    A = argparser.parse_args()

    log_level = 'INFO'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    quick = False

    for f in A.files:
        df = read(f, quick=quick)
        df.to_csv(f.split('.csv')[0] + '_processed.csv', index=False)
#end def

if __name__ == '__main__': main()
