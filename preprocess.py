import logging
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from nltk import word_tokenize, pos_tag
from nltk import RegexpParser
from nltk import Tree
from translate import Translator
from langdetect import detect
from quantulum import parser
import multiprocessing as mp
from multiprocessing import cpu_count
from nltk.stem.snowball import SnowballStemmer
import spacy
import pylab as pl
from collections import Counter

nlp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer("english")

logger = logging.getLogger(__name__)

categorical_features = [
    'lang', 'nouns', 'numbers', 'adj']

ncores = cpu_count()

NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
chunker = RegexpParser(NP)

#mobile
# Keywords = ["operating system", "operating", "system", "os", "symbian", "windows", "samsung", "blackberry", "nokia", "android", "ios", "features", "expandable memory", "touchscreen", "fingerprint sensor", "dustproof", "waterproof", "wifi", "gps", "network", "connections", "4g", "2g", "3g", "3.5g", "memory", "ram", "4gb", "2gb", "1.5gb", "16gb", "512mb", "8gb", "3gb", "10gb", "1gb", "6gb", "warranty", "period", "yr", "mth", "year", "month", "years", "months", "7 months", "4 months", "6 months", "3 months", "10 years", "2 month", "11 months", "10 months", "5 months", "3 years", "2 years", "1 month", "18 months", "1 year", "storage", "capacity", "gb", "mb", "256gb", "1.5gb", "128gb", "512mb", "64gb", "512gb", "8gb", "4mb", "6gb", "4gb", "2gb", "128mb", "32gb", "256mb", "10gb", "3gb", "1gb", "16gb", "color", "colour", "family", "blue", "gold", "brown", "navy blue", "yellow", "neutral", "rose gold", "light blue", "dark grey", "silver", "pink", "gray", "army", "green", "army green", "deep", "blue", "deep blue", "purple", "rose", "light", "grey", "light grey", "black", "deep black", "off", "white", "off white", "multicolor", "black", "apricot", "orange", "red", "camera", "mp", "single", "42mp", "dua slot", "5 mp", "3 mp", "1 mp", "8 mp", "single camera", "24 mp", "16mp", "13mp", "6 mp", "10mp", "2 mp", "20 mp", "4 mp", "phone", "screen", "size", "inch", "inches", "4.6 to 5 inches", "4.1 to 4.5 inches", "less than 3.5 inches", "3.6 to 4 inches", "more than 5.6 inches", "5.1 to 5.5 inches"]

#beauty
# Keywords = ["benefits", "high", "pigmentation", "natural", "light", "hydrating", "durable", "oil", "control", "spf", "colour", "group", "emas", "rose", "1 warna", "9 color", "emas", "hijau", "warna", "merah", "cabai", "warna merah cabai", "8 color", "5 color", "multiwarna", "perak", "krem", "peach", "coklat tua", "peanut", "biru", "ungu", "hitam", "abu", "6 color", "mawar", "4 color", "putih", "warna blush pink", "bening", "netral", "maroon", "kuning", "11 color", "merah", "warna koral", "3 color", "nude", "12 color", "ceri", "2 color", "warna fuchsia", "merah muda", "warna hotpink", "10 color", "merah semangka", "jeruk", "sawo matang", "7 color", "cokelat", "antique white", "product", "texture", "balm", "stick", "liquid", "crayon pensiln", "formula mousse", "cream", "solid", "powder", "solid powder", "cushion", "gel", "skin", "type", "dry", "sensitive", "fade", "combination", "normal", "aging", "age", "signs of aging", "acne", "greasy"]

#fashion
# Keywords = ["pattern", "paisley", "plaid", "threadwork", "patchwork", "plain", "graphic", "print", "gingham", "camouflage", "polka", "dot", "polka dot", "joint", "wave", "point", "wave point", "stripe", "knot", "floral", "brocade", "cartoon", "letter", "check", "embroidery", "collar", "collar type", "lapel", "hooded", "neck", "high", "high neck", "shawl collar", "o", "o neck", "scoop", "scoop neck", "boat", "boat neck", "off", "shoulder", "off the shoulder", "v", "v neck", "button", "down", "button down", "square", "square neck", "pussy", "pussy bow", "shirt", "shirt collar", "polo", "peter", "pan", "peter pan", "notched", "fashion trend", "trend", "office", "street style", "street", "tropical", "retro vintage", "retro", "vintage", "basic", "preppy heritage", "preppy", "heritage", "party", "sexy", "bohemian", "minimalis", "korean", "clothing material", "clothing", "material", "fleece", "nylon", "velvet", "lace", "chiffon", "denim", "viscose", "polyester", "lycra", "linen", "silk", "poly cotton", "poly", "modal", "net", "wool", "satin", "rayon", "jersey", "cotton", "sleeves", "sleeveless", "sleeve 3 4", "short", "short sleeve", "long", "long sleeve"]


def get_continuous_chunks(text):
    chunked = chunker.parse(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
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
    try:
        qty = parser.parse(text)
    except KeyError:
        qty = []
    qty = ['/'.join([str(q.value), str(q.unit.name)]) for q in qty]
    return ' '.join(qty) if qty else np.nan
#end def


def get_stem(text):
    tokens = text.split(' ')
    stemmed = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed)
#end def


def get_keyword_count(text):
    d = Counter([word for word in text.split() if word in Keywords])
    return [d[k] for k in Keywords]
#end def


def read(df_path, key_word_count=False, lang=False, numbers=False, nouns=False, stems=False, quick=False):
    logger.info("Reading in data from {}....".format(df_path))

    df = pd.read_csv(df_path)
    if quick:
        df = df[:1024]

    if lang:
        # Lang Detection
        logger.info("Detecting language....")
        with mp.Pool(ncores) as pool:
            langs = pool.imap(get_lang, df['title'], chunksize=10)
            langs = [lang for lang in langs]
        #end with
        df['lang'] = pd.Series(langs)
        logger.info("Done detecting language....")

    if numbers:
        # Get number/unit
        logger.info("Extracting number/unit....")
        with mp.Pool(ncores) as pool:
            numbers = pool.imap(get_num, df['title'].str.cat(df[['translated', 'ocr']], sep='. ', na_rep=''), chunksize=10)
            numbers = [num for num in numbers]
        #end with
        df['numbers'] = pd.Series(numbers)
        #end try
        logger.info("Done extracting number/unit....")

    if nouns:
        # Get NP
        logger.info("Extracting noun phrases....")
        with mp.Pool(ncores) as pool:
            nouns = pool.imap(get_continuous_chunks, df['title'].str.cat(df[['translated', 'ocr']], sep='. ', na_rep=''), chunksize=10)
            nouns = [noun for noun in nouns]
        #end with
        df['nouns'] = pd.Series(nouns)
        logger.info("Done extracting noun phrases....")

    if stems:
        logger.info("Stemming....")
        with mp.Pool(ncores) as pool:
            stems = pool.imap(get_stem, df['title'], chunksize=10)
            stems = [stem for stem in stems]
        #end with
        df['stemmed_title'] = pd.Series(stems)
        logger.info("Done stemming....")

    if key_word_count:
        logger.info("Counting keywords....")
        with mp.Pool(ncores) as pool:
            X = pool.imap(get_keyword_count, df['title'].str.cat(df[['translated', 'ocr']], sep=' ', na_rep=''), chunksize=10)
            X = [x for x in X]
        #end with
        temp = pl.array(list(X))
        for i, t in enumerate(Keywords):
            df[t + '_count'] = temp[:, i]
        #end for
        logger.info('Done counting keywords....')

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
        df = read(
            f,
            key_word_count=True, lang=False,
            numbers=True, nouns=True,
            stems=False, quick=quick)
        df.to_csv(f.split('.csv')[0] + '_processed.csv', index=False)
#end def

if __name__ == '__main__': main()
