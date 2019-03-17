from googletrans import Translator
from tqdm import tqdm
from argparse import ArgumentParser
from prettify_json import read_json, write_json

# #mobile
# mobile_keywords = ["operating system", "operating", "system", "os", "symbian", "windows", "samsung", "blackberry", "nokia", "android", "ios", "features", "expandable memory", "touchscreen", "fingerprint sensor", "dustproof", "waterproof", "wifi", "gps", "network", "connections", "4g", "2g", "3g", "3.5g", "memory", "ram", "4gb", "2gb", "1.5gb", "16gb", "512mb", "8gb", "3gb", "10gb", "1gb", "6gb", "warranty", "period", "yr", "mth", "year", "month", "years", "months", "7 months", "4 months", "6 months", "3 months", "10 years", "2 month", "11 months", "10 months", "5 months", "3 years", "2 years", "1 month", "18 months", "1 year", "storage", "capacity", "gb", "mb", "256gb", "1.5gb", "128gb", "512mb", "64gb", "512gb", "8gb", "4mb", "6gb", "4gb", "2gb", "128mb", "32gb", "256mb", "10gb", "3gb", "1gb", "16gb", "color", "colour", "family", "blue", "gold", "brown", "navy blue", "yellow", "neutral", "rose gold", "light blue", "dark grey", "silver", "pink", "gray", "army", "green", "army green", "deep", "blue", "deep blue", "purple", "rose", "light", "grey", "light grey", "black", "deep black", "off", "white", "off white", "multicolor", "black", "apricot", "orange", "red", "camera", "mp", "single", "42mp", "dua slot", "5 mp", "3 mp", "1 mp", "8 mp", "single camera", "24 mp", "16mp", "13mp", "6 mp", "10mp", "2 mp", "20 mp", "4 mp", "phone", "screen", "size", "inch", "inches", "4.6 to 5 inches", "4.1 to 4.5 inches", "less than 3.5 inches", "3.6 to 4 inches", "more than 5.6 inches", "5.1 to 5.5 inches"]

# #beauty
# beauty_keywords = ["benefits", "high", "pigmentation", "natural", "light", "hydrating", "durable", "oil", "control", "spf", "colour", "group", "emas", "rose", "1 warna", "9 color", "emas", "hijau", "warna", "merah", "cabai", "warna merah cabai", "8 color", "5 color", "multiwarna", "perak", "krem", "peach", "coklat tua", "peanut", "biru", "ungu", "hitam", "abu", "6 color", "mawar", "4 color", "putih", "warna blush pink", "bening", "netral", "maroon", "kuning", "11 color", "merah", "warna koral", "3 color", "nude", "12 color", "ceri", "2 color", "warna fuchsia", "merah muda", "warna hotpink", "10 color", "merah semangka", "jeruk", "sawo matang", "7 color", "cokelat", "antique white", "product", "texture", "balm", "stick", "liquid", "crayon pensiln", "formula mousse", "cream", "solid", "powder", "solid powder", "cushion", "gel", "skin", "type", "dry", "sensitive", "fade", "combination", "normal", "aging", "age", "signs of aging", "acne", "greasy"]

# #fashion
# fashion_keywords = ["pattern", "paisley", "plaid", "threadwork", "patchwork", "plain", "graphic", "print", "gingham", "camouflage", "polka", "dot", "polka dot", "joint", "wave", "point", "wave point", "stripe", "knot", "floral", "brocade", "cartoon", "letter", "check", "embroidery", "collar", "collar type", "lapel", "hooded", "neck", "high", "high neck", "shawl collar", "o", "o neck", "scoop", "scoop neck", "boat", "boat neck", "off", "shoulder", "off the shoulder", "v", "v neck", "button", "down", "button down", "square", "square neck", "pussy", "pussy bow", "shirt", "shirt collar", "polo", "peter", "pan", "peter pan", "notched", "fashion trend", "trend", "office", "street style", "street", "tropical", "retro vintage", "retro", "vintage", "basic", "preppy heritage", "preppy", "heritage", "party", "sexy", "bohemian", "minimalis", "korean", "clothing material", "clothing", "material", "fleece", "nylon", "velvet", "lace", "chiffon", "denim", "viscose", "polyester", "lycra", "linen", "silk", "poly cotton", "poly", "modal", "net", "wool", "satin", "rayon", "jersey", "cotton", "sleeves", "sleeveless", "sleeve 3 4", "short", "short sleeve", "long", "long sleeve"]

# print(len(mobile_keywords))
# print(len(fashion_keywords))
# print(len(fashion_keywords))

# combine = [mobile_keywords, beauty_keywords, fashion_keywords]

# mobile_dict = dict()
# beauty_dict = dict()
# fashion_keywords = dict()


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-j', '--json', type=str, required=True, help='Path to json file.')
    argparser.add_argument('-o', '--output_filename', type=str, required=True, help='Name of output json file')
    argparser.add_argument('-k', '--keyname', type=str, nargs='+', required=True, help='Name of the column(s).')
    A = argparser.parse_args()
    return A


def main():
    A = argParse()
    mapping_dict = read_json(A.json)
    translator = Translator()

    res = dict()
    for key in A.keyname:
        to_be_translated = list(mapping_dict[key].keys())

        for word in to_be_translated:
            translated = translator.translate(word, src="en", dest='id')
            res[translated.text.strip().lower()] = word

    print(A.output_filename)
    write_json(A.output_filename, res)



if __name__ == '__main__':
    main()
