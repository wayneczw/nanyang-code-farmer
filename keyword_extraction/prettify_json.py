from argparse import ArgumentParser
import json


def argParse():
    argparser = ArgumentParser(description='Select json file')
    argparser.add_argument('-f', '--file', type=str, required=True, help='Path to file.')
    A = argparser.parse_args()

    return A


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)

    return data


def write_json(file_path, data, indent=4):
    with open(file_path, 'w+') as f:
        json.dump(data, f, indent=indent)


def prettify_json():
    A = argParse()
    data = read_json(A.file)
    new_file = A.file.rsplit('.', 1)[0] + '_prettified.json'
    write_json(new_file, data)


if __name__ == '__main__':
    prettify_json()
