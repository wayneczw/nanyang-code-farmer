from argparse import ArgumentParser
import logging
import pandas as pd

logger = logging.getLogger(__name__)

beauty_targets = [
    'Benefits', 'Brand',
    'Colour_group', 'Product_texture',
    'Skin_type']

fashion_targets = [
    'Pattern', 'Collar Type',
    'Fashion Trend', 'Clothing Material',
    'Sleeves']

mobile_targets = [
    'Operating System', 'Features',
    'Network Connections', 'Memory RAM',
    'Brand', 'Warranty Period',
    'Storage Capacity', 'Color Family',
    'Phone Model', 'Camera', 'Phone Screen Size']


def main():
    parser = ArgumentParser(description='Combine predictions.')
    parser.add_argument('-f', '--files', type=str, nargs='+', metavar='<prediction_files>', required=True, help='List of prediction file pathsargparse.FileType('r'), nargs='+',.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    A = parser.parse_args()

    log_level = 'DEBUG'
    log_format = '%(asctime)-15s [%(name)s-%(process)d] %(levelname)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.getLevelName(log_level))

    out_df = pd.DataFrame(columns=['id', 'tagging'])
    for f in A.files:
        logger.info('Starting with {}....'.format(f))
        if 'beauty' in f:
            targets = beauty_targets
        elif 'fashion' in f:
            targets = fashion_targets
        elif 'mobile' in f:
            targets = mobile_targets
        #end if

        df = pd.read_csv(f)

        count = 0
        tmp_df = pd.DataFrame(columns=['id', 'tagging'])
        id_list = list()
        tagging_list = list()
        for i, row in df.iterrows():
            _id = str(row['itemid'])
            for target in targets:
                id_list.append(_id + '_' + target)
                tagging_list.append(row[target])
            #end for

            count += 1
            if count % 1000 == 0:
                logger.info('Hang on.... Has processed {} rows of data....'.format(count))
        #end for
        tmp_df['id'] = pd.Series(id_list)
        tmp_df['tagging'] = pd.Series(tagging_list)

        out_df = pd.concat([out_df, tmp_df])

        logger.info('Done with {}....'.format(f))
    #end for

    out_df.to_csv('./data/submission.csv', index=False)
#end def

if __name__ == '__main__': main()
