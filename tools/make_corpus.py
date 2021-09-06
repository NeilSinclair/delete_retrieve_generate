import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser(description="Combine corpus train_files for making a corpus")


parser.add_argument('--dataset',
                    default='yelp_15',
                    type=str,
                    help="String indicating which dataset to use; must be of the format dataset_percentNoise e.g. yelp_20")

parser.add_argument('--data_dir',
                    default='./data/',
                    type=str,
                    help="The directory where the data sits")

args = parser.parse_args()

# Concatenate the negative and positive training datasets and then write to file
d = args.dataset.split('_')

# This order matters because at the last step we concatenate the training data for making the vocab
datasets = ['test', 'train']

for set_ in datasets:
    # assume this run from the root directory
    x0 = pd.read_csv(f'{args.data_dir}{args.dataset}/{d[0]}_{set_}_{d[1]}.0')
    x1 = pd.read_csv(f'{args.data_dir}{args.dataset}/{d[0]}_{set_}_{d[1]}.1')
    for col in x0.columns:
        # Format the text
        x0[col] = x0[col].apply(lambda x: re.sub(r'\"', '', str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r'\.', ' .', str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r'!', ' !', str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r'\?', ' ?', str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r' {2,}', ' ', str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r',', ' ,', str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r"'s", " 's", str(x)))
        x0[col] = x0[col].apply(lambda x: re.sub(r"<mask> ", "", str(x)))

        x1[col] = x1[col].apply(lambda x: re.sub(r'\"', '', str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r'\.', ' .', str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r'!', ' !', str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r'\?', ' ?', str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r' {2,}', ' ', str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r',', ' ,', str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r"'s", " 's", str(x)))
        x1[col] = x1[col].apply(lambda x: re.sub(r"<mask> ", "", str(x)))
        
        # The yelp dataset is soo large, so if we're processing that, create a truncated version as this speeds
        # up the DRG evaluation a lot
        if d[0] == 'yelp' and set_=='test':
            x0.sample(frac=0.02).to_csv(f'{args.data_dir}{args.dataset}/{d[0]}_{set_}_short_{d[1]}.0', index=False)
            x1.sample(frac=0.02).to_csv(f'{args.data_dir}{args.dataset}/{d[0]}_{set_}_short_{d[1]}.1', index=False)

        x0.to_csv(f'{args.data_dir}{args.dataset}/{d[0]}_{set_}_{d[1]}.0', index=False)
        x1.to_csv(f'{args.data_dir}{args.dataset}/{d[0]}_{set_}_{d[1]}.1', index=False)

x0 = x0.Target
x1 = x1.Target

new_df = pd.concat([x0,x1])

new_df.to_csv(f'{args.data_dir}{args.dataset}/{d[0]}_train_{d[1]}.corpus', index=False)