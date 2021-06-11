"""
Script to check and visualice features and statistics of all the datasets in a folder
formatted with the our Spanish JSON structure.
"""

import os
import pandas as pd
import argparse
import json

# Función que lee y parsea los argumentos
def get_args():
    parser = argparse.ArgumentParser(description='Visualization of some usefull statistics about datasets.')
    parser.add_argument('-i', help='Path to the dataset folder.')
    args = parser.parse_args()

    return args

def count_captions(df):
    en_captions = 0
    es_captions = 0

    for image in df.loc['en_captions'].index:
        en_captions += len(df.loc['en_captions'][image])

    for image in df.loc['es_captions'].index:
        es_captions += len(df.loc['es_captions'][image])

    return en_captions, es_captions

def pandas_json(path_to_datasets):
    # Every file in the folder is a translated version of one dataset
    for dataset_file in os.listdir(path_to_datasets):
        dataset_file_path = os.path.join(path_to_datasets, dataset_file)

        # Creating a DataFrame with the JSON data
        with open(dataset_file_path, 'r') as f:
            dataset = json.load(f)
            print("Dataset name:", dataset['dataset']['name'])

            captions_df = pd.DataFrame(dataset['dataset']['data'])
            en_count, es_count = count_captions(captions_df)
            print("Total images in dataset: ", len(captions_df.columns))
            print("Total captions in English: ", en_count)
            print("Total captions in Spanish: ", es_count)

# main
def main():
    args = get_args()
    pandas_json(args.i)

if __name__ == "__main__":
    main()
