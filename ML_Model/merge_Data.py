import pandas as pd
import gzip
import os
import json
import argparse
import gc
import multiprocessing as mp


def appendCategory(category, appendCategory):
    # Appends appendcategory to category, then deletes appendCategory file
    print('Starting to Merge ' + appendCategory + ' into ' + category)
    df_file = pd.read_csv('./datasets/' + appendCategory + '.csv', dtype=str,
                          usecols=['review', 'category'])
    df_file['category'] = category
    df_file.to_csv('./datasets/' + category +
                   '.csv', header=False, index=False, mode='a')
    os.remove('./datasets/' + appendCategory + '.csv')
    print('Finished Merge of ' + appendCategory + ' into ' + category)


def renameCategory(inputCategory, outputCategory):
    # Renames category and deletes original category file
    print('Starting to Rename ' + inputCategory + ' into ' + outputCategory)
    df_file = pd.read_csv('./datasets/' + inputCategory + '.csv', dtype=str,
                          usecols=['review', 'category'])
    df_file['category'] = outputCategory
    df_file.to_csv('./datasets/' + outputCategory +
                   '.csv', index=False, mode='w')
    os.remove('./datasets/' + inputCategory + '.csv')
    print('Finished Rename of ' + inputCategory + ' into ' + outputCategory)


if __name__ == '__main__':
    mergeList = [
        ('All_Beauty', 'Luxury_Beauty'),
        ('Clothing_Shoes_and_Jewelry', 'AMAZON_FASHION'),
        ('CDs_and_Vinyl', 'Digital_Music'),
        ('Books', 'Kindle_Store'),
        ('Grocery_and_Gourmet_Food', 'Prime_Pantry'),
    ]
    processList = []
    for merge in mergeList:
        p = mp.Process(target=appendCategory, args=merge)
        processList.append(p)
        p.start()
    for p in processList:
        p.join()

    renameList = [
        ('All_Beauty', 'Beauty'),
        ('CDs_and_Vinyl', 'Music'),
        ('Grocery_and_Gourmet_Food', 'Food'),
    ]

    processList = []
    for rename in renameList:
        p = mp.Process(target=renameCategory, args=rename)
        processList.append(p)
        p.start()
    for p in processList:
        p.join()
