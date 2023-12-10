#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from itertools import product
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    args = parser.parse_args()
    
    np.random.seed(4999)
    
    print('Reading dataset')
    print("=======================================")
    
    df = pd.read_csv("data/B_Relabelled.csv")
    
    print("=======================================")
    
    
    print(df['classification'].value_counts())
    label_counter = df['classification'].value_counts().values
    
    print("=======================================")
    
    print(label_counter)
    
    tot_labels = 0
    
    n_rows, n_columns = df.shape
    
    for l in label_counter:
        tot_labels += l
    
    print('Total observations with labels : {}'.format(tot_labels))
    print('Total observations : {}'.format(n_rows))
    print('Missing labels : {}'.format(n_rows - tot_labels))

    
    df_values = df.iloc[:, 1:-1]
    
    df = df.iloc[:, 1:]
    
    duplicates = df_values.duplicated(keep=False)
    
    dup_data = df_values[duplicates]
    
    
    # print(dup_data)
    
    duplicates_grouped = dup_data.groupby(list(dup_data.columns)).apply(lambda x: x.index.tolist())
    
    wrong_labels = 0
    counter = 0

    # print(df[duplicates])
    
    print("=======================================")
    
    to_be_removed = []
    
    for group in duplicates_grouped:
        counter += 1
        print("Duplicated rows: {}".format(group))
        for i,j in product(group, group):
            if i <= j:
                continue
            # print('Row {0} and row {1}'.format(i, j))
            if df.iloc[i].equals(df.iloc[j]):
                print('The two observations are equal with the same label')
                print('Label {0} = {1}'.format(i, df.at[i, 'classification']))
                to_be_removed.append(j)
                print('Removing observation {}'.format(j))
                print("--------------------------------------")
            else:
                print('The two observations are equal BUT with different label')
                wrong_labels += 1
                print('Label {0} : {1}, Label {2} : {3}'.format(i, df.at[i,'classification'], j, df.at[j, 'classification']))
                to_be_removed.append(j)
                to_be_removed.append(i)
                print('Removing observations {0} and {1}'.format(i, j))
                print("--------------------------------------")
                
    
    print("=======================================")
    
    df.drop(to_be_removed, inplace=True)
    # print(df.shape)
    
    print('Number of duplicates : {}'.format(counter))
    print('Number of wrong labels in duplicates : {}'.format(wrong_labels))
    
    missing_values = df[df['classification'].isna()].index.tolist()
    print(missing_values)
    df_no_missing = df.drop(missing_values)
    
    print("=======================================")
    print("Starting KNN")
    print("=======================================")
    
    x = df_no_missing.drop('classification', axis=1) 
    target = df_no_missing['classification']
    
    x = pd.get_dummies(x)
    
    # print(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.33, random_state=4999)
    
    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    KNN = KNeighborsClassifier(n_neighbors=3)
    KNN.fit(x_train, y_train)
    
    y_pred = KNN.predict(x_train)
    
    conf_matrix = confusion_matrix(y_train, y_pred)
    
    plt.figure(figsize=(15,10))
    plt.title('Confusion matrix')
    sns.heatmap(conf_matrix, annot=True)
    plt.savefig('plots/Section_A_2/confusionMatrix.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_2/confusionMatrix.pdf')
    
    score = KNN.score(x_test, y_test)
    
    print('Score : {}'.format(score))
    
    print(conf_matrix)
    print(classification_report(y_train, y_pred))
    
    df_missing = df[df.index.isin(missing_values)]
    # print(df_missing)
    
    x_missing = df_missing.drop('classification', axis=1)
    scaler = StandardScaler()
    x_missing = scaler.fit_transform(x_missing)
    y_missing = KNN.predict(x_missing)
    
    df_out = df_missing.drop('classification', axis=1)
    
    # print(df_out)
    
    df_out['classification'] = y_missing
    
    final_x = df.drop('classification', axis=1)
    scaler = StandardScaler()
    final_x = scaler.fit_transform(final_x)
    final_y = KNN.predict(final_x)
    
    final_df = df.drop('classification', axis=1)
    
    final_df['classification'] = final_y
    
    final_class_counter = final_df['classification'].value_counts().values
    
    missing_counter = df_out['classification'].value_counts().values
    
    print('Prediction of missing labels : {}'.format(missing_counter))
    print('Known labels : {}'.format(label_counter))
    
    
    final_counter = label_counter + missing_counter
    
    print('Total labels : {}'.format(final_counter))
    print('Predicted labels: {}'.format(final_class_counter))
    
    
    
    if args.plots:
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section A: exercise 2")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section A:2 finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")