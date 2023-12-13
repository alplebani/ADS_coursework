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
    
    my_seed = 4999 # random seed
    
    # Reading dataset
    
    print('Reading dataset')
    print("=======================================")
    
    df = pd.read_csv("data/B_Relabelled.csv")
    
    print("=======================================")
    
    # Counting labels    
    
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

    
    df_values = df.iloc[:, 1:-1] # removing sample_name and labels
    
    df = df.iloc[:, 1:] # removing sample_name
    
    duplicates = df_values.duplicated(keep=False) # marks duplicates, with False option to mark all of them
    
    dup_data = df_values[duplicates] # df with only all duplicated data 
    
    duplicates_grouped = dup_data.groupby(list(dup_data.columns)).apply(lambda x: x.index.tolist())
    
    wrong_labels = 0
    counter = 0
    
    print("=======================================")
    
    # Working on removing duplicated data
    
    to_be_removed = []
    
    for group in duplicates_grouped:
        counter += 1
        print("Duplicated rows: {}".format(group))
        for i,j in product(group, group):
            if i <= j: # in order to avoid double counting
                continue
            if df.iloc[i].equals(df.iloc[j]): # True when also label is the same => remove only one of the two
                print('The two observations are equal with the same label')
                print('Label {0} = {1}'.format(i, df.at[i, 'classification']))
                to_be_removed.append(j)
                print('Removing observation {}'.format(j))
                print("--------------------------------------")
            else: # label is different => remove both
                print('The two observations are equal BUT with different label')
                wrong_labels += 1
                print('Label {0} : {1}, Label {2} : {3}'.format(i, df.at[i,'classification'], j, df.at[j, 'classification']))
                to_be_removed.append(j)
                to_be_removed.append(i)
                print('Removing observations {0} and {1}'.format(i, j))
                print("--------------------------------------")
                
    
    print("=======================================")
    
    df.drop(to_be_removed, inplace=True) # removes the duplicated data
    
    print('Number of duplicates : {}'.format(counter))
    print('Number of wrong labels in duplicates : {}'.format(wrong_labels))
    
    # Removing missing values
    
    missing_values = df[df['classification'].isna()].index.tolist()
    print(missing_values)
    df_no_missing = df.drop(missing_values) 
    
    # Now doing KNN
    
    print("=======================================")
    print("Starting KNN")
    print("=======================================")
    
    x = df_no_missing.drop('classification', axis=1) 
    target = df_no_missing['classification']
    
    x = pd.get_dummies(x) # one-hot encoding for categorical values
        
    x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.2, random_state=my_seed) # splitting df in train and test
    
    scaler = StandardScaler() # rescaling data
    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    KNN = KNeighborsClassifier(n_neighbors=3) # 3 neighbours because we have 3 true labels
    KNN.fit(x_train, y_train)
    
    y_pred = KNN.predict(x_train)
    
    conf_matrix = confusion_matrix(y_train, y_pred, normalize='all') # confusion matrix
    
    plt.figure(figsize=(15,10))
    plt.title('Confusion matrix')
    sns.heatmap(conf_matrix, annot=True)
    plt.savefig('plots/Section_A_2/confusionMatrix.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_2/confusionMatrix.pdf')
    
    # Evaluating performance of the KNN
    
    score = KNN.score(x_test, y_test)
    
    print('Score : {}'.format(score))
    
    print(conf_matrix)
    print(classification_report(y_train, y_pred))
    
    df_missing = df[df.index.isin(missing_values)] # selects only missing data
    
    # applying classifier to predict missing labels
    
    x_missing = df_missing.drop('classification', axis=1)
    scaler = StandardScaler() # re-scale the df
    x_missing = scaler.fit_transform(x_missing)
    y_missing = KNN.predict(x_missing)
    
    df_out = df_missing.drop('classification', axis=1)
    
    df_out['classification'] = y_missing # adding predicted values to the df
    
    # Now applying classifier to re-predict all labels, both the true ones and the missing ones
    
    final_x = df.drop('classification', axis=1)
    scaler = StandardScaler()
    final_x = scaler.fit_transform(final_x)
    final_y = KNN.predict(final_x)
    
    final_df = df.drop('classification', axis=1)
    
    final_df['classification'] = final_y
    
    # Now counting events in each class
    
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