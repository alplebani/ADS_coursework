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
from sklearn.mixture import GaussianMixture as GM
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    parser.add_argument('-n', '--number', help='Number of features you want to display in the plot for the ranking', type=int, required=False, default=20)
    args = parser.parse_args()
    
    np.random.seed(4999)
    
    print('Reading dataset')
    print("=======================================")
    
    df = pd.read_csv("data/ADS_baselineDataset.csv")
    
    print(df)
    
    df_vals = df.drop(df.columns[0], axis=1) # removing the sample name column
    
    print("=======================================")
    print('Beginning of pre-processing : looking for missing data')
    
    missing_data = df.isnull().sum()

    print('The following features have missing data : ')
    print(missing_data[missing_data > 0])
    
    print('=======================================')
    print('Now doing random forest')
    print('=======================================')
    
    df_vals = pd.get_dummies(df_vals) # one-hot encoding of categorical values. Probably useless but worth doing either way
    
    x = df_vals.drop('type', axis=1)
    y = df_vals['type']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=4999)
    
    rf = RandomForestClassifier()
    
    rf.fit(x_train, y_train)
    
    y_pred = rf.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred)
    
    print("Accuracy : {}".format(accuracy))
    print('Test set classification error : {}'.format(float(1 - accuracy)))
    print('=======================================')
    print('Classification report:')
    print(class_report)
    
    
    
    trees = np.linspace(50, 300, 26)
    
    oob_errs = []
    
    for tree in trees:
        rf_classifier = RandomForestClassifier(n_estimators=int(tree), oob_score=True, random_state=42)
        
        rf_classifier.fit(x_train, y_train)        
        oob_error = 1 - rf_classifier.oob_score_
        oob_errs.append(oob_error)

    plt.figure(figsize=(15,10))
    plt.plot(trees, oob_errs, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Out-of-Bag Error')
    plt.title('Error vs number of trees')
    plt.savefig('plots/Section_B_4/forest_optimisation.pdf')
    print('=======================================')
    print('Saving plot at plots/Section_B_4/forest_optimisation.pdf')

    best_trees = trees[np.argmin(oob_errs)]

    print("Optimal number of trees : {}".format(int(best_trees)))
    
    
    feature_importances = rf_classifier.feature_importances_

    feature_importance_dict = dict(zip(x_train.columns, feature_importances))

    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)
    
    number_of_features = args.number
    
    first_importances = importances[:number_of_features]
    first_features = features[:number_of_features]

    plt.figure(figsize=(15,10))
    plt.barh(range(len(first_features)), first_importances, align='center')
    plt.yticks(range(len(first_features)), first_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Input feature ranking')
    plt.savefig('plots/Section_B_4/forest_ranking_{}.pdf'.format(number_of_features))
    print('=======================================')
    print('Saving plot at plots/Section_B_4/forest_ranking_{}.pdf'.format(number_of_features))
    
    print('=======================================')
    print('Now re-training the model using only first features')
    
    values_features = [10, 20, 30, 50, 70, 100]
    
    accuracies = []
    errors = []
    
    for val in values_features:
        n_feat = features[:val]
        print(n_feat)
        sel_feats = x.columns[n_feat]
        x_train_subset = x_train[sel_feats]
        x_test_subset = x_test[sel_feats]
        model_subset = RandomForestClassifier(n_estimators=best_trees, random_state=42)
        model_subset.fit(x_train_subset, y_train)
        y_pred_subset = model_subset.predict(x_test_subset)
        accuracies.append(accuracy_score(y_test, y_pred_subset))
        errors.append(1 -accuracy_score(y_test, y_pred_subset))
    
    plt.figure(figsize=(15,10))
    plt.plot(values_features, errors)
    plt.xlabel('Number of features used for training')
    plt.ylabel('Error')
    plt.title('Best number of features')
    plt.savefig('plots/Section_B_4/number_of_features.pdf')
    print('=======================================')
    print('Saving plot at plots/Section_B_4/number_of_features.pdf')
    
    if args.plots:
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section B: exercise 4")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section B:2 finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")