#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    parser.add_argument('-n', '--number', help='Number of the most important features you want to include in the training', type=int, required=False, default=12)
    args = parser.parse_args()
    
    my_seed = 4999 # random seed to have reproducible code
    
    print('Reading dataset')
    
    df = pd.read_csv("data/ADS_baselineDataset.csv")
    
    print("=======================================")
    print('Beginning of pre-processing : looking for empty features') # removing columns with all zeros, because they carry no information
    
    df_vals = df.drop(df.columns[0], axis=1) # removing the sample name column
    df_vals_no_lab = df_vals.drop(df_vals.columns[-1], axis=1) # removing label
    df_vals = df_vals.loc[:, (df_vals != 0).any(axis=0)] # remove features with only zeros
    
    print("=======================================")
    print('Now looking at correlations')
    
    cor_matrix = df_vals_no_lab.corr().abs()
    cor_col = cor_matrix.unstack()
    print("The highest correlations are:")
    print(cor_col.sort_values(ascending=False)[960:980:2]) # printing out highest correlations
    
    print("---------------------------------------")
    print('Removing features with correlation greater than 90%: Fea345, Fea388 and Fea869')

    df_vals.drop(df_vals.columns[346], axis=1, inplace=True)
    df_vals.drop(df_vals.columns[389], axis=1, inplace=True)
    df_vals.drop(df_vals.columns[870], axis=1, inplace=True)

    print("=======================================")
    print('Now looking for missing data')
    
    missing_data = df_vals.isnull().sum() # looking for missing data

    print('The following features have missing data : ')
    print(missing_data[missing_data > 0])
    
    # Now doing random forest classifier
    
    print('=======================================')
    print('Now doing random forest')
    print('=======================================')
    
    df_vals = pd.get_dummies(df_vals) # one-hot encoding of categorical values. Probably useless but worth doing either way
    
    x = df_vals.drop('type', axis=1)
    y = df_vals['type']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=my_seed) # splitting dataset in training and testing
    
    # Scaler not applied because random forest doesn't need scaled data
    
    rf = RandomForestClassifier()
    
    rf.fit(x_train, y_train)
    
    y_pred = rf.predict(x_test)
    
    # Evaluating performance: accuracy and classification report
    
    accuracy = accuracy_score(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred)
    
    print("Accuracy : {}".format(accuracy))
    print('Test set classification error : {}'.format(float(1 - accuracy)))
    print('=======================================')
    print('Classification report:')
    print(class_report)
    
    # Optimising the forest: looking for different number of trees
    
    trees = np.linspace(1, 131, 14) 
       
    accuracies = []
    
    for tree in trees:
        rf_classifier = RandomForestClassifier(n_estimators=int(tree), oob_score=True, random_state=my_seed)
        rf_classifier.fit(x_train, y_train)        
        acc = rf_classifier.oob_score_
        accuracies.append(acc)
        
    # Plotting the results to select the optimal number of trees

    plt.figure(figsize=(15,10))
    plt.plot(trees, accuracies, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs number of trees')
    plt.savefig('plots/Section_B_4/forest_optimisation.pdf')
    print('=======================================')
    print('Saving plot at plots/Section_B_4/forest_optimisation.pdf')

    best_trees = trees[np.argmax(accuracies)]

    print("Optimal number of trees : {}".format(int(best_trees)))
    
    # Re-training with the optimal number of trees
    
    rf_classifier = RandomForestClassifier(n_estimators=int(best_trees), oob_score=True, random_state=my_seed)
    rf_classifier.fit(x_train, y_train)
    feature_importances = rf_classifier.feature_importances_ # extracting importances

    rank_feat = np.argsort(np.abs(feature_importances))[::-1] # sorting features by importance
    rank_import = feature_importances[rank_feat]
    
    number_of_features = args.number # Number of the most important features you want to include in the training
    
    first_features = rank_feat[:number_of_features]
    first_importances = rank_import[:number_of_features]
    
    # Plotting feature importance

    plt.figure(figsize=(15,10))
    plt.barh(range(len(first_features)), first_importances, align='center')
    plt.yticks(range(len(first_features)), first_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Input feature ranking')
    plt.savefig('plots/Section_B_4/forest_ranking_{}.pdf'.format(number_of_features))
    print('=======================================')
    print('Saving plot at plots/Section_B_4/forest_ranking_{}.pdf'.format(number_of_features))
    
    # Re-training with only best features
    
    print('=======================================')
    print('Now re-training the model using only first {} features'.format(number_of_features))
    
    sel_feats = x.columns[first_features]
    x_train_subset = x_train[sel_feats]
    x_test_subset = x_test[sel_feats]
    
    model_subset = RandomForestClassifier(n_estimators=int(best_trees), random_state=my_seed)
    
    model_subset.fit(x_train_subset, y_train)
    y_pred_subset = model_subset.predict(x_test_subset)
    print('Accuracy for first {0} features : {1}'.format(number_of_features, accuracy_score(y_test, y_pred_subset)))
    print('Error for first {0} features : {1}'.format(number_of_features, 1 - accuracy_score(y_test, y_pred_subset)))
    
    # Now doing logistic regression
    
    print('=======================================')
    print('Now doing different model : Logistic regression')
    print('=======================================')
    
    scaler = StandardScaler() # rescaling of data
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    lr = LR()
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    
    # Evaluating performance: accuracy and classification report
    
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_class_report = classification_report(y_test, y_pred_lr)
    print("Accuracy : {}".format(lr_accuracy))
    print('Test set classification error : {}'.format(float(1 - lr_accuracy)))
    print('=======================================')
    print('Classification report:')
    print(lr_class_report)
    print('=======================================')
    
    lr_features = np.argsort(lr.coef_)[:,:4].flatten() # extracting coefficients (importances)
    print('The 4 best features for each classifier:')
    feats = x.columns[lr_features].to_numpy()
    print(feats)
    
    print('---------------------------------------')
    print('For RandomForest the best 12 features were:')
    first_features = rank_feat[:12]
    print(x.columns[first_features].to_numpy())
    print('=======================================')
    print('Now re-training the model using only those first 12 features')

    # Re-training with only best features

    x_subset = x[x.columns[lr_features]]
    
    x_subset_train, x_subset_test, y_train, y_test = train_test_split(x_subset, y, test_size=0.2, random_state=my_seed) # splitting dataset in training and testing
    
    lr_feat = LR(max_iter=1000) # implemented because convergence wasn't reached with standard value of 100
    lr_feat.fit(x_subset_train, y_train)
    y_pred_lr_feat = lr_feat.predict(x_subset_test)
    
    # Evaluating performance: accuracy and classification report
    
    lr_accuracy_feat = accuracy_score(y_test, y_pred_lr_feat)
    lr_class_report_feat = classification_report(y_test, y_pred_lr_feat)
    print("Accuracy : {}".format(lr_accuracy_feat))
    print('Test set classification error : {}'.format(float(1 - lr_accuracy_feat)))
    print('=======================================')
    print('Classification report:')
    print(lr_class_report_feat)

    
    if args.plots: # showing the plots
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section B: exercise 4")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section B:4 finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")