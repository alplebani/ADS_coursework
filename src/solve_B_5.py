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
from sklearn.mixture import GaussianMixture as GM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from Helpers.HelperFunctions import show_clusters_size
from sklearn.linear_model import LogisticRegression as LR
from sklearn.decomposition import PCA

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    args = parser.parse_args()
    
    my_seed = 4999 # random seed to have reproducible code
    
    print('Reading dataset')
    print("=======================================")
    
    df = pd.read_csv("data/ADS_baselineDataset.csv")
    
    # Beginning of pre-processing : looking for empty features
    
    print("=======================================")
    print('Beginning of pre-processing : looking for empty features') # removing columns with all zeros, because they carry no information
    
    df_vals = df.drop([df.columns[0], df.columns[-1]], axis=1) # removing the sample name column and label (because it's unsupervised)
    df_vals = df_vals.loc[:, (df_vals != 0).any(axis=0)] # remove features with only zeros

    # Correlation between features: removing too-highly correlated features 
    
    print("=======================================")
    print('Now looking at correlations')
    
    cor_matrix = df_vals.corr().abs()
    cor_col = cor_matrix.unstack()
    print("The highest correlations are:")
    print(cor_col.sort_values(ascending=False)[960:980:2])# printing out highest correlations
    
    print("---------------------------------------")
    print('Removing features with correlation greater than 90%: Fea345, Fea388 and Fea869')

    df_vals.drop(df_vals.columns[346], axis=1, inplace=True)
    df_vals.drop(df_vals.columns[389], axis=1, inplace=True)
    df_vals.drop(df_vals.columns[870], axis=1, inplace=True)

    # Looking for missing data

    print("=======================================")
    print('Now looking for missing data')
    
    missing_data = df_vals.isnull().sum()

    print('The following features have missing data : ')
    print(missing_data[missing_data > 0])
    
    df_vals = pd.get_dummies(df_vals) # one-hot encoding of categorical values. Probably useless but worth doing either way
    
    temp_df = df_vals.copy()
    
    # Now performing k-means
    
    print('=======================================')
    print('Clustering : k-Means')
    print('=======================================')
    
    scaler = StandardScaler() # rescaling the data
    x = scaler.fit_transform(df_vals)
    
    km = KMeans(random_state=my_seed, n_init=10, n_clusters=3, max_iter=1000000) # 3 clusters chosen because we have three targets
    k = km.fit(x)
    clusters = k.predict(x)
    n_clusters = show_clusters_size(clusters) # printing number of events in each cluster
    
    df_vals['k-means'] = clusters # adding the labels obtained in the df
    
    # Now training logistic regression on k-Means
    
    print('=======================================')
    print('Now training classifier : logistic regression') 
    
    x = df_vals.drop('k-means', axis=1)
    y = df_vals['k-means']
    x_vals = scaler.fit_transform(x) # rescaling data
    
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y, test_size=0.33, random_state=my_seed) # splitting between train and test
    
    lr = LR(random_state=my_seed, multi_class='multinomial') # doing linear regression
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    full_y = lr.predict(x_vals)
    
    df_vals['LR_kMeans'] = full_y # adding labels to the df
    
    print('=======================================')
    print('Predicted outputs with linear regression :')
    print(df_vals['LR_kMeans'].value_counts())
    
    # Evaluating performance: accuracy and classification report

    lr_accuracy = accuracy_score(y_test, y_pred)
    lr_class_report = classification_report(y_test, y_pred)
    
    print('=======================================')
    print("Accuracy : {}".format(lr_accuracy))
    print('Test set classification error : {}'.format(float(1 - lr_accuracy)))
    print('=======================================')
    print('Classification report:')
    print(lr_class_report)
    print('=======================================')
    
    lr_features = np.argsort(lr.coef_)[:,:4].flatten() # extracting best features
    print('The 4 best features for each classifier:')
    feats = x.columns[lr_features].to_numpy()
    print(feats)
    
    # Now re-training only using best 12 features (4 for each classifiers)
    
    x_subset = x[x.columns[lr_features]]
    x_subset_vals = scaler.fit_transform(x_subset)
    
    x_subset_train, x_subset_test, y_train, y_test = train_test_split(x_subset_vals, y, test_size=0.2, random_state=my_seed) # splitting between train and test
    
    lr_feat = LR(max_iter=1000) # implemented because convergence wasn't reached with standard value of 100
    lr_feat.fit(x_subset_train, y_train)
    y_pred_lr_feat = lr_feat.predict(x_subset_test)
    full_y_feat = lr_feat.predict(x_subset_vals)
    
    df_vals['LR_KM_12feat'] = full_y_feat # adding labels to the df
    
    print('=======================================')
    print('Predicted outputs with linear regression for first 12 features:')
    print(df_vals['LR_KM_12feat'].value_counts())
    print('=======================================')
    
    # Evaluating performance: accuracy and classification report
    
    lr_accuracy_feat = accuracy_score(y_test, y_pred_lr_feat)
    lr_class_report_feat = classification_report(y_test, y_pred_lr_feat)
    print("Accuracy : {}".format(lr_accuracy_feat))
    print('Test set classification error : {}'.format(float(1 - lr_accuracy_feat)))
    print('=======================================')
    print('Classification report:')
    print(lr_class_report_feat)
    
    # Now doing Gaussian Mixture
    
    print('=======================================')
    print('Clustering : Gaussian Mixture') 
    print('=======================================')
    
    gm = GM(n_components=3, random_state=my_seed, max_iter=1000000) # gaussian mizture with 3 components
    g = gm.fit(x)
    clusters_gm = g.predict(x)
    n_clusters_gm = show_clusters_size(clusters_gm) # printing number of events in each cluster
    
    df_vals['gm'] = clusters_gm
    temp_df['gm'] = clusters_gm
    
    # Printing contingency matrix

    print('Contingency matrix. kMeans vs GM')
    print(contingency_matrix(df_vals['k-means'], df_vals['gm']))
    
    # Now training logistic regression on Gaussian Mixture output
    
    print('=======================================')
    print('Now training classifier : logistic regression') 
    
    x_gm = temp_df.drop('gm', axis=1)
    y_gm = temp_df['gm']
    x_vals_gm = scaler.fit_transform(x_gm) # rescaling data
    
    x_train_gm, x_test_gm, y_train_gm, y_test_gm = train_test_split(x_vals_gm, y_gm, test_size=0.33, random_state=my_seed) # splitting dataset in test and training
    
    lr_gm = LR(random_state=my_seed, multi_class='multinomial') # doing logistic regression
    lr_gm.fit(x_train_gm, y_train_gm)
    y_pred_gm = lr_gm.predict(x_test_gm)
    full_y_gm = lr_gm.predict(x_vals_gm)
    
    df_vals['LR_gm'] = full_y_gm # adding labels to df
    
    print('=======================================')
    print('Predicted outputs with linear regression for first 12 features:')
    print(df_vals['LR_gm'].value_counts())
    print('=======================================')
    
    # Evaluating performance: accuracy and classification report
    
    lr_accuracy_gm = accuracy_score(y_test_gm, y_pred_gm)
    lr_class_report_gm = classification_report(y_test_gm, y_pred_gm)

    print("Accuracy : {}".format(lr_accuracy_gm))
    print('Test set classification error : {}'.format(float(1 - lr_accuracy_gm)))
    print('=======================================')
    print('Classification report:')
    print(lr_class_report_gm)
    print('=======================================')
    
    # Now retraining using only 12 best features
    
    lr_features_gm = np.argsort(lr_gm.coef_)[:,:4].flatten() # extracting best features
    print('The 4 best features for each classifier:')
    feats_gm = x_gm.columns[lr_features_gm].to_numpy()
    print(feats_gm)
    
    x_subset_gm = x_gm[x_gm.columns[lr_features_gm]]
    x_subset_vals_gm = scaler.fit_transform(x_subset_gm) # rescaling data
    
    x_subset_train_gm, x_subset_test_gm, y_train_gm, y_test_gm = train_test_split(x_subset_vals_gm, y_gm, test_size=0.2, random_state=my_seed) # splitting dataset in training and testing
    
    lr_feat_gm = LR(max_iter=1000) # implemented because convergence wasn't reached with standard value of 100
    lr_feat_gm.fit(x_subset_train_gm, y_train_gm)
    y_pred_lr_feat_gm = lr_feat_gm.predict(x_subset_test_gm)
    full_y_feat_gm = lr_feat_gm.predict(x_subset_vals_gm)
    
    df_vals['LR_gm_12feat'] = full_y_feat_gm # adding labels to the df
    
    print('=======================================')
    print('Predicted outputs with linear regression for first 12 features:')
    print(df_vals['LR_gm_12feat'].value_counts())
    print('=======================================')
    
    # Evaluating performance: accuracy and classification report
    
    lr_accuracy_feat_gm = accuracy_score(y_test_gm, y_pred_lr_feat_gm)
    lr_class_report_feat_gm = classification_report(y_test_gm, y_pred_lr_feat_gm)
    print("Accuracy : {}".format(lr_accuracy_feat_gm))
    print('Test set classification error : {}'.format(float(1 - lr_accuracy_feat_gm)))
    print('=======================================')
    print('Classification report:')
    print(lr_class_report_feat_gm)
    
    # Visualizing data
    
    print('=======================================')
    print('Now visualizing data : PCA')
    print('=======================================')
    
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(x)
    
    print('Explained variance : {0}, {1}'.format(pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]))
    print('Explained variance cumulative sum : {0}, {1}'.format(pca.explained_variance_ratio_.cumsum()[0], pca.explained_variance_ratio_.cumsum()[1]))
    
    df_pca_2 = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    
    plt.figure(figsize=(20,12))
    plt.title('PCA')
    plt.subplot(3,2,1)
    plt.title('k-Means')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], c=df_vals['k-means'])
    plt.subplot(3,2,2)
    plt.title('GM')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], c=df_vals['gm'])
    plt.subplot(3,2,3)
    plt.title('Feature {}'.format(feats[0]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], c=df_vals[feats[0]])
    plt.subplot(3,2,4)
    plt.title('Feature {}'.format(feats_gm[0]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], c=df_vals[feats_gm[0]])
    plt.subplot(3,2,5)
    plt.title('Feature {}'.format(feats[1]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], c=df_vals[feats[1]])
    plt.subplot(3,2,6)
    plt.title('Feature {}'.format(feats_gm[1]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'], c=df_vals[feats_gm[1]])
    plt.savefig('plots/Section_B_5/PCA.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_B_5/PCA.pdf')
    
    if args.plots: # showing the plots
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section B: exercise 5")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section B:5 finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")