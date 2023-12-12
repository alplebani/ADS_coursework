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
from sklearn.mixture import GaussianMixture as GM
import re
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    args = parser.parse_args()
    
    np.random.seed(4999)
    
    print('Reading dataset')
    print("=======================================")
    
    df = pd.read_csv("data/C_MissingFeatures.csv")
    
    # print(df.describe())

    missing_data = df.isnull().sum()

    print('The following features have missing data : ')
    print(missing_data[missing_data > 0])
    
    miss_features = list(missing_data[missing_data > 0].index)
    
    print('Missing features : {}'.format(miss_features))
    
    rows_with_missing_data = df[df.isnull().any(axis=1)]


    print("The following observaions have missing data : ")
    print(rows_with_missing_data['Unnamed: 0'])
    
    rows_miss_data_names = rows_with_missing_data['Unnamed: 0'].to_numpy()
    
    row_numbers = [re.search(r'\d+(?=[^0-9]*$)', sample).group(0) for sample in rows_miss_data_names]
    
    missing_data_indicator = df.isnull()

    if args.plots: # because it takes time to generate this plot, so I plot it only with the flag option
        plt.figure(figsize=(20, 12))
        sns.heatmap(missing_data_indicator, cmap='viridis', cbar=False)
        plt.title('Visualisation of missing data')
        plt.savefig('plots/Section_A_3/missing_data.pdf')
        print("=======================================")
        print('Saving plot at plots/Section_A_3/missing_data.pdf')
    
    print("=======================================")
    print('Imputing missing data')

    
    df_miss = df[df[miss_features].isnull().any(axis=1)]
    df_not_miss = df.dropna(subset=miss_features)
    
    gmm = GM(n_components=1, random_state=4999)
    gmm.fit(df_not_miss[miss_features])
    
    added_vals = gmm.sample(len(df_miss))[0]
    
    for i,j in product(range(len(miss_features)), range(len(added_vals))):
        a_row = int(row_numbers[j]) - 1
        a_column = miss_features[i]
        df.at[a_row, a_column] = added_vals[j][i]
        
 
    print('=======================================')
    print('Now working on outliers: standardisation')
    
    # print(df.describe())
    
    scaler = StandardScaler()  
    columns_to_scale = df.columns[1:-1]
    data_scale = df[columns_to_scale]
    scaled_data = scaler.fit_transform(data_scale)
    df_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale)
    
    outlier_threshold = 3
    z_scores = pd.DataFrame((data_scale - data_scale.mean()) / data_scale.std())
    
    outliers = (z_scores > outlier_threshold) | (z_scores < -outlier_threshold)
    
    print(data_scale[outliers].stack().dropna())
    
    
    
    print('=======================================')
    print('Now working on outliers: model-based GMM')
    
    gmm = GM(n_components=2, random_state=42)  # 2 components: normal data (0) vs outlier data (1)
    gmm.fit(scaled_data)
    
    outliers = gmm.predict(scaled_data) == 1
    df_no_outliers = df[~outliers]

    print(df_no_outliers)
    
    print(df.describe())
    print(df_no_outliers.describe())
    
    original_distances = pdist(df.iloc[:, 1:-1].values)  
    no_outliers_distances = pdist(df_no_outliers.iloc[:, 1:-1].values)
    
    plt.figure(figsize=(15, 10))
    sns.histplot(original_distances, label='Original dataset', kde=True)
    sns.histplot(no_outliers_distances, label='Dataset without outliers', kde=True)
    plt.title('Pairwise distance comparison')
    plt.xlabel('Pairwise Distances')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/Section_A_3/pairwise.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_3/pairwise.pdf')
    
    if args.plots:
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section A: exercise 3")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section A:3 finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")