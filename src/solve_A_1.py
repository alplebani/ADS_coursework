#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from Helpers.HelperFunctions import features_plot, show_clusters_size, show_pca, show_single_silhouette
from sklearn.preprocessing import StandardScaler

plt.style.use('mphil.mplstyle')

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--features', help='Flag: if selected, will generate the seaborn pairplot for first 20 feature', required=False, action='store_true')
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    args = parser.parse_args()
    
    my_seed = 4999
    
    # Reading dataset
    
    print('Reading dataset')
    print("=======================================")
    
    df = pd.read_csv("data/A_NoiseAdded.csv")
    
    df_20 = df.iloc[:, 1:21] # saving only the first 20 features
    
    if args.features: # generating pairplot for 20 features, only if the features flag is turned on
        sns.pairplot(df_20)
        plt.savefig("plots/Section_A_1/features_pairplot.pdf")
        print("=======================================")
        print('Saving plot at plots/Section_A_1/features_pairplot.pdf')
        
     # Generating features distribution
        
    plt.figure(figsize=(25,15))
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.hist(df_20.iloc[:, i], bins=25, density=True)
        plt.xlabel('Feature {}'.format(i+1))
    plt.savefig('plots/Section_A_1/features_density.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_1/features_density.pdf')
    
    # generating features density distributions 
    
    plt.figure(figsize=(15,10)) 
    plt.title('Density plot for first 20 features')
    i = 1
    for column in df_20.columns:
        plt.subplot(4,5,i)
        sns.kdeplot(df_20[column], label=column)
        i += 1
    plt.savefig('plots/Section_A_1/density.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_1/density.pdf')
    
    # generating features density distributions overlaid
    
    plt.figure(figsize=(18,11)) 
    plt.title('Density for first 20 features : overlaid')
    df_20.plot(kind = "density", figsize=(18,11))
    plt.savefig('plots/Section_A_1/density_overlaid.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_1/density_overlaid.pdf')
    
    df_vals = df.iloc[:, 1:-1] # removing first and last column, using only the features
    
    # PCA
    
    print("=======================================")
    print('Applying PCA')
    
    pca_2 = PCA(n_components=2)
    scaler = StandardScaler() # re-scaling before doing the PCA
    df_vals_scaled = scaler.fit_transform(df_vals)
    df_vals_scaled = pd.DataFrame(df_vals_scaled, columns=df_vals.columns)
    pca_fit_2 = pca_2.fit_transform(df_vals_scaled) # applying PCA
    
    # Evaluating explained variance
    
    print('Explained variance : {0}, {1}'.format(pca_2.explained_variance_ratio_[0], pca_2.explained_variance_ratio_[1]))
    print('Explained variance cumulative sum : {0}, {1}'.format(pca_2.explained_variance_ratio_.cumsum()[0], pca_2.explained_variance_ratio_.cumsum()[1]))
    
    # Visualizing PCA
    
    df_pca_2 = pd.DataFrame(data=pca_fit_2, columns=['PC1', 'PC2'])
    plt.figure(figsize=(15,10))
    plt.title('PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'])
    plt.savefig('plots/Section_A_1/PCA.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_1/PCA.pdf')
    
    # Visualizing explained variance

    explained_variance = pca_2.explained_variance_ratio_.cumsum()
    
    plt.figure(figsize=(10,6))
    plt.bar(range(1,len(pca_2.explained_variance_ratio_ )+1),pca_2.explained_variance_ratio_ )
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(pca_2.explained_variance_ratio_ )+1),np.cumsum(pca_2.explained_variance_ratio_),c='red',label="Cumulative Explained Variance")
    plt.legend()
    plt.title('Cumulative explained variance : {}'.format(explained_variance[1]))
    plt.savefig('plots/Section_A_1/Explained_variance.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A_1/Explained_variance.pdf')
    
    # generating features_plot
    
    features_plot(score=pca_fit_2[:, 0:16], coeff=np.transpose(pca_2.components_[0:16, :]),n1=0, n2=1, labels=list(df_vals.columns), folder='A_1')
    df_vals = scaler.fit_transform(df_vals) # re-scaling data
    test_1, test_2 = train_test_split(df_vals, test_size=0.5, random_state=my_seed) # splittig dataset in two of equal size
    
    # Performing k-Means
    
    print("=======================================")
    print('Starting with k-Means')
    
    km_1 = KMeans(random_state=4999, n_init=10) # n_clusters = 8 (default)
    km_2 = KMeans(random_state=4999, n_init=10)
    
    k_1 = km_1.fit(test_1)
    k_2 = km_2.fit(test_2)
    
    clusters_1 = k_1.predict(df_vals)
    clusters_2 = k_2.predict(df_vals)
    
    # Displaying events in each cluster
    
    a = show_clusters_size(clusters_1) 
    b = show_clusters_size(clusters_2)
    
    # Displaying single silhouette
    
    show_single_silhouette(k_model=clusters_1, k=a, df_model=df_vals, name='set_1', folder='A_1')
    show_single_silhouette(k_model=clusters_2, k=b, df_model=df_vals, name='set_2', folder='A_1')
    
    new_df_vals = df.iloc[:, 1:-1]
    
    # Adding cluster label to the df
    
    new_df_vals['Cluster_1'] = clusters_1
    new_df_vals['Cluster_2'] = clusters_2
    
    # Printing contingency matrix
    
    print('Contingency matrix. kMeans1 vs kMeans 2')
    print(contingency_matrix(new_df_vals['Cluster_1'], new_df_vals['Cluster_2']))
    
    # Finding optimal k
    
    print("=======================================")
    print('Testing different values of k')
    print("=======================================")
    
    for i in range(2,10):
        km_1 = KMeans(random_state=my_seed, n_clusters=i, n_init=10)
        km_2 = KMeans(random_state=my_seed, n_clusters=i, n_init=10)
        k1 = km_1.fit(test_1)
        k2 = km_2.fit(test_2)
        c1 = k1.predict(df_vals)
        c2 = k2.predict(df_vals)
        show_single_silhouette(k_model=c1, k=i, df_model=df_vals, name='set_1', folder='A_1')
        show_single_silhouette(k_model=c2, k=i, df_model=df_vals, name='set_2', folder='A_1')
        name = 'set_1_{}'.format(i)
        show_pca(df_pca_2, c1, name=name, folder='A_1')
        
        temp_df_vals = df.iloc[:, 1:-1]
        temp_df_vals['C1'] = c1
        temp_df_vals['C2'] = c2
        
        print('Contingency matrix for k = {}. kMeans1 vs kMeans 2'.format(i))
        print(contingency_matrix(temp_df_vals['C1'], temp_df_vals['C2']))
      
    # Best k = 2 => working with k=2 from now on        

    print("=======================================")
    print('Now working with k=2') 
    print("=======================================")
    
    kmeans_2 = KMeans(random_state=my_seed, n_clusters=2, n_init=10)
    km2 = kmeans_2.fit_predict(df_vals)
    show_single_silhouette(k_model=km2, k=2, df_model=df_vals, name='kMeans_before', folder='A_1')
    show_pca(df_pca=df_pca_2, clusters=km2, name='kMeans_before', folder='A_1')
    
    # Doing k-means on the PCA 
    
    print("=======================================")
    print('Now doing k-means on the PCA') 
    print("=======================================")
    
    kmeans_2 = KMeans(random_state=my_seed, n_clusters=2, n_init=10)
    km2 = kmeans_2.fit_predict(df_pca_2)
    show_single_silhouette(k_model=km2, k=2, df_model=df_pca_2, name='pca_before', folder='A_1')
    show_pca(df_pca=df_pca_2, clusters=km2, name='PCA_before', folder='A_1')
    
    if args.plots: # display the plots
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section A: exercise 1")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section A:1 finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")