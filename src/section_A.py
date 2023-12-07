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
from sklearn.metrics import silhouette_score, silhouette_samples
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tabulate import tabulate

plt.style.use('mphil.mplstyle')
    
def features_plot(score, coeff, n1, n2, labels):
    plt.figure(figsize=(15,10))
    xs = score[:,int(n1)]
    ys = score[:,int(n2)]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,int(n1)], coeff[i,int(n2)],color = 'r',alpha = 0.5)
        plt.text(coeff[i,int(n1)]* 1.15, coeff[i,int(n2)] * 1.15, labels[i], color = 'r', ha = 'center', va = 'center')

    plt.xlabel("PC{}".format(n1+1))
    plt.ylabel("PC{}".format(n2+1))
    plt.title('PC' + str(n1+1) + ' vs PC' + str(n2+1) +', features direction')
    plt.grid()
    plt.savefig('plots/Section_A/features_direction.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A/features_direction.pdf')
    
def show_clusters_size(clusters):
    unique, counts = np.unique(clusters, return_counts=True)
    print(dict(zip(unique, counts)))
    return len(dict(zip(unique, counts)))
    
def show_pca(df_pca, clusters, name):
    
    plt.figure(figsize=(15,10))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], hue=clusters, palette='Set1')
    plt.title('kMeans: PCA for training set {}'.format(name))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('plots/Section_A/kMeans_{}.pdf'.format(name))
    print("=======================================")
    print('Saving plot at plots/Section_A/kMeans_{}.pdf'.format(name))


def show_single_silhouette(k_model, k, df_model, name):
    palette = sns.color_palette('Set1', k)
    silhouette_avg = silhouette_score( df_model, k_model)
    sing_silhouette_value = silhouette_samples(df_model, k_model)
    skplt.metrics.plot_silhouette(df_model, k_model)
    plt.xlim(np.min(sing_silhouette_value), np.max(sing_silhouette_value))
    plt.axvline(x=0, c='black')
    plt.savefig('plots/Section_A/silhouette_{0}_{1}.pdf'.format(k, name))


def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--features', help='Flag: if selected, will generate the seaborn pairplot for first 20 feature', required=False, action='store_true')
    parser.add_argument('--plots', help='Flag: if selected, will show the plots instead of only saving them', required=False, action='store_true')
    args = parser.parse_args()
    
    np.random.seed(4999)
    
    df = pd.read_csv("data/A_NoiseAdded.csv")
    
    df_20 = df.iloc[:, 1:21]

    if args.features:
        sns.pairplot(df_20)
        plt.savefig("plots/Section_A/features_pairplot.pdf")
        print("=======================================")
        print('Saving plot at plots/Section_A/features_pairplot.pdf')
        
    plt.figure(figsize=(25,15))
    for i in range(20):
        plt.subplot(4,5,i+1)
        plt.hist(df_20.iloc[:, i], bins=25, density=True)
        plt.xlabel('Feature {}'.format(i+1))
    plt.savefig('plots/Section_A/features_density.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A/features_density.pdf')
    
    df_vals = df.iloc[:, 1:-1]
    
    pca_2 = PCA(n_components=2)
    pca_fit_2 = pca_2.fit_transform(df_vals)
    print('Explained variance : {0}, {1}'.format(pca_2.explained_variance_ratio_[0], pca_2.explained_variance_ratio_[1]))
    print('Explained variance cumulative sum : {0}, {1}'.format(pca_2.explained_variance_ratio_.cumsum()[0], pca_2.explained_variance_ratio_.cumsum()[1]))
    
    df_pca_2 = pd.DataFrame(data=pca_fit_2, columns=['PC1', 'PC2'])
    plt.figure(figsize=(15,10))
    plt.title('PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.scatter(df_pca_2['PC1'], df_pca_2['PC2'])
    plt.savefig('plots/Section_A/PCA.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A/PCA.pdf')

    explained_variance = pca_2.explained_variance_ratio_.cumsum()
    
    plt.figure(figsize=(10,6))
    plt.bar(range(1,len(pca_2.explained_variance_ratio_ )+1),pca_2.explained_variance_ratio_ )
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(pca_2.explained_variance_ratio_ )+1),np.cumsum(pca_2.explained_variance_ratio_),c='red',label="Cumulative Explained Variance")
    plt.legend()
    plt.title('Cumulative explained variance : {}'.format(explained_variance[1]))
    plt.savefig('plots/Section_A/Explained_variance.pdf')
    print("=======================================")
    print('Saving plot at plots/Section_A/Explained_variance.pdf')
    
    features_plot(score=pca_fit_2[:, 0:16], coeff=np.transpose(pca_2.components_[0:16, :]),n1=0, n2=1, labels=list(df_vals.columns))
    
    test_1, test_2 = train_test_split(df_vals, test_size=0.5, random_state=4999)
    
    km_1 = KMeans(random_state=4999, n_init=10)
    km_2 = KMeans(random_state=4999, n_init=10)
    
    k_1 = km_1.fit(test_1)
    k_2 = km_2.fit(test_2)
    
    clusters_1 = k_1.predict(df_vals)
    clusters_2 = k_2.predict(df_vals)
    
    a = show_clusters_size(clusters_1)
    b = show_clusters_size(clusters_2)
    
    show_single_silhouette(k_model=clusters_1, k=a, df_model=df_vals, name='set_1')
    show_single_silhouette(k_model=clusters_2, k=b, df_model=df_vals, name='set_2')
    
    df_vals['Cluster_1'] = clusters_1
    df_vals['Cluster_2'] = clusters_2
    
    events_1 = np.bincount(df_vals['Cluster_1'])
    events_2 = np.bincount(df_vals['Cluster_2'])
    
    # for i in range
    
    # events = np.arange(start=0, stop=408, step=1, dtype=int)
    # ev_1 = df_vals.iloc[:, -2].to_numpy()
    # ev_2 = df_vals.iloc[:, -1].to_numpy()
    
    # table_data = list(zip(events, ev_1, ev_2))
    
    # print(tabulate(table_data, headers=['Events', 'kMeans1', 'kMeans2'], tablefmt='grid'))
        
    
    
    
    
    
    if args.plots:
        plt.show()
   
    
    
    
if __name__ == "__main__":
    print("=======================================")
    print("Initialising section A")
    print("=======================================")
    start_time = time.time()
    main()
    end_time = time.time()
    print("=======================================")
    print("Section A finished. Exiting!")
    print("Time it took to run the code : {} seconds". format(end_time - start_time))
    print("=======================================")