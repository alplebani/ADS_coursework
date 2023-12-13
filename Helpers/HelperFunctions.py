#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
import scikitplot as skplt

# ===============================================
# Helper functions 
# ===============================================

def features_plot(score, coeff, n1, n2, labels, folder):
    '''
    Function to generate the features plot in the PCA
    '''
    
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
    plt.savefig('plots/Section_{}/features_direction.pdf'.format(folder))
    print("=======================================")
    print('Saving plot at plots/Section_{}/features_direction.pdf'.format(folder))
    
def show_clusters_size(clusters):
    '''
    Function that prints out the number of events in each cluster, as well as returning the number of clusters
    '''
    unique, counts = np.unique(clusters, return_counts=True)
    print(dict(zip(unique, counts)))
    return len(dict(zip(unique, counts)))
    
def show_pca(df_pca, clusters, name, folder):
    '''
    Function that plots the events in each centroid with a PCA with 2 components
    '''
    
    plt.figure(figsize=(15,10))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=clusters, palette='Set1')
    plt.title('kMeans: PCA for training set {}'.format(name))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig('plots/Section_{0}/kMeans_{1}.pdf'.format(folder, name))
    print("=======================================")
    print('Saving plot at plots/Section_{0}/kMeans_{1}.pdf'.format(folder, name))


def show_single_silhouette(k_model, k, df_model, name, folder):
    '''
    Function that plots the single silhouette for a specific k-Means model
    '''
    
    palette = sns.color_palette('Set1', k)
    silhouette_avg = silhouette_score( df_model, k_model)
    sing_silhouette_value = silhouette_samples(df_model, k_model)
    skplt.metrics.plot_silhouette(df_model, k_model)
    plt.xlim(np.min(sing_silhouette_value), np.max(sing_silhouette_value))
    plt.axvline(x=0, c='black')
    plt.savefig('plots/Section_{0}/silhouette_{1}_{2}.pdf'.format(folder, k, name))
    print("=======================================")
    print('Saving plot at plots/Section_{0}/silhouette_{1}_{2}.pdf'.format(folder, k, name))