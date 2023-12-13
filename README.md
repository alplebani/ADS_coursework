# COURSEWORK Alberto Plebani (ap2387)

README containing instructions on how to run the code for the coursework.

The repository can be cloned with 
```shell
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/m1_assessment/ap2387.git
```

The conda environment can be created using the [conda_env.yml](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/conda_env.yml?ref_type=heads), which contains all the packages needed to run the code
```shell
conda env create -n mphil --file conda_env.yml
```

# Report

The final report is presented in [ap2387.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/ap2387.pdf?ref_type=heads). The file is generated using LaTeX, but all LaTeX-related files are not being committed as per the instructions on the [.gitignore](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/.gitignore?ref_type=heads) file

# Code structure

The codes to run the exercises can be found in the ```src``` folder, whereas the file [Helpers/HelperFunctions.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/Helpers/HelperFunctions.py?ref_type=heads) contains the definition for the functions used to generate some plots. All plots are stored in the ```plots``` folder.

## SECTION A

# Part 1

The code for this exercise can be found in [src/solve_A_1.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/src/solve_A_1.py). The code can be run using ```parser``` options, which can be accessed with the following command
```shell
python src/solve_A_1.py -h
```
There are two possible flags that can be called: ```--features``` and ```--plots```. The former is used to generate the seaborn pairplot for the first 20 features. This flag was implemented because it takes a lot of time to generate the plot. The latter is a flag that is implemented also in all the other codes, and when set it shows the plots generated while running the code instead of simply saving them in the specific ```plots``` folder, which for this part is ```plots/Section_A_1/```

This code starts by generating density plots for the first 20 features. Then it applies PCA and it visualises the PCA, as well as the explained variance. Afterwards, the code will perform k-means on two subsets of the data, with the default number of clusters (8). After displaying the contingency matrix, the code tests values of k from 2 to 10, and for each it plots the silhouette and it displays the contingency matrix. Finally, the code will display the two clusters on the PCA, and then it will apply k-means after doing the PCA. These two plots can be found in ```plots/Section_A_1/kMeans_kMeans_before.pdf``` and in ```plots/Section_A_1/kMeans_PCA_before.pdf```, respectively. The code takes 19 seconds to run without the ```--features``` flag and 

# Part 2

The code for this exercise can be found in [src/solve_A_2.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/src/solve_A_2.py). The ```--plots``` is the only parser option available.

This code starts showing the labels in the dataset, displaying also the number of missing labels. 

After this, the code will look for duplicated rows, displaying which rows are equal, and in particular whether the duplicated rows have also the same label. If the labels are different, both rows are removed from the dataset, otherwise only the first one is kept.

Finally, the code will try to predict the classification labels of the missing labels, plotting the confusion matrix of the k-nearest neighbour classifier and displaying the true labels, the true labels plus the predicted missing labels and also the predicted labels for the whole dataset, to check how well the classifier performed. The code takes 0.8 seconds to run

# Part 3

The code for this exercise can be found in [src/solve_A_3.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/src/solve_A_3.py). The ```--heatmap``` is an additional flag to the ```--plots```, which plots the ```sns.heatmap``` plot for the missing features.

This code looks for missing features, displaying which samples and which features have missing, alongside plotting the missing features with ```sns.heatmap``` if the ```--feature``` flag is selected, with this plot being saved in ```plots/Section_A_3/missing_data.pdf```. Then the code will look for the outliers, first with the standardisation and then with the model-based GMM. With standardisation, the values for which $Z>3$ are printed out, whereas with the latter, the predicted outliers are removed from the dataframe, and then the datasets with and without the outliers are compared with the pairwise distance comparison, whose plot is saved in ```plots/Section_A_3/pairwise.pdf```. The code takes 19 seconds to run with the ```--heatmap``` flag and 3 seconds without.

# Part 4

The code for this exercise can be found in [src/solve_B_4.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/src/solve_B_4.py). The ```-n, --number``` option is an additional parser option to the ```--plots``` flag, and it determined the number of the most important features that can be included in the training in part e), with this value set to default at 12.

The code starts by pre-processing the dataset, displaying also the highest correlated features and the missing data. Then the random forest is applied, and an optimisation on the number of trees is performed. The accuracy vs number of trees is presented in a plot in ```plots/Section_B_4/forest_optimisation.pdf```. Then the code takes the ```N = args.number``` parameter and display the N most important features with their importance, and then re-trains the forest using only those N features. 

Afterwards, the code does the same thing with the Logistic regression, printing the 4 best features for each of the three classifiers. 

For both forest and regression, the accuracies for both the full training and the subset training are evaluated and printed out, as well as the classification report. 

The code takes 5 seconds to run

# Part 5

The code for this exercise can be found in [src/solve_B_5.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/m1_assessment/ap2387/-/blob/main/src/solve_B_5.py). The ```--plots``` is the only parser option available.

The code starts by doing the same pre-processing of Part 4. Afterwards, k-means is applied, and a logistic regression classifier is performed on the k-means clusters. Once again, the first 12 most important features are selected, and the classifier is retrained again. The same thing is done with the GMM instead of k-means, and the two models are then compared in the contingency matrix and also in a visualisation with the PCA (```plots/Section_B_5/PCA.pdf```). Furthermore, accuracy and classification report are printed for every training.

The code takes 4 seconds to run

