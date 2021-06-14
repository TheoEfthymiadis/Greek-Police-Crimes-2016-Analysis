import pandas as pd
import xlrd
import decimal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import scale
import random
from sklearn.metrics import silhouette_score
import seaborn as sns
import openpyxl
from scipy.spatial import distance

# Basic Operations of this Script
# Imports the dataset, executes some basic preprocessing and plots the distribution of Euclidean distances between the
# instances of the data set in the 9D space. Then, a similar approach with the 1D clustering is followed, the different
# parameter settings of different clustering algorithms are calculated and printed, along with a grid search for the
# silhouette coefficient for DBScan. Finally, a graph of the mean values of all features per cluster is printed along
# with the correlation matrix of the data set.


# Setting seeds to ensure reproducibility
random.seed(5)
np.random.seed(0)


def plot_silhouette(labels, clusters, algorithm):    # Function that plots the silhouette coefficient
    print(f'Average Silhouette Coefficient for {clusters}-clusters {algorithm}')
    print(silhouette_score(X, labels.values))
    print('------------------------------------------------------------------------------')


def drange(x, y, jump):   # Function that defines a range with decimals. Needed for loops
    while x < y:
        yield float(x)
        x = round(x + float(decimal.Decimal(jump)), 2)


# Importing data and doing basic preprocessing
police_df = pd.read_excel('2016_epikrateia.xls')   # sheet='ΕΠΙΚΡΑΤΕΙΑ'
police_df = police_df.fillna(value=0)   # Missing values are replaced with 0

# Rescale Numeric Features by subtracting the mean and dividing with the standard deviation
police_df[['τελ/να', 'απόπειρες', 'εξιχνιάσεις', 'ημεδαποί', 'αλλοδαποί']] = \
    scale(police_df[['τελ/να', 'απόπειρες', 'εξιχνιάσεις', 'ημεδαποί', 'αλλοδαποί']])

# One Hot encoding for categorical feature: Κατηγορία
police_df = pd.concat([police_df, pd.get_dummies(police_df['Κατηγορία'], prefix='Κατηγορία')], axis=1)
police_df = police_df.drop(labels='Κατηγορία', axis=1)

# Create the features list
police_df = police_df.set_index('Έγκλημα', drop=True)
features = police_df.columns.tolist()

# Calculate the distribution of euclidean distance between points in 9 dimensions
points = police_df.index.tolist()     # Create a list to iter over the different points
distances = pd.DataFrame(index=points, columns=points)    # Empty DataFrame to store the distances

# Distance calculation
for point1 in police_df.index:
    a = tuple(police_df.loc[point1, :].values)
    for point2 in police_df.index:
        b = tuple(police_df.loc[point2, :].values)
        distances.at[point1, point2] = dst = distance.euclidean(a, b)

# Only keep the upper diagonal part of the distance matrix to discard duplicates
diag = distances.to_numpy()[np.triu_indices(distances.to_numpy().shape[0], k=1)]
print('Average of euclidean 9D distance between points =  ' + str(np.mean(pd.Series(diag))))
print('Standard Deviation of euclidean 9D distance between points =  ' + str(np.std(pd.Series(diag))))

# Print a Histogram of the 9D distance distribution
plt.hist(diag, bins=[i for i in range(0, 11)])
plt.title('Distribution of Euclidean 9D Distances between points')
plt.show()

# Calculate the average distance of each point to all other points
average_distance = pd.DataFrame(index=police_df.index)   # Data frame to store the average distance results
for index, point in enumerate(police_df.index.tolist()):
    average_distance.at[point, 'Average Distance'] = np.mean(distances[point].to_numpy())

# Box plot of average distance
average_distance.plot.box()
plt.title('Average Distance of each crime type from all other crime types')
plt.show()

# Print the 3 identified potential outliers
outliers = average_distance.sort_values(by='Average Distance', ascending=False)[0:3]
print('------------------------------------------------------------------------------')
print('The 3 potential outlies of the data set are:')
print(outliers)

# Print the distances between the 3 potential outliers
print('------------------------------------------------------------------------------')
print('The Distances between the 3 potential outliers')
for outlier in outliers.index.tolist():
    print(distances.loc[outlier, outliers.index.tolist()])
print('------------------------------------------------------------------------------')

# Clustering with all features
X = police_df[features].to_numpy().reshape(-1, len(features))   # Feature array that will be used for 9D clustering

# K-means grid search
for cluster in range(2, 7):
    km = KMeans(n_clusters=cluster, random_state=0).fit(X)
    police_df['9D_Kmeans'] = km.fit_predict(X)
    plot_silhouette(police_df['9D_Kmeans'], cluster, '9D-Kmeans')

# Agglomerative grid search
for cluster in range(2, 7):
    ac = AgglomerativeClustering(n_clusters=cluster, affinity='euclidean', linkage='single')
    police_df['9D_Agglomerative'] = ac.fit_predict(X)
    plot_silhouette(police_df['9D_Agglomerative'], cluster, '9D_Agglomerative')

# DBSCAN clustering with all features
# Grid Search over parameters epsilon and minimum sample size
db_scan_epsilon = list(drange(1, 6, '0.5'))    # Grid search over epsilon
db_scan_min_sample = [i for i in range(3, 8)]   # Grid search over min_sample
db_scan_silhouette = np.zeros((len(db_scan_epsilon), len(db_scan_min_sample)))    # List to store the evaluation metrics
                                                                                  # for DB scan parameter search

epsilon_counter = 0
sample_counter = 0
for epsilon in db_scan_epsilon:
    sample_counter = 0
    for min_sample in db_scan_min_sample:
        db = DBSCAN(eps=epsilon, min_samples=min_sample, metric='euclidean')
        y_db = db.fit_predict(X)
        silhouette = silhouette_score(X, y_db)
        db_scan_silhouette[epsilon_counter][sample_counter] = silhouette
        sample_counter = sample_counter + 1
    epsilon_counter = epsilon_counter + 1

# Plot DBSCAN grid search results
ax = sns.heatmap(db_scan_silhouette, annot=True, linewidths=0.5,
                 xticklabels=db_scan_min_sample, yticklabels=db_scan_epsilon, cmap="rocket_r")
plt.xlabel('Minimum Number of Samples')
plt.ylabel('Epsilon Parameter')
plt.title('Sihlouette Coefficient Grid Search for 9D DBSCAN')
plt.show()

# Clustering with the optimal parameters

# This is an example of a cycle of computations. Different algorithms were used and the results were saved in an excel
# sheet. Manual observation of different excel sheets led to the intuitive description of the data set clusters
# DBSCAN with epsilon = 4 and minimum number of samples = 4
db = DBSCAN(eps=1, min_samples=4, metric='euclidean')
police_df['9D_DBSCAN'] = db.fit_predict(X)

# Agglomerative 2 clusters , 'complete linkage'
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
police_df['9D_Agglomerative'] = ac.fit_predict(X)

# K-means for 2 clusters
km = KMeans(n_clusters=4, random_state=0).fit(X)
police_df['9D_Kmeans'] = km.fit_predict(X)

police_df[['9D_Kmeans', '9D_Agglomerative', '9D_DBSCAN']].to_excel('9D_Results3.xlsx')

# Exploration of the clusters
class1_crimes = ['ΑΠΑΤΕΣ',	'Κλοπές - Διαρρήξεις καταστημάτων', 'ΕΠΑΙΤΕΙΑ',	'Κλοπές - Διαρρήξεις λοιπές',
                 'ΚΥΚΛΟΦΟΡΙΑ ΠΑΡΑΧΑΡΑΓΜΕΝΩΝ', 'Κλοπές Τροχοφόρων ΙΧΕ αυτ/των', 'Ν περί ΟΠΛΩΝ',
                 'Κλοπές Τροχοφόρων Μοτοσυκλετών', 'ΠΛΑΣΤΟΓΡΑΦΙΑ']

police_df['Class'] = 0
for crime in class1_crimes:
    police_df.at[crime, 'Class'] = 1

police_df.at['Κλοπές - Διαρρήξεις από ιχε αυτ/τα', 'Class'] = 2
police_df.at['Κλοπές - Διαρρήξεις οικιών', 'Class'] = 2

police_df.at['Ν περί ΝΑΡΚΩΤΙΚΩΝ', 'Class'] = 3

print('Class 0')
print(police_df[police_df['Class'] == 0]['Class'])
print('---------------------------------------------------')
print('Class 1')
print(police_df[police_df['Class'] == 1]['Class'])
print('---------------------------------------------------')
print('Class 2')
print(police_df[police_df['Class'] == 2]['Class'])
print('---------------------------------------------------')
print('Class 3')
print(police_df[police_df['Class'] == 3]['Class'])
print('---------------------------------------------------')

interesting_columns = ['τελ/να',  'απόπειρες',  'εξιχνιάσεις',  'ημεδαποί',  'αλλοδαποί',  'Κατηγορία_ΕΠΙΚΡΑΤΕΙΑ',
                       'Κατηγορία_ΚΛΟΠΕΣ - ΔΙΑΡΡΗΞΕΙΣ', 'Κατηγορία_ΚΛΟΠΕΣ ΤΡΟΧΟΦΟΡΩΝ',  'Κατηγορία_ΛΗΣΤΕΙΕΣ']

cluster_df = police_df[interesting_columns + ['Class']]

average_cluster = cluster_df.groupby(["Class"]).mean()
# std_cluster = cluster_df.groupby(["Class"]).std()
# median_cluster = cluster_df.groupby(["Class"]).median()

sns.heatmap(average_cluster, annot=True)
plt.title('Mean values for all attributes per crime type cluster')
plt.show()

# The features seem to be correlated. We plot the correlation matrix
sns.heatmap(police_df[interesting_columns].corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()
