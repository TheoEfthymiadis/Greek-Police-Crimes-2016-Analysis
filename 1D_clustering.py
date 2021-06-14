import pandas as pd
import xlrd
import decimal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import scale
from pyclustertend import hopkins
import random
from sklearn.metrics import silhouette_score
import seaborn as sns
import openpyxl

# Basic Operations of this Script
# Imports the dataset, executes some basic preprocessing, introduces the ‘Police Success Rate’ column and plots the
# Hopkings Statistic graph. Moreover, it conducts 1D clustering with K-means and Agglomerative clustering and plots the
# results for different parameters. Finally, it executes a grid search for the parameters of DBscan and plots the
# clustering results of two different parameter setting for DBscan

# Setting seeds to ensure reproducibility
random.seed(5)
np.random.seed(0)


def drange(x, y, jump):   # Function that defines a range with decimals. Needed for loops
    while x < y:
        yield float(x)
        x = round(x + float(decimal.Decimal(jump)), 2)


# Importing data and doing basic preprocessing
police_df = pd.read_excel('2016_epikrateia.xls')   # sheet='ΕΠΙΚΡΑΤΕΙΑ'
police_df = police_df.fillna(value=0)   # Missing values are replaced with 0

# Define the police success rate column
police_df['Επιτυχία'] = police_df['εξιχνιάσεις']/(police_df['τελ/να'] + police_df['απόπειρες'])
police_df.loc[police_df['Έγκλημα'] == 'ΣΕΞΟΥΑΛΙΚΗ ΕΚΜΕΤΑΛΛΕΥΣΗ', 'Επιτυχία'] = 1   # Set value to ignore older crimes

# Rescale Numeric Features by subtracting the mean and dividing with the standard deviation
police_df[['τελ/να', 'απόπειρες', 'εξιχνιάσεις', 'ημεδαποί', 'αλλοδαποί', 'Επιτυχία']] = \
    scale(police_df[['τελ/να', 'απόπειρες', 'εξιχνιάσεις', 'ημεδαποί', 'αλλοδαποί', 'Επιτυχία']])

# One Hot encoding for categorical feature: Κατηγορία
police_df = pd.concat([police_df, pd.get_dummies(police_df['Κατηγορία'], prefix='Κατηγορία')], axis=1)
police_df = police_df.drop(labels='Κατηγορία', axis=1)

# Dictionary to track id - Crime
crime_id = {}
for i in police_df.index:
    crime_id[i] = police_df.at[i, 'Έγκλημα']

#  Evaluating Clustering Tendencies with Hopkins Statistic
# Create features for Hopkins statistic
features = police_df.columns.tolist()
features.remove('Έγκλημα')

m_sampler = 15   # Maximum value of the sampling parameter for the Hopkins statistic
n_experiments = 10   # Number of sampling repetitions for each value  of m_sampler
hopkins_list = []  # List to store the results for different values of m_sampler
for i in range(1, m_sampler+1):
    hopkins_sum = 0
    for j in range(1, n_experiments+1):
        hopkins_index = hopkins(police_df[features].to_numpy().reshape(-1, len(features)), i)
        hopkins_sum = hopkins_sum + hopkins_index
    hopkins_list.append(hopkins_sum/n_experiments)

# Plot results of Hopkins Statistics
plt.plot([i for i in range(1, m_sampler+1)], hopkins_list)
plt.hlines(0.3, 1, m_sampler+1, colors='r')
plt.xlabel('Sampling Parameter M')
plt.ylabel('Hopkins Statistic')
plt.title('Average Value of Hopkins Statistic')
plt.show()

# 1D plot of the range of police success for specific crimes (normalized)
plt.figure()
a = police_df['Επιτυχία']
plt.hlines(1, min(police_df['Επιτυχία']), max(police_df['Επιτυχία']))  # Draw a horizontal line
plt.text(min(police_df['Επιτυχία'])-0.4, 1, str(round(min(police_df['Επιτυχία']), 2)), ha='left', va='center')
plt.text(max(police_df['Επιτυχία'])+0.4, 1, str(round(max(police_df['Επιτυχία']), 2)), ha='right', va='center')
plt.eventplot(a, orientation='horizontal', colors='b')
plt.axis('off')
plt.title('Different Crimes - Police Success in solving them (normalized)')
plt.show()

# Clustering
X = police_df['Επιτυχία'].to_numpy().reshape(-1, 1)   # Feature array that will be used for 1D clustering


def plot_1d_clusters(X, Y_2C, Y_3C, algorithm):   # Function that plots the 1D clustering results for 2 and 3 clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)          # X = feature, Y_2C = 2 clusters, Y_3C = 3 clusters,
    fig.suptitle('1D Clustering - ' + algorithm)  # algorithm = the clustering algorithm that was used (string)

    # Plot algorithm clustering with 2 clusters
    ax1.hlines(1, min(X), max(X))  # Draw a horizontal line
    ax1.eventplot(X[Y_2C == 0], orientation='horizontal', colors='b')
    ax1.eventplot(X[Y_2C == 1], orientation='horizontal', colors='g')
    ax1.set_title('2 Clusters')
    ax1.axis('off')

    # Plot algorithm clustering with 3 clusters
    ax2.hlines(1, min(X), max(X))  # Draw a horizontal line
    ax2.eventplot(X[Y_3C == 0], orientation='horizontal', colors='b')
    ax2.eventplot(X[Y_3C == 1], orientation='horizontal', colors='g')
    ax2.eventplot(X[Y_3C == 2], orientation='horizontal', colors='r')
    ax2.set_title('3 Clusters')
    ax2.axis('off')
    plt.show()


def plot_silhouette(labels, clusters, algorithm):    # Function that plots the silhouette coefficient
    print(f'Average Silhouette Coefficient for {clusters}-clusters {algorithm}')
    print(silhouette_score(X, labels.values))
    print('------------------------------------------------------------------------------')


# K-means clustering with 2 clusters
km = KMeans(n_clusters=2, random_state=0).fit(X)
police_df['1D_Kmeans_2C'] = km.fit_predict(X)

# K-means clustering with 3 clusters
km = KMeans(n_clusters=3, random_state=0).fit(X)
police_df['1D_Kmeans_3C'] = km.fit_predict(X)

# Plot K-means results
plot_1d_clusters(police_df['Επιτυχία'], police_df['1D_Kmeans_2C'], police_df['1D_Kmeans_3C'], 'K-means')

# Plot Silhouette Coefficient for K-means
plot_silhouette(police_df['1D_Kmeans_2C'], 2, 'K-means')
plot_silhouette(police_df['1D_Kmeans_3C'], 3, 'K-means')

# Agglomerative clustering with 2 clusters
ac2 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
police_df['1D_Agglomerative_2C'] = ac2.fit_predict(X)

# Agglomerative clustering with 3 clusters
ac3 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
police_df['1D_Agglomerative_3C'] = ac3.fit_predict(X)

# Plot Agglomerative results
plot_1d_clusters(police_df['Επιτυχία'], police_df['1D_Agglomerative_2C'], police_df['1D_Agglomerative_3C'],
                 'Agglomerative Clustering')

# Plot Silhouette Coefficient for Agglomerative
plot_silhouette(police_df['1D_Agglomerative_2C'], 2, 'Agglomerative')
plot_silhouette(police_df['1D_Agglomerative_3C'], 3, 'Agglomerative')

# DBSCAN clustering
# Grid Search over parameters epsilon and minimum sample size
db_scan_epsilon = list(drange(0.1, 0.43, '0.03'))    # Grid search over epsilon
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
plt.title('Sihlouette Coefficient Grid Search for 1D DBSCAN')
plt.show()

# DBSCAN with optimal parameters
db = DBSCAN(eps=0.19, min_samples=7, metric='euclidean')
police_df['1D_DBSCAN'] = db.fit_predict(X)

# Plot DBSCAN clusters
plt.hlines(1, min(police_df['Επιτυχία']), max(police_df['Επιτυχία']))  # Draw a horizontal line
plt.eventplot(police_df['Επιτυχία'][police_df['1D_DBSCAN'] == 0], orientation='horizontal', colors='b')
plt.eventplot(police_df['Επιτυχία'][police_df['1D_DBSCAN'] == -1], orientation='horizontal', colors='r')
plt.title('1D DBSCAN (eps = 0.19, min_sample=7')
plt.text(min(police_df['Επιτυχία'])-0.4, 1, str(round(min(police_df['Επιτυχία']), 2)), ha='left', va='center')
plt.text(max(police_df['Επιτυχία'])+0.4, 1, str(round(max(police_df['Επιτυχία']), 2)), ha='right', va='center')
plt.axis('off')
plt.show()

# DBSCAN with epsilon = 0.25  and min_sample = 5
db = DBSCAN(eps=0.25, min_samples=5, metric='euclidean')
police_df['1D_DBSCAN'] = db.fit_predict(X)

# Plot DBSCAN clusters
plt.hlines(1, min(police_df['Επιτυχία']), max(police_df['Επιτυχία']))  # Draw a horizontal line
plt.eventplot(police_df['Επιτυχία'][police_df['1D_DBSCAN'] == 0], orientation='horizontal', colors='b')
plt.eventplot(police_df['Επιτυχία'][police_df['1D_DBSCAN'] == -1], orientation='horizontal', colors='r')
plt.eventplot(police_df['Επιτυχία'][police_df['1D_DBSCAN'] == 1], orientation='horizontal', colors='g')
plt.title('1D DBSCAN (eps = 0.25, min_sample=5')
plt.text(min(police_df['Επιτυχία'])-0.4, 1, str(round(min(police_df['Επιτυχία']), 2)), ha='left', va='center')
plt.text(max(police_df['Επιτυχία'])+0.4, 1, str(round(max(police_df['Επιτυχία']), 2)), ha='right', va='center')
plt.axis('off')
plt.show()
