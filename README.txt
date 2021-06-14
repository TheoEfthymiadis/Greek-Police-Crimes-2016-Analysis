This is the list of the required Python libraries to run the scripts. The libraries were installed in a virtual environment using PyCharm and the scripts were ran inside the VE.

Package		Version		Latest Version
Pillow		8.1.0		8.1.0
cycler		0.10.0		0.10.0
et-xmlfile	1.0.1		-
jdcal		1.4.1		1.4.1
joblib		1.0.0		1.0.1
kiwisolver	1.3.1		1.3.1
matplotlib	3.3.4		3.3.4
numpy		1.19.5		1.20.1
openpyxl	3.0.6		3.0.6
pandas		1.1.5		1.2.2
pip		21.0		21.0.1
pyclustertend	1.4.9		1.4.9
pyparsing	2.4.7		2.4.7
python-dateutil	2.8.1		2.8.1
pytz		2020.5		2021.1
scikit-learn	0.24.1		0.24.1
scipy		1.6.0		1.6.0
seaborn		0.11.1		0.11.1
setuptools	40.8.0		53.0.0
six		1.15.0		1.15.0
sklearn		0.0		0.0
threadpoolctl	2.1.0		2.1.0
xlrd		2.0.1		2.0.1

All calculations were implemented using python scripts in the PyCharm environment. Specifically, the 3 different scripts that were created and their functionality will be briefly explained. It’s important to note that the original excel file was altered slightly manually. In order to reproduce the results, all python files should read the excel file that can be found in this repo.
•	1D_clustering.py: Imports the dataset, executes some basic preprocessing, introduces the ‘Police Success Rate’ column and plots the Hopkings Statistic graph. Moreover, it conducts 1D clustering with K-means and Agglomerative clustering and plots the results for different parameters. Finally, it executes a grid search for the parameters of DBscan and plots the clustering results of two different parameter setting for DBscan. 
•	9D_clustering.py: Imports the dataset, executes some basic preprocessing and plots the distribution of Euclidian distances between the instances of the data set in the 9D space. Then, a similar approach with the 1D clustering is followed, the different parameter settings of different clustering algorithms are calculated and printed, along with a grid search for the silhouette coefficient for DBScan. Finally, a graph of the mean values of all features per cluster is printed along with the correlation matrix of the data set.
•	Classification.py: The data set is imported once again. However, it is used to create new, uncorrelated features. After some preprocessing, the correlation matrix of the reformed data set is printed. Furthermore, an ExtraTrees classifier is trained and used to estimate the importance of the different features, which is illustrated using a bar chart. Finally, a grid search is conducted to identify the best performing decision tree classifier and its parameters are printed along with a graphical representation of the tree nodes.
For reproducibility of results, it’s recommended to create a virtual environment, install the necessary libraries and make sure that the excel file is present (as noted earlier, it’s slightly altered compared to the original data set) before running the scripts.
