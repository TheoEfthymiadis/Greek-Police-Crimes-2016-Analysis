import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import random
import seaborn as sns
import openpyxl
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# Basic Operations of this Script
# Imports the data set. Creates new, uncorrelated features. After some preprocessing, the correlation matrix of the
# reformed data set is printed. Furthermore, an ExtraTrees classifier is trained and used to estimate the importance
# of the different features, which is illustrated using a bar chart. Finally a grid search is conducted to identify the
# best performing decision tree classifier and its parameters are printed along with a graphical representation of
# the tree nodes

# Setting seeds to ensure reproducibility
random.seed(5)
seed = np.random.seed = 7

# Importing data and doing basic preprocessing
police_df = pd.read_excel('2016_epikrateia.xls')   # sheet='ΕΠΙΚΡΑΤΕΙΑ'
police_df = police_df.fillna(value=0)   # Missing values are replaced with 0

# Define the new features to deal with correlations between the existing ones

# Police Success Rate
police_df['Επιτυχία'] = police_df['εξιχνιάσεις']/(police_df['τελ/να'] + police_df['απόπειρες'])
police_df.loc[police_df['Έγκλημα'] == 'ΣΕΞΟΥΑΛΙΚΗ ΕΚΜΕΤΑΛΛΕΥΣΗ', 'Επιτυχία'] = 1   # Set value to ignore older crimes

# Criminals per solved case
police_df['δράστες/έγκλημα'] = (police_df['ημεδαποί'] + police_df['αλλοδαποί'])/police_df['εξιχνιάσεις']

# Proportion of local to foreign criminals
police_df['ημεδαποί'] = police_df['ημεδαποί'].apply(lambda x: x+1)   # Some values are 0 and result to inf values
police_df['αλλοδαποί'] = police_df['αλλοδαποί'].apply(lambda x: x+1)   # We apply the Laplacian correction
police_df['ημεδαποί/αλλοδαποί'] = police_df['ημεδαποί']/police_df['αλλοδαποί']

# Proportion of attempts to committed crimes
police_df['απόπειρες/τελ/να'] = police_df['απόπειρες']/police_df['τελ/να']
new_features = ['τελ/να', 'απόπειρες/τελ/να', 'Επιτυχία', 'δράστες/έγκλημα', 'ημεδαποί/αλλοδαποί', 'Κατηγορία']
new_police_df = police_df[new_features]

# Print Summary Statistics of each column. Will be needed to interpret the Predictions Later on:
print(new_police_df.describe().loc[['mean', 'std'], :])

# Rescale Numeric Features by subtracting the mean and dividing with the standard deviation
new_police_df[['τελ/να', 'απόπειρες/τελ/να', 'Επιτυχία', 'δράστες/έγκλημα', 'ημεδαποί/αλλοδαποί']] = \
    scale(police_df[['τελ/να', 'απόπειρες/τελ/να', 'Επιτυχία', 'δράστες/έγκλημα', 'ημεδαποί/αλλοδαποί']])

print('Rescaled Data Set with New Features')
print(new_police_df.head())

# Plot the correlation matrix of the reformed data set
sns.heatmap(new_police_df.corr(), annot=True)
plt.title('Correlation Matrix of reformed data set')
plt.show()

# Classification
# Feature Vector
X = new_police_df[['τελ/να', 'απόπειρες/τελ/να', 'Επιτυχία', 'δράστες/έγκλημα', 'ημεδαποί/αλλοδαποί']]

# Label Vector
y = new_police_df['Κατηγορία']

# Stratified 7-fold split object that will be used. 42 = 7*6. Each fold will train on 36 instances and validate on 6
cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)

# Feature Importance with Extra Trees Classifier
# feature importance extraction
model = ExtraTreesClassifier(n_estimators=100, random_state=seed)
model.fit(X, y)

df_feature_importance = pd.DataFrame(model.feature_importances_,
                                     index=list(X.columns),
                                     columns=['feature importance']).sort_values('feature importance', ascending=False)
df_feature_importance = df_feature_importance.reset_index()
df_feature_importance = df_feature_importance.rename(columns={"index": "feature"})

# Accuracy of cross-validation with Extra Tree Classifier
y_pred = cross_val_predict(model, X, y, cv=cv)
print('Accuracy of the 7-fold Cross Validation with ExtraTree')
print(accuracy_score(y, y_pred))

# Plotting the Features sorted by their importance
print('Feature importance according to the ExtraTreesClassifier')
print(df_feature_importance)
print('-------------------------------------------------------')
df_feature_importance.plot.bar(x='feature', y='feature importance')
plt.xticks(rotation='horizontal')
ax1 = plt.axes()
ax1.xaxis.set_label_text('foo')
ax1.xaxis.label.set_visible(False)
plt.ylabel('Feature Importance')
plt.title(f'Feature importance according to the ExtraTreesClassifier with 100 estimators'
          f' \n 7-fold CV Accuracy: {round(accuracy_score(y, y_pred), 4)}')
plt.show()

# Decision Tree
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(2, 15)}
# decision tree model
dtree_model = DecisionTreeClassifier(random_state=seed)
# use gridsearch to test all values
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=cv)
# fit model to data
dtree_gscv.fit(X, y)
y_pred = dtree_gscv.predict(X)
print('Best Tree')
print(dtree_gscv.best_params_)
print('Accuracy Score')
print(accuracy_score(y, y_pred))

# Plot the Decision Tree figure
plt.figure(figsize=(15, 10))
tree.plot_tree(dtree_gscv.best_estimator_.fit(X, y), feature_names=X.columns)
plt.title(f'Best Decision Tree Classifier: Accuracy = 88.1%'
          f'\n Parameters: {dtree_gscv.best_params_}')
plt.show()
