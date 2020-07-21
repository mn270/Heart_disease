import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # for data splitting
from eli5.sklearn import PermutationImportance
import eli5
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

pd.options.mode.chained_assignment = None  # hide any pandas warnings
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz  # plot tree
from sklearn.model_selection import cross_val_score
# import warnings filter
from warnings import simplefilter
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Path of the file to read
heart_filepath = "/home/marcin/Pobrane/heart-disease-dataset(1)/heart.csv"

# Read the file into a variable heart_data
heart_data = pd.read_csv(heart_filepath)

heart_data.head()
print(heart_data.columns)

heart_data.columns = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG',
                      'Max_Heart_Rate',
                      'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia', 'Patient']
heart_data.head()

bg_color = (0.25, 0.25, 0.25)
sns.set(rc={"font.style": "normal",
            "axes.facecolor": bg_color,
            "figure.facecolor": bg_color,
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.labelcolor": "white",
            "axes.grid": False,
            'axes.labelsize': 25,
            'figure.figsize': (10.0, 5.0),
            'xtick.labelsize': 15,
            'ytick.labelsize': 15})

# data show
fig = plt.figure(figsize=(15, 20))
# create a heatmap
da = heart_data.corr()
sns.heatmap(data=da, annot=True, cmap='Reds')
# create a boxplots for multivalue varriable
f, axes = plt.subplots(1, 5, figsize=(20, 10))
sns.set(font_scale=1)
sns.boxplot(x=heart_data.Patient, y=heart_data.Max_Heart_Rate, ax=axes[0], fliersize=4);
sns.boxplot(x=heart_data.Patient, y=heart_data.Cholesterol, ax=axes[1]);
sns.boxplot(x=heart_data.Patient, y=heart_data.Age, ax=axes[2]);
sns.boxplot(x=heart_data.Patient, y=heart_data.Resting_BP, ax=axes[3]);
sns.boxplot(x=heart_data.Patient, y=heart_data.ST_Depression, ax=axes[4]);
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()

# RECG
grap = pd.crosstab(heart_data.RECG, heart_data.Patient).plot(kind="bar", figsize=(20, 10),
                                                             color=['darkolivegreen', 'darkred', ])
ax = plt.gca()
ax.set_facecolor((0.25, 0.25, 0.25))
plt.legend(['No Heart Disease', 'Has Heart Disease'], prop={'size': 30})
plt.xlabel('Resting ECG Type', fontsize=30)
plt.xticks(rotation=0)
plt.ylabel('Frequency of Heart Disease', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
locs, labels = plt.xticks()
new_xticks = ['Normal Waves', 'Abnormal Waves', 'Showing Left Ventricular Hypertrophy']
plt.xticks(locs, new_xticks)
plt.title("Correlation between RECG and Heart Disease", fontsize=40)

for p in grap.patches:
    grap.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Major_Vessels
grap = pd.crosstab(heart_data.Major_Vessels, heart_data.Patient).plot(kind="bar", figsize=(20, 10),
                                                                      color=['darkolivegreen', 'darkred', ])
ax = plt.gca()
ax.set_facecolor((0.25, 0.25, 0.25))
plt.legend(['No Heart Disease', 'Has Heart Disease'], prop={'size': 30})
plt.xlabel('Number of major vessels', fontsize=30)
plt.xticks(rotation=0)
plt.ylabel('Frequency of Heart Disease', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Correlation between Major Vessels and Heart Disease", fontsize=40)

for p in grap.patches:
    grap.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Thallasemia
grap = pd.crosstab(heart_data.Thalassemia, heart_data.Patient).plot(kind="bar", figsize=(20, 10),
                                                                    color=['darkolivegreen', 'darkred', ])
ax = plt.gca()
ax.set_facecolor((0.25, 0.25, 0.25))
plt.legend(['No Heart Disease', 'Has Heart Disease'], prop={'size': 30})
plt.xlabel('Thalassemia', fontsize=30)
plt.xticks(rotation=0)
plt.ylabel('Frequency of Heart Disease', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
locs, labels = plt.xticks()
new_xticks = ['None', 'Normal', 'Fixed defect', 'Reversable defect']
plt.xticks(locs, new_xticks)
plt.title("Correlation between Thallasemia and Heart Disease", fontsize=40)

for p in grap.patches:
    grap.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# ST_Segmen
grap = pd.crosstab(heart_data.ST_Segmen, heart_data.Patient).plot(kind="bar", figsize=(20, 10),
                                                                  color=['darkolivegreen', 'darkred', ])
ax = plt.gca()
ax.set_facecolor((0.25, 0.25, 0.25))
plt.legend(['No Heart Disease', 'Has Heart Disease'], prop={'size': 30})
plt.xlabel('ST_Segmen', fontsize=30)
plt.xticks(rotation=0)
plt.ylabel('Frequency of Heart Disease', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
locs, labels = plt.xticks()
new_xticks = ['Downsloping', 'Flat', 'Upsloping']
plt.xticks(locs, new_xticks)
plt.title("Correlation between st segment and Heart Disease", fontsize=40)

for p in grap.patches:
    grap.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# Fasting blood sugar
grap = pd.crosstab(heart_data.Fasting_BS, heart_data.Patient).plot(kind="bar", figsize=(20, 10),
                                                                   color=['darkolivegreen', 'darkred', ])
ax = plt.gca()
ax.set_facecolor((0.25, 0.25, 0.25))
plt.legend(['No Heart Disease', 'Has Heart Disease'], prop={'size': 30})
plt.xlabel('Fasting blood sugar', fontsize=30)
plt.xticks(rotation=0)
plt.ylabel('Frequency of Heart Disease', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
locs, labels = plt.xticks()
new_xticks = ['True', 'False']
plt.xticks(locs, new_xticks)
plt.title("Correlation between Fasting blood sugar and Heart Disease", fontsize=40)

for p in grap.patches:
    grap.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')

# count male/female desease
pd.crosstab(heart_data.Age, heart_data.Patient).plot(kind="bar", figsize=(20, 10), color=['pink', 'b'])
plt.legend(['Females', 'Males'], prop={'size': 20})

plt.title('Heart Disease Count for Ages', fontsize=40)
plt.xlabel('Age', fontsize=30)
plt.ylabel('Count', fontsize=30)
plt.xticks(fontsize=20, rotation=360)
plt.yticks(fontsize=20)

# correlation between chest pain and heart desease
grap = pd.crosstab(heart_data.Chest_Pain, heart_data.Patient).plot(kind="bar", figsize=(20, 10),
                                                                   color=['olivedrab', 'firebrick'])
ax = plt.gca()
ax.set_facecolor((0.25, 0.25, 0.25))
plt.legend(['No Heart Disease', 'Has Heart Disease'], prop={'size': 30})

plt.xlabel('Chest Pain Type', fontsize=30)
plt.xticks(rotation=0)
plt.ylabel('Frequency of Heart Disease', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
locs, labels = plt.xticks()
new_xticks = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptotic']
plt.xticks(locs, new_xticks)
plt.title("Correlation between Chest Pain and Heart Disease", fontsize=40)

for p in grap.patches:
    grap.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.figure()
sns.scatterplot(x=heart_data['Cholesterol'], y=heart_data['ST_Depression'])
sns.regplot(x=heart_data['Cholesterol'], y=heart_data['ST_Depression'])

# Models full DATA
target = heart_data.Patient
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
print(features.describe())
print(features.head())

# Decision Tree
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
heart_model = DecisionTreeRegressor(max_leaf_nodes=100)
heart_model.fit(train_features, train_target)
val_predictions = heart_model.predict(val_features)

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for full data", fontsize=15)

print(val_predictions)
print(val_target)
print('MAE_regretion_tree:')
MAE_DT = mean_absolute_error(val_target, val_predictions)
print(MAE_DT)
cm = confusion_matrix(val_target, val_predictions)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (Decision tree)", fontsize=15)

# Random forest
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
forest_model = RandomForestRegressor(n_estimators=100, random_state=0)
forest_model.fit(train_features, train_target)
melb_preds = forest_model.predict(val_features)
print('MAE_random_forrest:')
MAE_RF = mean_absolute_error(melb_preds, melb_preds)
print(MAE_RF)

# random forest - cross validation
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=10,
                                                              random_state=0))
                              ])
scores = -1 * cross_val_score(my_pipeline, features, target,
                              cv=10,
                              scoring='neg_mean_absolute_error')

print("MAE cross:\n", scores)
print("Average MAE score (across experiments):")
cross = scores.mean()
print(scores.mean())

# extreme graddient boost
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
my_model = XGBRegressor(n_estimators=1000, learning_rate=1)
my_model.fit(train_features, train_target, early_stopping_rounds=5,
             eval_set=[(val_features, val_target)],
             verbose=False)
predictions = my_model.predict(val_features)
MAE_XGBR = mean_absolute_error(predictions, val_target)
print("MAE XGBR: " + str(MAE_XGBR))

# K nearest Neighbours
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train_features, train_target)
predictions = knn.predict(val_features)
print("MAE knn: " + str(mean_absolute_error(predictions, val_target)))
cm = confusion_matrix(val_target, predictions)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (knn)", fontsize=15)

# SVM
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
svm = SVC(random_state=1, kernel='poly', degree=20)
svm.fit(train_features, train_target)
svm_pred = svm.predict(val_features)

MAE_SVM = mean_absolute_error(svm_pred, val_target)
print("MAE SVC: " + str(MAE_SVM))
cm = confusion_matrix(val_target, svm_pred)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (SVM)", fontsize=15)

# Naive Bayes
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Resting_BP', 'Cholesterol', 'Fasting_BS', 'RECG', 'Max_Heart_Rate',
                  'Exercise_Ang', 'ST_Depression', 'ST_Segmen', 'Major_Vessels', 'Thalassemia']
features = heart_data[heart_features]
nb = GaussianNB()
nb.fit(train_features, train_target)
nb_pred = nb.predict(val_features)

MAE_NB = mean_absolute_error(nb_pred, val_target)
print("MAE Naive Bayes: " + str(MAE_NB))
cm = confusion_matrix(val_target, nb_pred)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (NB)", fontsize=15)

# MAE models
models = ['Decision tree', 'Random Forest', 'Random Forest Cross validation', 'Extreme graddient boost', 'SVM',
          'Naive Bayes']
MAE = [MAE_DT, MAE_RF, cross, MAE_XGBR, MAE_SVM, MAE_NB]
plt.figure()
ax = sns.barplot(models, MAE)

plt.title("MAE of differet models")

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right');

# Models not full DATA
target = heart_data.Patient
heart_features = ['Age', 'Cholesterol', 'Fasting_BS', 'Chest_Pain', 'Max_Heart_Rate',
                  'ST_Segmen']
features = heart_data[heart_features]
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
print(features.describe())
print(features.head())

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for DT", fontsize=15)

# Decision Tree
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
heart_model = DecisionTreeRegressor(max_leaf_nodes=100)
heart_model.fit(train_features, train_target)
val_predictions = heart_model.predict(val_features)
print('MAE_regretion_tree not full DATA:')
MAE_DT = mean_absolute_error(val_target, val_predictions)
print(MAE_DT)
cm = confusion_matrix(val_target, val_predictions)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (Decision tree not full DATA)", fontsize=15)

# Random forest
heart_features = ['Age', 'Cholesterol', 'Fasting_BS', 'Chest_Pain', 'Max_Heart_Rate',
                  'ST_Segmen']
features = heart_data[heart_features]

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE RF", fontsize=15)

train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
forest_model = RandomForestRegressor(n_estimators=100, random_state=0)
forest_model.fit(train_features, train_target)
melb_preds = forest_model.predict(val_features)
print('MAE_random_forrest not full DATA:')
MAE_RF = mean_absolute_error(melb_preds, melb_preds)
print(MAE_RF)

# random forest - cross validation
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Max_Heart_Rate',
                  'ST_Segmen', 'RECG', 'Exercise_Ang']
features = heart_data[heart_features]

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for cross", fontsize=15)

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=10,
                                                              random_state=0))
                              ])
scores = -1 * cross_val_score(my_pipeline, features, target,
                              cv=10,
                              scoring='neg_mean_absolute_error')

print("MAE cross not full DATA:\n", scores)
print("Average MAE score (across experiments not full DATA):")
cross = scores.mean()
print(scores.mean())

# extreme graddient boost
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Max_Heart_Rate',
                  'ST_Segmen']

features = heart_data[heart_features]

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for XGBR", fontsize=15)

train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
my_model = XGBRegressor(n_estimators=1000, learning_rate=1)
my_model.fit(train_features, train_target, early_stopping_rounds=5,
             eval_set=[(val_features, val_target)],
             verbose=False)
predictions = my_model.predict(val_features)
MAE_XGBR = mean_absolute_error(predictions, val_target)
print("MAE XGBR not full DATA: " + str(MAE_XGBR))

# K nearest Neighbours
heart_features = ['Age', 'Chest_Pain', 'Gender', 'Max_Heart_Rate',
                  'ST_Segmen', 'RECG', 'Major_Vessels']

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for KNN", fontsize=15)

features = heart_data[heart_features]
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train_features, train_target)
predictions = knn.predict(val_features)
print("MAE knn not full DATA: " + str(mean_absolute_error(predictions, val_target)))
cm = confusion_matrix(val_target, predictions)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label ', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (knn not full DATA)", fontsize=15)

# SVM
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Max_Heart_Rate',
                  'ST_Segmen', 'Resting_BP', 'Exercise_Ang', 'Thalassemia', 'Major_Vessels']
features = heart_data[heart_features]

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for SVM", fontsize=15)

train_features, val_features, train_target, val_target = train_test_split(features, target, random_state=0)
svm = SVC(random_state=1, kernel='poly', degree=25)
svm.fit(train_features, train_target)
svm_pred = svm.predict(val_features)

MAE_SVM = mean_absolute_error(svm_pred, val_target)
print("MAE SVC not full DATA: " + str(MAE_SVM))
cm = confusion_matrix(val_target, svm_pred)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (SVM not full DATA)", fontsize=15)

# Naive Bayes
heart_features = ['Age', 'Gender', 'Chest_Pain', 'Max_Heart_Rate',
                  'ST_Segmen', 'Resting_BP', 'Thalassemia', 'Major_Vessels']
features = heart_data[heart_features]

X = TSNE(n_components=2, n_iter=1000).fit_transform(features)
plt.figure()
sns.scatterplot(X[:, 0], X[:, 1], hue=target)
plt.title("TSNE for NB", fontsize=15)

nb = GaussianNB()
nb.fit(train_features, train_target)
nb_pred = nb.predict(val_features)

MAE_NB = mean_absolute_error(nb_pred, val_target)
print("MAE Naive Bayes not full DATA: " + str(MAE_NB))
cm = confusion_matrix(val_target, nb_pred)
plt.figure()
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted label', fontsize=10)
plt.xticks(rotation=0)
plt.ylabel('True label', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
locs, labels = plt.xticks()
plt.title("Confusion matrix (NB not full DATA)", fontsize=15)

# MAE models
models = ['Decision tree', 'Random Forest', 'Random Forest Cross validation', 'Extreme graddient boost', 'SVM',
          'Naive Bayes']
MAE = [MAE_DT, MAE_RF, cross, MAE_XGBR, MAE_SVM, MAE_NB]
plt.figure()
ax = sns.barplot(models, MAE)

plt.title("MAE of differet models not full DATA")

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
plt.show()
