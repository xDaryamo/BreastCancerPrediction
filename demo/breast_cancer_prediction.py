import pandas as pd
import numpy as np
import os
from numpy import set_printoptions

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, mutual_info_classif, chi2
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, plot_confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

to_delete = []
scaler = StandardScaler()

def remove_collinear_features(x, threshold):
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                if(x[[col.values[0]]].corrwith(x.diagnosis).iloc[0] >= x[[row.values[0]]].corrwith(x.diagnosis).iloc[0]):
                  if row.values[0] not in drop_cols:
                    drop_cols.append(row.values[0])
                else:
                  if col.values[0] not in drop_cols:
                    drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    return drop_cols



def get_trained_model():
    csv = "https://raw.githubusercontent.com/xDaryamo/BreastCancerPrediction/master/dataset.csv"
    # Caricamento del csv dalla repo GitHub
    df = pd.read_csv(csv)
    df.diagnosis.replace({"M": 1, "B": 0}, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)
    corr_matrix = df.corr()
    threshold = 0.1
    filtre = np.abs(corr_matrix["diagnosis"]) <= threshold
    to_delete.extend(corr_matrix.columns[filtre].tolist())
    drop_cols = remove_collinear_features(df, 0.9)
    to_delete.extend(drop_cols)
    df = df.drop(to_delete, axis=1)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    # Split del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    return logisticRegr

def load_demo_csv(path):
  df_demo = pd.read_csv(path)
  id = df_demo["id"].tolist()
  return df_demo,id

def normalize_ds(ds):
  ds.drop(to_delete, axis = 1, inplace=True)
  ds=scaler.fit_transform(ds) 
  array = np.asarray(ds)
  return array



def get_predictions(path):
    ds_demo, id = load_demo_csv(path)
    model = get_trained_model()
    predictions = model.predict(normalize_ds(ds_demo))

    data = {'Id paziente': id, 'Diagnosi': predictions}

    result = pd.DataFrame(data)

    result.Diagnosi.replace({1:"Maligno",0:"Benigno"},inplace=True)
    pd.set_option('display.max_rows', None)
    return result