import os
import numpy as np
import pandas as pd
from variables import csv_path, label_encode, file_name, cutoff
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df['label'] = df.apply(y2indicator, axis=1)
    del df['species']
    df.to_csv(file_name, encoding='utf-8')

def y2indicator(x):
    species = x['species']
    return label_encode[species]

def get_data():
    if not os.path.exists(file_name):
        preprocess_data(csv_path)
    df = pd.read_csv(file_name)
    df = shuffle(df)
    Xdata = df.copy()[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    Ydata = df.copy()[['label']].to_numpy()
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(Xdata)
    train_set = int(cutoff * len(df))
    Xtrain, Xtest = scaled_X[:train_set], scaled_X[train_set:]
    Ytrain, Ytest = Ydata[:train_set], Ydata[train_set:]
    return  Xtrain, Xtest, Ytrain, Ytest
