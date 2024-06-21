import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def normalize(values):
    scaler = MinMaxScaler()
    return scaler.fit_transform(values)

def calculate_distances(x_train, x_test, k):
    # Using NearestNeighbors to find the k-nearest neighbors and their distances
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='manhattan').fit(x_train)
    distances, indices = nbrs.kneighbors(x_test)
    return distances

def format_export_answers(answers, k_value):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'answer_' + str(dt_string) + "_" + str(k_value) + ".csv"
    
    print("Formating and writing answers in csv file...")
    fields = ["id", "outliers"]
    rows = [[i, answers[i]] for i in range(len(answers))]

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    
    print(f"Succesfully formated and wrote answers in a csv file: {filename}")

def main():
    train_set = "training.csv"
    test_set = "test_X.csv"

    df_train = pd.read_csv(train_set)
    df_test = pd.read_csv(test_set)

    y_train = df_train['lettr'].values
    df_train = df_train.drop('lettr', axis=1)

    df = pd.concat([df_train, df_test])
    df_values = df.values
    df_values = normalize(df_values)

    x_train = df_values[:len(y_train)]
    x_test = df_values[len(y_train):]

    # Number of neighbors to consider for the computation of the average distance
    k = 3

    # Calculate distances
    distances = calculate_distances(x_train, x_test, k)
    avg_distances = distances.mean(axis=1)

    format_export_answers(avg_distances, k)

if __name__ == '__main__':
    main()
