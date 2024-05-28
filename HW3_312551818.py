import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

def normalize(values):
    min_max_scaler = preprocessing.MinMaxScaler()
    return(min_max_scaler.fit_transform(values))

def manhattan_distance(origin_point, end_point):
    res = 0
    for i in range(len(origin_point)):
        res += abs(origin_point[i]-end_point[i])
    return res

def euclidian_distance(origin_point, end_point):
    res = 0
    for i in range(len(origin_point)):
        res += (origin_point[i]-end_point[i])**2
    return res**(1/2)

def distance_matrix(df, distance_method):
    distance_matrix = np.zeros((len(df),len(df)))
    print("Computing distance matrix...")
    for i in range(len(df)):
        for j in range(i+1,df.shape[0]):
            distance = distance_method(df[i],df[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    print("Finished computing distance matrix!")
    return distance_matrix

def compute_avg_distances(matrix, k):
    avg_dist_list = []
    for i in range(len(matrix)):
        distances = matrix[i]
        # We exclude the first value that is equal to 0
        distances = sorted(distances)[1:k+1]
        avg_dist_list.append(np.mean(distances))
    return avg_dist_list

def plot_hist(list):
    plt.hist(list,bins=10)
    plt.show()
    plt.hist(list[4200:],bins=10)
    plt.show()

# Formats and exports the answers to a CSV file 
def format_export_answers(answers, k_value):
    res = input("Do you want to output the answers (Y/N)")
    if res == "N":
        return
    elif res == "Y":
        print("Formating and writing answers in csv file...")
        fields = ["id","outliers"]
        rows = []
        for i in range(len(answers)):
            rows.append([i, answers[i]])

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        with open('answer_' + str(dt_string) + "_" +str(k_value)+".csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
        print("Succesfully formated and wrote answers in a csv file !")
    else:
        format_export_answers(answers, k_value)

def main():
    train_set = "training.csv"
    test_set = "test_X.csv"

    df_train = pd.read_csv(train_set)
    df_test = pd.read_csv(test_set)

    # Concatenates both our training and testing dataset since we won't go with a model training method
    df_train = df_train.drop('lettr', axis=1)
    df = pd.concat([df_train,df_test])
    # Converts our concatenated dataframe to a numpy list
    df_values = df.values
    df_values = normalize(df_values)

    # Number of neighbors to consider for the computation of the average distance
    k = 3

    results = compute_avg_distances(distance_matrix(df_values,euclidian_distance),k)

    format_export_answers(results[4200:],k)

if __name__ == '__main__':
    main()