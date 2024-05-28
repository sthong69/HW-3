import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_set = "training.csv"
test_set = "test_X.csv"

df_train = pd.read_csv(train_set)
df_test = pd.read_csv(test_set)

df_train = df_train.drop('lettr', axis=1)

df = pd.concat([df_train,df_test])
df_values = df.values

distance_matrix = np.zeros((df.shape[0],df.shape[0]))

def manhattan_distance(origin_point, end_point):
    res = 0
    for i in range(len(origin_point)):
        res += abs(origin_point[i]-end_point[i])
    return res

print("Computing distance matrix...")
for i in range(df.shape[0]):
    for j in range(i+1,df.shape[0]):
        distance = manhattan_distance(df_values[i],df_values[j])
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance
print("Finished computing distance matrix!")

# Number of neighbors to consider for the average distance
k = 3
avg_dist_list = []
for i in range(df.shape[0]):
    distances = distance_matrix[i]
    distances = sorted(distances)[1:k+1]
    avg_dist_list.append(np.mean(distances))


plt.hist(avg_dist_list,bins=10)
plt.show()
plt.hist(avg_dist_list[4200:],bins=10)
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

        with open('answer_' + str(dt_string) + "_" +str(k)+".csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
        print("Succesfully formated and wrote answers in a csv file !")
    else:
        format_export_answers(answers, k_value)

format_export_answers(avg_dist_list[4200:],k)