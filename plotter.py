import json
import pandas as pd
import matplotlib.pyplot as plt
from clustering import CLUSTERING_FILE

NUM_CLUSTER = 5

# Marker color pairs
mc_pairs = [
    ['.', 'tab:blue'],
    ['^', 'tab:orange'],
    ['1', 'tab:green'],
    
    ['s', 'tab:red'],
    ['+', 'tab:purple'],
    ['P', 'tab:brown'],
    
    ['*', 'tab:gray'],
    ['x', 'tab:olive'],
    ['X', 'tab:cyan']
]

df = pd.read_csv(filepath_or_buffer="metrics.csv", delimiter=',')
df['file'] = df['file'].str[13:-4]

# Picking the 5 largest clusters 
with open(CLUSTERING_FILE) as file:
    clusters = json.load(file)
clusters = list(clusters.values())
clusters.sort(key = len, reverse=True)
clusters = clusters[:NUM_CLUSTER]

# For each metric
for i in range(1, 7):
    metric = df.columns[i]
    # For each cluster
    for j in range(NUM_CLUSTER):
        temp_df = df.filter(items=clusters[j], axis=0)
        plt.scatter(temp_df['file'], temp_df[metric], marker=mc_pairs[j][0], c=mc_pairs[j][1])
    plt.xticks(rotation=90)
    plt.title(metric)
    plt.subplots_adjust(left=0.065, right=0.945, top=0.95, bottom=0.145)
    plt.show()