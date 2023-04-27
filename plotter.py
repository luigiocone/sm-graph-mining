from clustering import CLUSTERING_FILE
import json
import pandas as pd
import matplotlib.pyplot as plt

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
NUM_CLUSTER = len(mc_pairs)

if __name__ == "__main__":
    # Read df and modify file name (save only "month_day" part of file name) 
    df = pd.read_csv(filepath_or_buffer="metrics.csv", delimiter=',')
    df['file'] = df['file'].str[18:-4]

    # Read clusters from a json file
    with open(CLUSTERING_FILE) as file:
        clusters = json.load(file)
    clusters = list(clusters.values())

    # For each metric
    metrics = df.columns
    for i in range(1, len(metrics)):
        metric = metrics[i]
        # For each cluster
        for j in range(NUM_CLUSTER):
            mask = df['file'].isin(clusters[j])
            temp_df = df[mask]
            plt.scatter(temp_df['file'], temp_df[metric], marker=mc_pairs[j][0], c=mc_pairs[j][1])
        plt.xticks(rotation=90)
        plt.title(metric)
        plt.subplots_adjust(left=0.065, right=0.945, top=0.95, bottom=0.145)
        plt.show()