import pandas as pd
import matplotlib.pyplot as plt
from clusters import clusters

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

for i in range(6):
    metric = df.columns[i+1]
    for i in range(len(mc_pairs)):
        mask = df['file'].isin(clusters[i])
        temp_df = df[mask]
        plt.scatter(temp_df['file'], temp_df[metric], marker=mc_pairs[i][0], c=mc_pairs[i][1])
    plt.xticks(rotation=90)
    plt.title(metric)
    plt.subplots_adjust(left=0.065, right=0.945, top=0.95, bottom=0.145)
    plt.show()