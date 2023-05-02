import json
import pandas as pd
import numpy as np
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

def scatter_plot(df : pd.DataFrame, metric):
    df.sort_values(by='tot_visitors')
    #plt.bar(df['tot_visitors'], df[metric])
    plt.scatter(df['tot_visitors'], df[metric])
    plt.xticks(rotation=90)
    plt.title(metric)

def bridges_and_cc(df):
    start = 50; stop = 450; step = 10;
    temp_df = df.groupby(pd.cut(df['tot_visitors'], range(start, stop+step, step))).mean(numeric_only=True)
    plt.bar(range(start, stop, step), temp_df['bridges'             ], align='edge', width=step, color='b', edgecolor='k')
    plt.bar(range(start, stop, step), temp_df['connected_components'], align='edge', width=step, color='r', edgecolor='k')
    
    # CLUSTERING-BASED PLOT
    #temp_df = df.groupby('cluster_id').mean(numeric_only=True)
    #plt.bar(range(min(cluster_id), max(cluster_id)+1, 1), temp_df['bridges'], align='center', width=1, color='b', edgecolor='k')
    #plt.bar(range(min(cluster_id), max(cluster_id)+1, 1), temp_df['connected_components'], align='center', width=1, color='r', edgecolor='k')

    plt.xlabel('Daily visitors', fontsize=13)
    plt.legend(loc='upper right', labels=['Bridges', 'Connected components'])

def plot_metric_by_visitors(df : pd.DataFrame, metric):
    start = 50; stop = 450; step = 10;
    temp_df = df.groupby(pd.cut(df['tot_visitors'], range(start, stop+step, step))).mean(numeric_only=True)
    plt.bar(range(start, stop, step), temp_df[metric], align='edge', width=step, color='b', edgecolor='k')
    plt.xlabel('Daily visitors', fontsize=13)
    plt.ylabel(metric.capitalize() + ' (average)', fontsize=13)

def mean_degree_distribution(df):
    degrees = df['degrees'].apply(lambda str : json.loads(str)).sum()
    plt.hist(degrees, edgecolor='k', bins=10, density=True)
    #plt.yscale('log')
    plt.xlabel(r'$\theta$', fontsize=13)
    plt.ylabel(r'$P(\theta)$', fontsize=13)

def boxplot(series):
    plt.boxplot(series, vert=False)
    plt.vlines(x=series.mean(), color='r', ymin=0, ymax=2)

if __name__ == "__main__":
    # Read df and modify file name (save only "month_day" part of file name) 
    df = pd.read_csv(filepath_or_buffer="metrics.csv", delimiter=',')
    df['file'] = df['file'].str[18:-4]
    df['cluster_id'] = df['cluster_id'].astype(int)
    #df['degrees'] = df['degrees'].apply(lambda str : json.loads(str))
    #df['degrees'] = df['degrees'].apply(lambda ls : np.median(ls))
    
    #plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    plt.tick_params(axis='both', labelsize=10)
    #bridges_and_cc(df)
    plot_metric_by_visitors(df, 'density')
    plt.show()
    
    exit(0)

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
        plt.tight_layout()
        plt.show()