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

def scatter_plot(df : pd.DataFrame, metric, pair):
    df.sort_values(by='num_nodes')
    #plt.bar(df['num_nodes'], df[metric])
    plt.scatter(df['num_nodes'], df[metric], marker=mc_pairs[pair][0], c=mc_pairs[pair][1])
    plt.xticks(rotation=90)
    plt.title(metric)

def multiple_scatter_plot(df : pd.DataFrame, metrics, ax):
    df.sort_values(by='num_nodes')
    plt.grid(linestyle = '--', linewidth = 0.2, alpha = 0.5)
    ax.set_axisbelow(True)
    for i in range(len(metrics)):
        ax.scatter(df['num_nodes'], df[metrics[i]], marker=mc_pairs[i][0], c=mc_pairs[i][1])

    # set major ticks to black and larger size
    # set the positions of the minor ticks
    ax.set_xticks(np.arange(75, 450, 25), minor=True)
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)
    plt.xticks(np.arange(100, 450, 50))
    plt.xlabel('Daily visitors')


def bridges_and_cc(df):
    start = 50; stop = 450; step = 10;
    temp_df = df.groupby(pd.cut(df['num_nodes'], range(start, stop+step, step))).mean(numeric_only=True)
    plt.bar(range(start, stop, step), temp_df['bridges'             ], align='edge', width=step, color='b', edgecolor='k')
    plt.bar(range(start, stop, step), temp_df['connected_components'], align='edge', width=step, color='r', edgecolor='k')
    plt.bar(range(start, stop, step), temp_df['brokers'             ], align='edge', width=step, color='forestgreen', edgecolor='k')
    
    #plt.bar(range(start, stop, step), temp_df['avg_global_clustering_coeff'             ], align='edge', width=step, color='forestgreen', edgecolor='k')
    #plt.bar(range(start, stop, step), temp_df['avg_local_clustering_coeff'], align='edge', width=step, color='r', edgecolor='k')
    #plt.bar(range(start, stop, step), temp_df['avg_betweenness']  / temp_df['avg_betweenness'].max(), align='edge', width=step, color='b', edgecolor='k')

    # CLUSTERING-BASED PLOT
    #temp_df = df.groupby('cluster_id').mean(numeric_only=True)
    #plt.bar(range(min(cluster_id), max(cluster_id)+1, 1), temp_df['bridges'], align='center', width=1, color='b', edgecolor='k')
    #plt.bar(range(min(cluster_id), max(cluster_id)+1, 1), temp_df['connected_components'], align='center', width=1, color='r', edgecolor='k')

    plt.xlabel('Daily visitors', fontsize=13)
    plt.legend(loc='upper right', labels=['Bridges', 'Connected components', r'Brokers ($\theta_i \leq 3$, \ $\beta_i \geq \mbox{quantile}(\beta, \ 0.85))$'])

def grouped_barchart(df):
    start = 50; stop = 450; step = 50;
    temp_df = df.groupby(pd.cut(df['num_nodes'], range(start, stop+step, step))).mean(numeric_only=True)

    x = np.arange(50, 450, 50)
    plt.bar(x-15, temp_df['avg_betweenness']  / temp_df['avg_betweenness'].max(), align='edge', width=10, edgecolor='k')
    plt.bar(x-5,  temp_df['avg_local_clustering_coeff'] , align='edge', width=10, edgecolor='k')
    plt.bar(x+5,  temp_df['avg_global_clustering_coeff'], align='edge', width=10, edgecolor='k')
    plt.xlabel('Daily visitors', fontsize=13)
    plt.legend(loc='upper left', labels=['Betweenness', 'Local clustering coeff.', 'Global clustering coeff.'])


def plot_metric_by_visitors(df : pd.DataFrame, metric):
    start = 50; stop = 450; step = 10;
    temp_df = df.groupby(pd.cut(df['num_nodes'], range(start, stop+step, step))).mean(numeric_only=True)
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
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    fig, ax = plt.subplots()
    plt.tick_params(axis='both', labelsize=10)
    
    #df['cluster_id'] = df['cluster_id'].astype(int)
    #df['degrees'] = df['degrees'].apply(lambda str : json.loads(str))
    #df['degrees'] = df['degrees'].apply(lambda ls : np.median(ls))
    
    #grouped_barchart(df)
    df['med_betweenness'] = df['med_betweenness']  / df['med_betweenness'].max()
    multiple_scatter_plot(df, ['med_betweenness', 'avg_local_clustering_coeff', 'density'], ax)
    plt.legend(loc='upper left', labels=['Betweenness (normalized)', 'Local clustering coefficient', 'Density'])
    plt.tight_layout()
    plt.show()

"""
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
"""