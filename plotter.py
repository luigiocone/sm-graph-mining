import json
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Marker color pairs
mc_pairs = [
    ['.', 'tab:blue'],
    ['+', 'tab:orange'],
    ['x', 'tab:green'],
    ['s', 'tab:red'],
    ['^', 'tab:purple'],
    ['P', 'tab:brown'],
    ['*', 'tab:gray'],
    ['1', 'tab:olive'],
    ['X', 'tab:cyan']
]
NUM_CLUSTER = len(mc_pairs)

def multiple_scatter_plot(df : pd.DataFrame, metrics):
    df.sort_values(by='num_nodes')
    for i in range(len(metrics)):
        plt.scatter(df['num_nodes'], df[metrics[i]], marker=mc_pairs[i][0], c=mc_pairs[i][1])

    # set the positions of the minor ticks and major ticks
    plt.xticks(np.arange(75, 450, 25), minor=True)
    plt.xticks(np.arange(100, 450, 50))
    plt.xlabel('Daily visitors', fontsize=13)


def bridges_brokers_cc(df):
    start = 50; stop = 450; step = 10;
    temp_df = df.groupby(pd.cut(df['num_nodes'], range(start, stop+step, step))).mean(numeric_only=True)
    plt.bar(range(start, stop, step), temp_df['bridges'             ], align='edge', width=step, color='b', edgecolor='k')
    plt.bar(range(start, stop, step), temp_df['connected_components'], align='edge', width=step, color='r', edgecolor='k')
    plt.bar(range(start, stop, step), temp_df['brokers'             ], align='edge', width=step, color='forestgreen', edgecolor='k')

    plt.xlabel('Daily visitors', fontsize=13)
    plt.legend(loc='upper right', labels=['Bridges', 'Connected components', r'Brokers ($\theta_i \leq 3$, \ $\beta_i \geq \mbox{quantile}(\beta, \ 0.85))$'])


def grouped_barchart(df):
    """ Not used """
    start = 50; stop = 450; step = 50;
    temp_df = df.groupby(pd.cut(df['num_nodes'], range(start, stop+step, step))).mean(numeric_only=True)

    x = np.arange(50, 450, 50)
    plt.bar(x-15, temp_df['median_betweenness']  / temp_df['median_betweenness'].max(), align='edge', width=10, edgecolor='k')
    plt.bar(x-5,  temp_df['local_clustering_coeff'] , align='edge', width=10, edgecolor='k')
    plt.bar(x+5,  temp_df['global_clustering_coeff'], align='edge', width=10, edgecolor='k')
    plt.xlabel('Daily visitors', fontsize=13)
    plt.legend(loc='upper left', labels=['Betweenness', 'Local clustering coeff.', 'Global clustering coeff.'])


def btw_lcc_density(df):
    df['median_betweenness'] = df['median_betweenness'] / df['median_betweenness'].max()
    df['median_katz'] = df['median_katz'] / df['median_katz'].max()
    multiple_scatter_plot(df, ['median_betweenness', 'mean_local_clustering_coeff', 'density'])
    plt.legend(loc='upper left', labels=['Betweenness (normalized)', 'Local clustering coefficient', 'Density'])


def katz_over_time(df):
    xticks = np.arange(0, df.shape[0], 1)
    df['file'] = df['file'].str.replace('_', '-')
    plt.plot(df['file'], df['mean_katz'], linewidth='2')
    plt.plot(df['file'], df['largest_cc_mean_degree'], linewidth='2', color='salmon')
    # plt.plot(df['file'], df['largest_cc_density']*df['largest_cc_num_nodes'], linewidth='2', color='g')
    
    plt.xticks(xticks, rotation=90, fontsize=7, labels=df['file'])
    plt.xlabel(r"$t$", fontsize=13)
    plt.legend(labels=[r'mean$(c_K)$', r'mean$(\theta)$'], fontsize=11)


def katz_and_degree(df, day_index):
    day_index=0
    lcc_degrees = df.loc[day_index, 'largest_cc_degree']
    lcc_degrees = [x/15 for x in lcc_degrees]
    plt.plot(df.loc[day_index, 'katz'])
    plt.plot(lcc_degrees, color='salmon')
    
    plt.xticks(np.arange(0, 200, 25))
    plt.xticks(np.arange(0, 200, 12.5), minor=True)
    plt.xlabel(r"Visitor ID", fontsize=13)
    plt.legend(labels=[r'$c_K$', r'$\theta/15$'], fontsize=11)

def katz_by_nodes(df):
    plt.scatter(df['num_nodes'], df['median_katz'], edgecolor='k', color='r')
    plt.xticks(np.arange(75, 425, 25), minor=True)
    #plt.yticks(np.arange(3, 14, 1), minor=True)
    plt.grid(linestyle='--', linewidth=0.2, alpha=0.5)
    plt.xlabel("Daily visitors", fontsize=13)
    plt.ylabel(r"Katz (median)", fontsize=13)

def hubs_by_nodes(df):
    df = df.sort_values(by='num_nodes')
    df["hubs>=10"] = df["degree"].apply(lambda x: len([i for i in x if i >= 10]))
    df["hubs>=20"] = df["degree"].apply(lambda x: len([i for i in x if i >= 20]))
    df["hubs>=30"] = df["degree"].apply(lambda x: len([i for i in x if i >= 30]))
    
    # Linear regression
    slope10, intercept10, _, _, _ = linregress(df['num_nodes'], df['hubs>=10'])
    slope30, intercept30, _, _, _ = linregress(df['num_nodes'], df['hubs>=30'])

    # Non linear regression
    popt, pcov = curve_fit(pos_exponential, df['num_nodes'], df['hubs>=20'], p0=[0.1, 0.02, 0.01])
    exp_latex = str(round(popt[0], 2)) + 'e^{' + str(round(popt[1], 2)) + 'n} ' + str(round(popt[2], 2))
    x = np.arange(df['num_nodes'].min(), df['num_nodes'].max(), 0.1) # For a smooth curve

    # Plots of regression models
    plt.plot(df['num_nodes'], slope10*df['num_nodes'] + intercept10, 'r-', zorder=1)
    plt.plot(x, pos_exponential(x, *popt), 'g-', zorder=1)
    plt.plot(df['num_nodes'], slope30*df['num_nodes'] + intercept30, 'b-', zorder=1)

    # Data points
    plt.scatter(df['num_nodes'], df['hubs>=10'], edgecolor='k', color='r', zorder=2)
    plt.scatter(df['num_nodes'], df['hubs>=20'], edgecolor='k', color='g', zorder=2)
    plt.scatter(df['num_nodes'], df['hubs>=30'], edgecolor='k', color='b', zorder=2)

    plt.xticks(np.arange(75, 425, 25), minor=True)
    plt.xlabel("Daily visitors", fontsize=13)
    plt.ylabel(r"Hubs", fontsize=13)
    plt.legend(fontsize=11, labels=[
        r'$\theta \geq 10; \quad h_{10}(n) =' + f'{str(round(slope10, 2))}n {str(round(intercept10, 2))}$', 
        r'$\theta \geq 20; \quad h_{20}(n) =' + f'{exp_latex})$',
        r'$\theta \geq 30; \quad h_{30}(n) =' + f'{str(round(slope30, 2))}n {str(round(intercept30, 2))}$', 
    ])


def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def pos_exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def diameter_by_closeness(df):
    temp_df = df.sort_values(by='median_closeness')
    popt, pcov = curve_fit(exponential, temp_df['median_closeness'], temp_df['diameter'])
    plt.plot(temp_df['median_closeness'], exponential(temp_df['median_closeness'], *popt), 'r-')
    plt.plot(temp_df['median_closeness'], temp_df['diameter'], linestyle='', marker='o', color='b', markeredgecolor='k')
    plt.xlabel(r'Closeness $c_\ell$ (median value for each day)', fontsize=13)
    plt.ylabel(r'Diameter $d$', fontsize=13)
    
    # Legend label and icon
    exp_latex = r'$ d(c_\ell) =' + str(round(popt[0], 2)) + 'e^{-' + str(round(popt[1], 2)) + r'c_\ell} +' + str(round(popt[2], 2)) + '$'
    handle = plt.Line2D([], [], linestyle='-', linewidth=1.5, marker='', color='r')
    plt.legend(loc='upper right', labels=[exp_latex], handles=[handle], fontsize=13)


def plot_correlation_matrix(df):
    corr_matrix = df.corr(numeric_only=True)
    sn.heatmap(corr_matrix, annot=True)


def mean_degree_distribution(df: pd.DataFrame, day_index):
    # Get degrees of one graph:
    degrees = pd.Series(df.loc[day_index, 'degree'])
    day = df.loc[day_index, 'file'].replace('_', '-')
    
    """
    # Compute alpha and beta
    num_nodes = df.loc[0, 'num_nodes']
    alpha = 0.95
    beta = 1/np.sum(np.power(np.arange(1, num_nodes, 1), -alpha))
    # Plot the curve of: beta*theta^(-alpha)
    du = degrees.unique()
    du.sort()
    plt.plot(du, beta*np.power(du, -alpha), 'r-')
    """
    
    # 'normalize=True' to obtain frequencies insted of count
    freqs = degrees.value_counts(normalize=True)
    # plt.stem(freqs.index, freqs.values, basefmt=' ', markerfmt='o', markerfacecolor='none')
    
    # Build a lollipop chart
    plt.plot(freqs.index, freqs.values, color='tab:blue', marker='o', linestyle='None', markerfacecolor='none')
    plt.vlines(freqs.index, ymin=0, ymax=freqs.values, colors='tab:blue')
    
    plt.ylim(bottom=0, top=max(freqs.values)+0.005)
    plt.xticks(np.arange(1, max(freqs.index)+2, 1), minor=True)
    plt.xticks(np.arange(1, max(freqs.index)+2, 2))
    plt.xlabel(r'$\theta$', fontsize=13)
    plt.ylabel(r'$p(\theta)$ of ' + day, fontsize=13)


def max_degree_densities(df: pd.DataFrame):
    df['degree_densities'] = df['degree'].apply(lambda x: pd.value_counts(x, normalize=True).to_dict())
    df['degree_densities'] = df['degree_densities'].apply(lambda x: max(x.items(), key=lambda k: k[1]))
    temp = df['degree_densities'].apply(lambda x: x[0])

    c = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    density = 0.1
    for i in range(len(c)):
        temp = df[df['degree_densities'].apply(lambda x: x[1] <= density and x[1] > density-0.05)]
        plt.scatter(temp['num_nodes'], temp['degree_densities'].apply(lambda x: x[0]), color=c[i], edgecolor='k')
        density += 0.05
    temp = df[df['degree_densities'].apply(lambda x: x[1] > density-0.05)]
    plt.scatter(temp['num_nodes'], temp['degree_densities'].apply(lambda x: x[0]), color=c[-1], edgecolor='k')
    
    # Legend
    labels = [
        r'$ p(\theta) \leq 0.1$', 
        r'$ 0.10 < p(\theta) \leq 0.15$', 
        r'$ 0.15 < p(\theta) \leq 0.20$',
        r'$ p(\theta) > 0.20$'
    ]
    
    handles = []
    for i in range(len(c)):
        handles.append(plt.Line2D([], [], linestyle='', marker='o', color=c[i], markeredgecolor='k', markersize=10))
    
    plt.xlabel(r'Daily visitors', fontsize=13)
    plt.xticks(np.arange(100, 400, 25), minor=True)
    plt.ylabel(r'Degree $\theta$ having max density', fontsize=13)
    plt.legend(handles=handles, labels=labels, fontsize=11)


def boxplot(series, mean):
    plt.boxplot(series, vert=False)
    plt.vlines(x=mean, color='r', ymin=0, ymax=2)


def visit_duration_hist(df, day_index):
    data = df.loc[day_index, 'nodes_visit_duration']
    bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num=30)
    hist, _ = np.histogram(data, bins=bins)
    plt.xscale('log')
    plt.xlabel(r'Visit duration', fontsize=13)
    plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', edgecolor='black')


def intervals_count(series):
    minutes = 20
    interval = pd.Timedelta(minutes=minutes)

    # Create a list by summing (appending) all lists of entry times
    entry_times = series.sum()
    entry_times = pd.to_datetime(entry_times, format='%H:%M')
    entry_times = pd.Series(entry_times)
    intervals = entry_times.dt.round(f'{minutes}min')
    
    # Count how many nodes have entered in a certain time interval and change key string to the starting entry time
    counts = intervals.value_counts().to_dict()
    counts = {(key - interval/2).strftime('%H:%M'): value for key, value in counts.items()}
    
    # Fill the dictionary for the keys having no occurrence 
    start_time = '08:50'
    end_time = '18:30'
    curr = pd.to_datetime(start_time, format='%H:%M')
    while curr <= pd.to_datetime(end_time, format='%H:%M'):
        key = curr.strftime('%H:%M')
        curr += interval
        if key in counts: continue
        counts[key] = 0
    counts = dict(sorted(counts.items()))
    return counts

def visitors_entry_times(df):    
    # Get counts of visitors entry times interval
    visitors_et = intervals_count(df['nodes_entry_time'])
    brokers_et = intervals_count(df['brokers_entry_time'])
    brokers_et = {key:value*50 for key, value in brokers_et.items()}
    del visitors_et['19:30'] # Single outlier
    
    bar_width = 1
    plt.bar(visitors_et.keys(), visitors_et.values(), color='tab:blue', edgecolor='k', align='edge', width=bar_width)
    plt.bar(brokers_et.keys(), brokers_et.values(), color='tab:green', edgecolor='k', align='edge', width=bar_width)
    #plt.plot(visitors_et.keys(), visitors_et.values(), color='tab:blue', marker='o')
    #plt.plot(brokers_et.keys(), brokers_et.values(), color='tab:green', marker='o')
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel(r"Entry time", fontsize=13)
    plt.ylabel(r"Count", fontsize=13)
    plt.legend(labels=[r"Visitors", r"Brokers (scaled $\times 50$)"])
    
    # calculate asymmetry of x and y axes (used to draw a circle)
    ax = plt.gca()
    x0, y0 = ax.transAxes.transform((0, 0)) # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1)) # upper right in pixes
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    width = 0.75 * maxd / dx
    height = 20 * maxd / dy
    
    shift_x = bar_width/2
    x = -1 + shift_x
    shift_y = 20
    for _, value in brokers_et.items():
        x += 1
        if value == 0: continue
        circle = Ellipse((x, value-shift_y), width, height, facecolor='white', edgecolor='black')
        plt.gca().add_patch(circle)
        plt.text(x, value-shift_y-1, str(round(value/50)), ha='center', va='center', fontsize=8)


def non_infected_cc(df):
    temp_df = df[["num_nodes", "connected_components_size"]].copy()
    # Remove the largest connected component size
    temp_df['connected_components_size'] = temp_df['connected_components_size'].apply(lambda x: x.remove(max(x)) or x)
    temp_df = temp_df[temp_df['connected_components_size'].apply(lambda x: len(x) > 0)]
    temp_df['connected_components_size'] = temp_df['connected_components_size'].apply(lambda x: max(x))
    temp_df.sort_values(by='num_nodes')
    plt.scatter(temp_df['num_nodes'], temp_df['connected_components_size'],  color='tab:green', edgecolor='k')
    plt.xlabel(r'Daily visitors', fontsize=13)
    plt.ylabel(r'Length of non-infected components', fontsize=13)
    plt.yticks(np.arange(0, 90, 10), minor=True)

def btw_lcc_density(df):
    """ Not used"""
    multiple_scatter_plot(df, ['mean_betweenness', 'mean_local_clustering_coeff', 'density'])
    plt.legend(labels=['Betweenness (mean)', 'Local clustering coeff. (mean)', 'Density'])


if __name__ == "__main__":
    # Read df and modify file name (save only "month_day" part of file name)
    with open('metrics.json', 'r') as metrics:
        data = json.load(metrics)
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(level=0, inplace=True, names=['file'])    
    df['file'] = df['file'].str[18:-4]
    
    # Plt options
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    fig, ax = plt.subplots()
    plt.tick_params(axis='both', labelsize=10)
    
    # Summarize metrics of interests
    df["brokers"] = df["brokers"].apply(lambda x: len(x))
    df["connected_components"] = df["connected_components_size"].apply(lambda x: len(x))
    df["median_betweenness"] = df["betweenness"].apply(lambda x: np.median(x))
    df["median_betweenness"] = df["median_betweenness"] / np.power(df['num_nodes'], 2)
    df["mean_betweenness"] = df["betweenness"].apply(lambda x: np.mean(x))
    df["mean_betweenness"] = df["mean_betweenness"] / np.power(df['num_nodes'], 2)
    df['median_closeness'] = df['closeness'].apply(lambda x: np.median(x))
    df['median_katz'] = df['katz'].apply(lambda x: np.median(x))
    df['mean_katz'] = df['katz'].apply(lambda x: np.mean(x))
    df["mean_local_clustering_coeff"] = df["local_clustering_coeff"].apply(lambda x: np.mean(x))
    df['mean_degree'] = df['degree'].apply(lambda x: np.mean(x))
    df['largest_cc_degree'] = df.apply(lambda row: [row['degree'][i] for i in row['largest_connected_component']], axis=1)
    df['largest_cc_mean_degree'] = df['largest_cc_degree'].apply(lambda x: np.mean(x))
    df['nodes_entry_hour'] = df['nodes_entry_time'].apply(lambda lst: [int(x[:2]) for x in lst])
    
    day_index = 0                                 # 0=Apr_28; 5=May_3; 6=May_5;
    # bridges_brokers_cc(df)                      # Fig. 2
    # non_infected_cc(df)                         # Fig. 3a
    # visitors_entry_times(df)                    # Fig. 3b
    # diameter_by_closeness(df)                   # Fig. 3c
    # mean_degree_distribution(df, day_index)     # Fig. 5a
    # max_degree_densities(df)                    # Fig. 5c
    # katz_and_degree(df, day_index)              # Fig. 6a
    # katz_over_time(df)                          # Fig. 6b
    # hubs_by_nodes(df)                           # Fig. 6c
    # visit_duration_hist(df, day_index)          # Fig. 8a
    
    # OTHER ANALYSIS
    # boxplot(df.loc[day_index, 'closeness'], np.mean(df.loc[day_index, 'closeness']))
    # plot_correlation_matrix(df)
    
    print("Day index: " + str(day_index) + " | Day: " + df.loc[day_index, 'file'])
    plt.grid(linestyle='--', linewidth=0.2)
    plt.tight_layout()
    plt.show()
