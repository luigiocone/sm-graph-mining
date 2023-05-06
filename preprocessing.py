import os
import shutil
import datetime
import numpy as np
import pandas as pd

DS_ORIGINAL = os.path.join("dataset", "original")
DS_REDUCED = os.path.join("dataset", "reduced")
DS_TRANSFORMED = os.path.join("dataset", "transformed")

def net_summary(dir) -> pd.DataFrame:
    """ Return a dataframe reporting the number of nodes and edges for each network file """
    rows = []
    for file_name in os.listdir(dir):
        if not file_name.startswith("listcontacts"): continue

        # Read current file in dataset directory
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(dir, file_name), 
            delimiter='\t',
            header=None,
            names=['timestamp', 'node1', 'node2']
        )
        
        # Get num of distinct IDs in current file
        n1_ids = df['node1'].unique()
        n2_ids = df['node2'].unique()
        tot_ids = np.union1d(n1_ids, n2_ids)
        
        # Save this data as a row in the final dataframe
        row = {
            'file': file_name,
            'num_nodes': tot_ids.shape[0],
            'num_edges': df.shape[0],
            'weekday': datetime.datetime.strptime(file_name[13:-4], '%Y_%m_%d').strftime('%a')
        }
        rows.append(row)
    return pd.DataFrame(rows)

def reduce_edges(df : pd.DataFrame, duration : int):
    """ Remove connections between the nodes pairs that are less than 'duration' seconds apart """    
    # Compares between sets are order-independent
    df['node_set'] = df.apply(lambda row: set([row['node1'], row['node2']]), axis=1) 
    
    i = 0
    while i < df.shape[0]:
        row = df.iloc[i]
        rm_mask = (
            (df.index > i) & 
            (df['timestamp'] < (row['timestamp'] + duration)) & 
            (df['node_set'] == row['node_set'])
        )
        df = df[~rm_mask]
        df = df.reset_index(drop=True)
        i += 1
    df = df.drop('node_set', axis=1)
    return df

def progressive_node_ids(df : pd.DataFrame):
    """ Replace node IDs with progressive IDs preserving events order """
    nodes = df[['node1', 'node2']].stack().reset_index(drop=True)
    nodes = nodes.drop_duplicates(keep='first')
    node_dict = dict(zip(nodes, np.arange(len(nodes))))
    df[['node1', 'node2']] = df[['node1', 'node2']].applymap(node_dict.get)
    return df

if __name__ == '__main__':
    # Stop execution if output dir already exists
    if os.path.exists(DS_REDUCED) and os.path.exists(DS_TRANSFORMED):
        print(f'Directory "{DS_REDUCED}" and "{DS_TRANSFORMED}" already exists')
        exit(0)
    
    # Since some dir doesn't exists, delete old dirs and repeat all computations
    if os.path.exists(DS_REDUCED    ): shutil.rmtree(DS_REDUCED)
    if os.path.exists(DS_TRANSFORMED): shutil.rmtree(DS_TRANSFORMED)
    os.mkdir(DS_REDUCED)
    os.mkdir(DS_TRANSFORMED)

    # Get a net summary of tsv files (tab-separated-value) and save it in a csv file
    df = net_summary(DS_ORIGINAL)
    df.to_csv(path_or_buf=os.path.join(DS_ORIGINAL, 'netsummary.csv'), index=False)
    print('Total nodes: ' + str(df['num_nodes'].sum()))
    print('Total edges: ' + str(df['num_edges'].sum()))
    
    # Apply reductions and transformations to all dataset files
    for file_name in os.listdir(DS_ORIGINAL):
        if not file_name.startswith("listcontacts"): continue
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(DS_ORIGINAL, file_name), 
            delimiter='\t', # tsv (tab-separated-value) file
            header=None,    # files don't have an header row
            names=['timestamp', 'node1', 'node2']
        )
        
        #### Reductions ####
        # 1) Reduce number of edges of network files and save reduced file
        df = reduce_edges(df, duration=60)
        df.to_csv(path_or_buf=os.path.join(DS_REDUCED, file_name), sep='\t', index=False, header=False)
        
        #### Transformations ####
        # 1) Replace complex IDs with incremental IDs
        df = progressive_node_ids(df)

        # 2) Since the network is undirected, use a standard connection order of the nodes for each row
        df[['node1', 'node2']] = np.sort(df[['node1', 'node2']], axis=1)
        
        # 3) Add an intensity column by grouping duplicated edges. The first timestamp will be used as a "visitor entry time" 
        df['intensity'] = pd.Series(1, df.index)
        df = df.groupby(['node1', 'node2'], as_index=False, sort=False).agg({'intensity':'sum', 'timestamp':'first'})
        df.to_csv(path_or_buf=os.path.join(DS_TRANSFORMED, file_name[:-3] + 'csv'), index=False, header=True, columns=['timestamp', 'node1', 'node2', 'intensity'])
        print("File '" + file_name + "' reduced and transformed")
    
    df = net_summary(DS_REDUCED)
    df.to_csv(path_or_buf=os.path.join(DS_REDUCED, 'netsummary_(reduced).csv'), index=False)
