import numpy as np
import pandas as pd
import networkx as nx
import data_utils as data

def construct_graph(data_path, out_path, report_stats=True):
    """Constructs graph"""
    df = pd.read_csv(data_path)
    # Get user nodes
    user_nodes = data.get_user_nodes(df)
    N = len(user_nodes)
    print("Constructing graph from {} users".format(N))
    g = nx.Graph()
    added_users = set()
    for i in range(N):
        user1 = user_nodes[i]
        if user1[0] not in added_users:
            g.add_node(user1[0])
        for j in range(i+1, N):
            user2 = user_nodes[j]
            if user2[0] not in added_users:
                g.add_node(user2[0])
            if link_nodes(user_nodes[i], user_nodes[j]):
                g.add_edge(user1[0], user2[0])


    # Save graph
    nx.readwrite.edgelist.write_edgelist(g, out_path)

    if report_stats:
        print("Graph contains {} edges".format(g.number_of_edges()))
        data.report_hash_tag_stats(user_nodes)

    return g


def link_nodes(node1, node2, thresh=1):
    """Comparison function. Returns True if node1 and node2 should be linked"""
    if len(node1[1].intersection(node2[1])) >= thresh:  # Users have at least thresh tags in common
        return True
    return False


def main():
    construct_graph('backbones/dataset.csv', 'user_user_graph.txt')

if __name__ == '__main__':
    main()
