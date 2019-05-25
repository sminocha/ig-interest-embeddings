from sklearn.cluster import KMeans
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import networkx as nx


def run_kmeans(embeddings, k=10):
    kmeans = KMeans(n_clusters=k).fit(embeddings)
    labels = kmeans.labels_
    return labels


def main():
    # Count communities
    g = nx.readwrite.edgelist.read_edgelist('user_user_graph.txt')
    communities = nx.community.greedy_modularity_communities(g)
    print(len(communities))
    # Load embeddings

    wv = KeyedVectors.load_word2vec_format("user_embeddings.kv")
    print(wv.most_similar('alex_garza'))


if __name__ == '__main__':
    main()
