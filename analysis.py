from sklearn.cluster import KMeans
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import networkx as nx
import pickle


def run_kmeans(embeddings, k=10):
    kmeans = KMeans(n_clusters=k).fit(embeddings)
    labels = kmeans.labels_
    return labels


def main():
    # Count communities
    # g = nx.readwrite.edgelist.read_edgelist('user_user_graph.txt')
    # communities = nx.community.greedy_modularity_communities(g)
    # print(len(communities))
    with open('node2vec_embeddings.pkl', 'rb') as f:
        ret_di = pickle.load(f)
        print(type(ret_di['theworldguru']))

    with open('cnn_embeddings_2.pkl', 'rb') as f:
        ret_di = pickle.load(f)
        # print(ret_di['1misssmeis'].shape)
        # print(sorted(list(ret_di.keys())))

    with open('lda_embeddings.pkl', 'rb') as f:
        ret_di = pickle.load(f)
        print(ret_di['1misssmeis'])

    # # Load embeddings
    #
    # wv = KeyedVectors.load_word2vec_format("user_embeddings.kv")
    # print(wv.most_similar('alex_garza'))


if __name__ == '__main__':
    main()
