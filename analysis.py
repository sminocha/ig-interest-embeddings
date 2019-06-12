from sklearn.cluster import KMeans
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import networkx as nx
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import collections
import matplotlib.pyplot as plt

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

    with open('cnn_embeddings.pkl', 'rb') as f:
        ret_di = pickle.load(f)
        print(ret_di['1misssmeis'].shape)
        # print(sorted(list(ret_di.keys())))

    with open('lda_embeddings.pkl', 'rb') as f:
        ret_di = pickle.load(f)
        print(ret_di['1misssmeis'])

    with open('pool_embeddings.pkl', 'rb') as f:
        ret_di = pickle.load(f)
        print(ret_di['1misssmeis'].shape)

    # # Load embeddings
    #
    # wv = KeyedVectors.load_word2vec_format("user_embeddings.kv")
    # print(wv.most_similar('alex_garza'))

def sample_user(embedding_dict):
    username, _ = random.choice(list(embedding_dict.items()))
    return username


def get_most_similar_users(name, embeddings_dict, key_username, num_similar=1):
    """Get top num_similar similar embeddings"""
    similarities = []
    for username in embeddings_dict.keys():
        if username != key_username:
            sim = cosine_similarity(embedding_dict[key_username].reshape(1,-1), embeddings_dict[username].reshape(1, -1))
            similarities.append((username, sim.flatten()[0]))
    # Sort similarities
    sorted_sims = sorted(similarities, key=lambda s: s[1], reverse=True)
    return sorted_sims[:num_similar]


def plot_degree_distribution(name, g):
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram for {} embeddings".format(name))
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()
    plt.clf()

def get_avg_cluster_coeff(g):
    cluster_coeffs = nx.clustering(g)
    N = len(cluster_coeffs)
    avg = 0.0
    for node, coeff in cluster_coeffs.items():
        avg += coeff
    return avg / N


def get_largest_smallest_cc(g):
    ccs = list(nx.connected_components(g))
    largest, smallest = max(ccs, key=len), min(ccs, key=len)
    return largest, smallest

def get_average_shortest_path_length(g):
    avg = 0.0
    ccs = list(nx.connected_component_subgraphs(g))
    N = len(ccs)
    for g_p in ccs:
        if nx.is_connected(g_p):
            avg += nx.average_shortest_path_length(g_p)
    return avg / N

def safe_diameter(g):
    eccentricities = nx.eccentricity(g)
    # Remove node eccentricities that are inf
    filtered_ecc = [v for k, v in eccentricities.items() if v < np.inf]
    return max(filtered_ecc)

def inv_euc(u, v):
    return 1. / (1. + euclidean(u, v))


def report_stats(g):
    communities = nx.community.greedy_modularity_communities(g)
    # diameter = safe_diameter(g)
    avg_clustering_coeff = get_avg_cluster_coeff(g)
    avg_shortest_path = get_average_shortest_path_length(g)
    largest_cc, smallest_cc = get_largest_smallest_cc(g)

    print("  num_communities: {}".format(len(communities)))
    # print("  diameter: {}".format(diameter))
    print("  avg_shortest_path: {}".format(avg_shortest_path))
    print("  avg_clustering_coeff: {}".format(avg_clustering_coeff))
    print("  largest_cc: {}".format(len(largest_cc)))
    print("  smallest_cc: {}".format(len(smallest_cc)))

def analyze_embedding_graph(name, embedding_dict, user_vocab, sim_thresh=0.6):
    """Construct a similarity graph where u1 is linked to u2 (and vice versa)
    if cosine sim of u1 and u2 > sim_thresh (-1 < cosine sim < 1 )"""
    N = len(user_vocab)
    print("Constructing graph from {} users".format(N))
    g = nx.Graph()
    added_users = set()
    for i in range(N):
        user1 = user_vocab[i]
        if user1 not in added_users:
            g.add_node(user1)
        for j in range(i+1, N):
            user2 = user_vocab[j]
            if user2 not in added_users:
                g.add_node(user2)
            if dist_func(embedding_dict[user1].reshape(1,-1), embedding_dict[user2].reshape(1,-1)) > sim_thresh:
                g.add_edge(user1, user2)

    # Report graphs stats
    plot_degree_distribution(name, g)
    print("\nGraph structure for: {}".format(name))
    report_stats(g)



if __name__ == '__main__':
    # main()
    # Open embeddings
    node2vec_embeddings = None
    lda_embeddings = None
    cnn_embeddings = None
    pool_embeddings = None
    with open('node2vec_embeddings.pkl', 'rb') as f:
        node2vec_embeddings = pickle.load(f)

    with open('lda_embeddings.pkl', 'rb') as f:
        lda_embeddings = pickle.load(f)

    with open('cnn_embeddings.pkl', 'rb') as f:
        cnn_embeddings = pickle.load(f)
        # print("Euc dist: {}".format(euclidean(cnn_embeddings['sejkko'], cnn_embeddings['mo'])))
        # print("Euc dist: {}".format(inv_euc(cnn_embeddings['sejkko'], cnn_embeddings['mo'])))

    with open('pool_embeddings_512_conctractive.pkl', 'rb') as f:
        pool_embeddings = pickle.load(f)

    user_vocab = sorted(list(pool_embeddings.keys()))
    # Sample user*
    key_username = 'sejkko' #sample_user(pool_embeddings)
    # For each set of embeddings:
    # Find similar users to user*
    for embedding_pair in [('node2vec', node2vec_embeddings), ('lda', lda_embeddings), ('cnn', cnn_embeddings), ('pooling', pool_embeddings)]:
        embedding_name, embedding_dict = embedding_pair
        most_similar = get_most_similar_users(embedding_name, mbedding_dict, key_username, num_similar=1)
        print ("Most similar user in {} embedding space has similarity {}".format(embedding_name, most_similar[0][1]))
        # Construct similarity graph and get structure of graph
        analyze_embedding_graph(embedding_name, embedding_dict, user_vocab)

    # Test random graph
