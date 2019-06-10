import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
import pickle


def load_embeddings(types, paths, concatenate=True):
    """
    Load the embeddings for each user

    Args:
        types (list): list of embedding types to load ['node2vec', 'lda', 'cnn']
        paths (list): list of string paths to the pickle files containing the
            embeddings in the types list. paths[i] is the path to the embeddings
            corresponding to types[i]
        concatenate (bool): whether the embeddings should be concatenated. If
            False, len(types) numpy arrays will be returned will be returned.
    Returns:
        ordered_vocab (list): list of usernames
        embeddings (list): list of numpy array embeddings. If concatenate is
            True, list will contain only one array. Otherwise, it will contain
            an array for each type
    """
    embeddings_dicts = []
    # Read all dictionaries from their pickle files
    for dict_path in paths:
        with open(dict_path, 'rb') as f:
            embeddings_dicts.append(pickle.load(f))
    # print(embeddings_dicts)
    # Determine combined embedding size
    embedding_size = 0
    embedding_sizes = []
    for d in embeddings_dicts:
        size = list(d.items())[0][1].shape[-1]
        embedding_size += size
        embedding_sizes.append(size)
    # Store all embeddings into a single base dictionary by name
    base_dict = dict(embeddings_dicts[0])
    for d_idx in range(1, len(embeddings_dicts)):
        dict_2 = embeddings_dicts[d_idx]
        for k in dict_2.keys():
            if k in base_dict:
                base_dict[k] = np.concatenate([base_dict[k], dict_2[k]])
    # Remove entries in the joined dictionary that have small embeddings
    filtered_dict = {k:v for k,v in base_dict.items() if v.size == embedding_size}
    # Extract vocab of usernames
    usernames = list(filtered_dict.keys())
    # Sort vocab of usernames
    ordered_vocab = sorted(usernames)
    # Extract np array of embeddings from each dict
    embeddings = []
    for i, d in enumerate(embeddings_dicts):
        N, D = len(ordered_vocab), embedding_sizes[i]
        embedding = np.zeros((N, D))
        for idx in range(N):
            embedding[idx] = d[ordered_vocab[idx]]
        embeddings.append(embedding)
    if concatenate:
        embeddings = [np.hstack(tuple(embeddings))]
        assert(embeddings[0].shape == (N, embedding_size))
        print(embeddings[0].shape)
    return ordered_vocab, embeddings
